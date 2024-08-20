import torch
import pytorch_lightning as pl
from audiossl.models.ATST.audio_transformer import AST_base
from audiossl.models.ATST.utils import load_pretrained_weights
from audiossl.models.ATST.transform import FreezingTransform

class ATSTPredModule(pl.LightningModule):
    """This module has been modified for frame-level prediction"""

    def __init__(self, pretrained_ckpt_path=None, dataset_name="as_strong", **kwargs):
        super().__init__()
        self.encoder = AST_base(use_cls=True, **kwargs)
        '''if pretrained_ckpt_path is not None:
            load_pretrained_weights(self.encoder, pretrained_ckpt_path, checkpoint_key="teacher")'''
        checkpoint = torch.load(pretrained_ckpt_path, map_location="cpu")
        print(f"Load pre-trained checkpoint from: {pretrained_ckpt_path}" )
        checkpoint_model = checkpoint['teacher'] 
        state_dict = self.encoder.state_dict()
        selected_keys = state_dict.keys()
        checkpoint_model = {key.replace('module.backbone.', ''): value for key, value in checkpoint_model.items()}
        selected_params = {k: v for k, v in checkpoint_model.items() if k in selected_keys}  # 选择需要加载的参数
        for k in selected_keys:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
       
        # load pre-trained model
        self.encoder.load_state_dict(selected_params, strict=True)
        self.embed_dim = self.encoder.embed_dim
        self.transform = FreezingTransform(max_len=10)
        self.last_layer = dataset_name != "as_strong"

    def forward(self, batch):
        (x, length), y = batch ##x: torch.Size([256, 64, 1001])
        x = x.unsqueeze(1) ##x: torch.Size([256, 1, 64, 1001])
        x = self.encoder.get_intermediate_layers(
            x,
            length,
            1
        )
        x = [item[:, 1:, :] for item in x]
        x = torch.concat(x, dim=-1)
        return x, y

    def finetune_mode(self):
        if self.last_layer:
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.blocks):
                if i == len(self.encoder.blocks) - 1:
                    for n, p in layer.named_parameters():
                        p.requires_grad = True
            # Unfreeze last norm layer
            for n, p in self.encoder.norm.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.encoder.named_parameters():
                if "mask_embed" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def finetune_mannual_train(self):
        if self.last_layer:
            for i, layer in enumerate(self.encoder.blocks):
                if i == len(self.encoder.blocks) - 1:
                    layer.train()
            self.encoder.norm.train()
        else:
            self.train()

if __name__ == '__main__':
    input_tdim = 1001
    fake_x = torch.zeros([1, input_tdim, 64])
    fake_x[:, 16:32, :] = 1
    model = ATSTPredModule("/public/home/03455/ZYY/ATST-RCT-main/pretraining/ATST/base.ckpt", "dcase")
    output = model(fake_x)
    print('111111111111111:',output)
    print('2222222222:',output.shape)