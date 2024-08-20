import torch
import pytorch_lightning as pl
from .models.atst import AST_base
from desed_task.audiossl.model.ATST.transform import FreezingTransform

class ATST(pl.LightningModule):
    """This module has been modified for frame-level prediction"""

    def __init__(self, pretrained_ckpt_path=None, dataset_name="as_strong", **kwargs):
        super().__init__()
        self.encoder = AST_base(use_cls=True, **kwargs)
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

    def forward(self, clip_atst_feat):
        clip_atst_feat = clip_atst_feat.unsqueeze(1) 
        clip_atst_x = self.encoder.get_intermediate_layers(
            clip_atst_feat,
            None, ## length
            1
        )
        clip_atst_x = [item[:, 1:, :] for item in clip_atst_x]
        clip_atst_x = torch.concat(clip_atst_x, dim=-1)
        
        return clip_atst_x