import numpy as np
import torch
import torch.nn.functional
import torch.nn as nn
import pytorch_lightning as pl
from .models.audioMAE import vit_base_patch16, PatchEmbed_new

class AudioMAEModel(pl.LightningModule):
    def __init__(self, pretrained_path):
        super(AudioMAEModel, self).__init__()
        # Load pre-trained model
        self.encoder = vit_base_patch16()
        self.encoder.patch_embed = PatchEmbed_new(img_size=(1024, 128), patch_size=(16,16), in_chans=1, embed_dim=768, stride=16) # no overlap. stride=img_size=16
        num_patches = self.encoder.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        self.encoder.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False)  # fixed sin-cos embedding
        
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        print(f"Load pre-trained checkpoint from: {pretrained_path}" )
        checkpoint_model = checkpoint['model']
        state_dict = self.encoder.state_dict()

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # load pre-trained model
        self.encoder.load_state_dict(checkpoint_model, strict=False)
        self.feat_mean = nn.AvgPool1d(8, 8)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder.patch_embed(x)
        B, T, _ = x.shape
        x = x + self.encoder.pos_embed[:, 1: T + 1, :]
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        
        x = self.encoder.pos_drop(x)

        # apply Transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        
        x = x[:, 1:, :]
        x = self.encoder.norm(x)
        x = self.feat_mean(x.transpose(-1, -2)).transpose(-1, -2)
        
        return x
    
    
class AudioMAE(torch.nn.Module):
    def __init__(self, audioMAE_path, *args, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.audioMAE = AudioMAEModel(audioMAE_path)

    def forward(self, audioMAE_feat):

        audioMAE_x = self.audioMAE(audioMAE_feat)   
                
        return audioMAE_x



