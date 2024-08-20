import torch
import torch.nn as nn
from .models.asit import ASTModel

class ASiT(torch.nn.Module):
    def __init__(self, asit_path, *args, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.asit = ASTModel(finetune_from=asit_path, label_dim=10, fstride=16, tstride=16, input_fdim=128, input_tdim=998, patch_size=16, 
                             model_name='vit_base', verbose=True, keyword='teacher')
        # Average pooling
        self.feat_mean = nn.AvgPool1d(8, 8)

    def forward(self, asit_feat, n=1):

        asit_x = self.asit(asit_feat) ## asit_x: torch.Size([B, 512, 768])      
        
        asit_x = self.feat_mean(asit_x.transpose(-1, -2)).transpose(-1, -2)
                
        return asit_x
