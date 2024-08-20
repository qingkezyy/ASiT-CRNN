import torch
import torch.nn.functional
import torch.nn as nn
import pytorch_lightning as pl
from .models.ssast import ASTModel
 
class SSASTModel(ASTModel):
    def __init__(self, label_dim=527, fshape=128, tshape=2, fstride=128, tstride=2, input_fdim=128, input_tdim=1024, model_size='base', pretrain_stage=True, load_pretrained_mdl_path=None):
        super(SSASTModel, self).__init__(label_dim, fshape, tshape, fstride, tstride, input_fdim, input_tdim, model_size, pretrain_stage, load_pretrained_mdl_path)
        self.feat_mean = nn.AvgPool2d([8, 1], padding=[1, 0])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        # Rewrite ASTModel forward
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = x[:, self.cls_token_num:, :].reshape(B, 8, -1, 768)
        # average output of all tokens except cls token(s)
        x = x.permute(0, 3, 1, 2)
        x = self.feat_mean(x)
        x = x.squeeze(-2).transpose(1, 2)
        return x

class SSAST(torch.nn.Module):
    def __init__(self, ssast_path, *args, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.ssast = SSASTModel(label_dim=1, fshape=16, tshape=16, fstride=16, tstride=16, 
                                 input_fdim=128, input_tdim=998, model_size="base", pretrain_stage=False,
                                 load_pretrained_mdl_path=ssast_path)

    def forward(self, ssast_feat):

        ssast_x = self.ssast(ssast_feat)   
                
        return ssast_x
