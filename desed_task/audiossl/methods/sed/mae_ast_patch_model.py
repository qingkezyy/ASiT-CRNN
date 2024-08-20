import torch
import torchaudio
import os
import pytorch_lightning as pl
import torch.nn as nn
from .models.mae_ast import MAE_AST

class MAE_ASTModel(MAE_AST):
    def __init__(self):
        super().__init__()
        self.pad_matrix = torch.zeros(256, 998)
        self.feat_mean = torch.nn.AvgPool1d(8, 8)
    
    def forward(self, source, mask=False, ret_conv=False, output_layer=None):
        res = super().forward(
            source,
            padding_mask=self.pad_matrix.to(source).bool(),
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        feature = feature.transpose(-1, -2)
        feature = self.feat_mean(feature).transpose(-1, -2)
        return feature, res["padding_mask"]


class MAE_AST(torch.nn.Module):
    def __init__(self, mae_ast_path, *args, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.mae_ast = MAE_ASTModel()
        self.embed_dim = 768
        load_weigts = torch.load(mae_ast_path)
        state_dicts = load_weigts["model"]
        self.mae_ast.load_state_dict(state_dict=state_dicts, strict=True)

    def forward(self, mae_ast_feat):

        mae_ast_x,_ = self.mae_ast(mae_ast_feat)   
                
        return mae_ast_x
