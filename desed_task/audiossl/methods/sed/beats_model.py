import torch
import pytorch_lightning as pl
import torch.nn as nn
from desed_task.audiossl.methods.sed.models.beats.BEATs import BEATsModule, BEATsConfig

class BEATs(torch.nn.Module):
    def __init__(self, pertrained_ckpt_path) -> None:
        super(BEATs, self).__init__()
        checkpoint = torch.load(pertrained_ckpt_path)
        cfg = BEATsConfig(checkpoint["cfg"])
        # cfg.set("layer_wise_gradient_decay_ratio", 0.75)
        cfg.set("predictor_class", 10)
        self.encoder = BEATsModule(cfg)
        self.encoder.load_state_dict(checkpoint["model"])
        self.feat_mean = nn.AvgPool1d(8, 8)
        self.embed_dim = 768

    def forward(self, beats_feat):

        beats_feat = self.encoder.extract_features(beats_feat, None)[0]
        beats_x = self.feat_mean(beats_feat.transpose(-1, -2)).transpose(-1, -2)

        return beats_x