import torch
import torchaudio

import pytorch_lightning as pl
import torch.nn as nn
from audiossl.methods.sspt.downstream.comparison_models.models.mae_ast_patch import MAE_AST

audio_configs = {
    "feature_type": "fbank", 
    "sample_rate": 16000,
    "max_sample_size": 250000,
    "min_sample_size": 32000,
    "feature_rate": 100,
    "feature_dim": 128,
    "normalize": False, # must be consistent with extractor
    "deltas": False
}

class MAEASTModel(MAE_AST):
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

class PatchMAEASTPredModule(pl.LightningModule):
    def __init__(self, pertrained_ckpt_path, dataset_name="as_strong") -> None:
        super(PatchMAEASTPredModule, self).__init__()
        self.encoder = MAEASTModel()
        self.embed_dim = 768
        load_weigts = torch.load(pertrained_ckpt_path)
        state_dicts = load_weigts["model"]
        self.encoder.load_state_dict(state_dict=state_dicts, strict=True)
        self.last_layer = dataset_name != "as_strong"

    def forward(self, batch):
        (x, length), y = batch
        x, _ = self.encoder(x)
        return x, y

    def transform(self, wav):
        wav = wav.unsqueeze(0)
        feat = torchaudio.compliance.kaldi.fbank(
                waveform=wav,
                sample_frequency=audio_configs["sample_rate"],
                use_energy=False,
                num_mel_bins=audio_configs["feature_dim"],
                frame_shift=10
            )
        feat = feat[:, :audio_configs["feature_dim"]]
        return feat, feat.shape[0]
    
    def finetune_mode(self):
        if self.last_layer:
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i == len(self.encoder.encoder.layers) - 1:
                    for n, p in layer.named_parameters():
                        p.requires_grad = True
        else:
            for n, p in self.named_parameters():
                if (".decoder" in n) or \
                (".final_proj" in n) or \
                ("encoder_mask_emb" in n) or \
                ("encoder.layer_norm" in n):
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def finetune_mannual_train(self):
        if self.last_layer:
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i == len(self.encoder.encoder.layers) - 1:
                    layer.train()
        else:
            self.train()