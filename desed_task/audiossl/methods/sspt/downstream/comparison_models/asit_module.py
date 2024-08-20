import torch
import csv
import json
import os
import torchaudio
import numpy as np
import torch.nn.functional
import torch.nn as nn
import pytorch_lightning as pl
from desed_task.audiossl.methods.sspt.downstream.comparison_models.models.asit_models import ASTModel

audio_configs = {
    "n_mels": 128,
    "num_frames":992,
    "sr": 16000,
    "norm_mean": -6.030435443767988, 
    "norm_std": 4.102992546322562, 
}

class ASiTPredModule(pl.LightningModule):
    def __init__(self, pretrained_path, dataset_name="as_strong") -> None:
        super().__init__()
        self.encoder = ASTModel(finetune_from=pretrained_path, label_dim=1, fstride=16, tstride=16, 
                                  input_fdim=128, input_tdim=992, patch_size=16, 
                                  model_name='vit_base', verbose=True, keyword='teacher')
        self.embed_dim = 768
        self.last_layer = dataset_name != "as_strong"
        
    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder(x)
        return x, y
    

    @staticmethod
    def transform(wav):

        wav = (wav - wav.mean()).unsqueeze(0)   # add fake channel

        # LogFBank
        fbank = torchaudio.compliance.kaldi.fbank(
            wav, 
            htk_compat=True, 
            sample_frequency=audio_configs["sr"], 
            use_energy=False, 
            window_type='hanning', 
            num_mel_bins=audio_configs["n_mels"], 
            dither=0.0, 
            frame_shift=10
            )

        fbank = (fbank - audio_configs['norm_mean']) / (audio_configs['norm_std'] * 2)
        
        return fbank, fbank.shape[0]
    

    def finetune_mode(self):
        if self.last_layer:
            print("Enable last layer freezing")
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.v.blocks):
                for n, p in layer.named_parameters():
                    if i == len(self.encoder.v.blocks) - 1: 
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            # Unfreeze last norm layer
            for n, p in self.encoder.v.norm.named_parameters():
                p.requires_grad = True
        else:
            print("Enable all layer freezing")
            for n, p in self.named_parameters():
                if (".v.head" in n) or (".mlp_head." in n):
                    p.requires_grad = False
                else:
                    p.requires_grad = True


    def finetune_mannual_train(self):
        if self.last_layer:
            for i, layer in enumerate(self.encoder.v.blocks):
                if i == len(self.encoder.v.blocks) - 1:
                    layer.train()
            self.encoder.v.norm.train()        
        else:
            self.train()
