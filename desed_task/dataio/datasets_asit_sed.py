from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import torchaudio
import random
import torch
import glob
import h5py
import torch.nn as nn
from pathlib import Path
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


def to_mono(mixture, random_ch=False):

    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture


def pad_audio(audio, target_len, fs):
    
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )

        padded_indx = [target_len / len(audio)]
        onset_s = 0.000
    
    elif len(audio) > target_len:
        
        rand_onset = random.randint(0, len(audio) - target_len)
        audio = audio[rand_onset:rand_onset + target_len]
        onset_s = round(rand_onset / fs, 3)

        padded_indx = [target_len / len(audio)] 
    else:

        onset_s = 0.000
        padded_indx = [1.0]

    offset_s = round(onset_s + (target_len / fs), 3)
    return audio, onset_s, offset_s, padded_indx

def process_labels(df, onset, offset):
    
    
    df["onset"] = df["onset"] - onset 
    df["offset"] = df["offset"] - onset
        
    df["onset"] = df.apply(lambda x: max(0, x["onset"]), axis=1)
    df["offset"] = df.apply(lambda x: min(10, x["offset"]), axis=1)

    df_new = df[(df.onset < df.offset)]
    
    return df_new.drop_duplicates()


def read_audio(file, multisrc, random_channel, pad_to):
    
    mixture, fs = torchaudio.load(file)
    
    if not multisrc:
        mixture = to_mono(mixture, random_channel)

    if pad_to is not None:
        mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, pad_to, fs)
    else:
        padded_indx = [1.0]
        onset_s = None
        offset_s = None

    mixture = mixture.float()
    return mixture, onset_s, offset_s, padded_indx


class SEDTransform:
    def __init__(self, feat_params):
        self.transform = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
    def __call__(self, x):
        return self.transform(x)

audio_configs = {
    "n_mels": 128,
    "num_frames":1024,
    "sr": 16000,
    "norm_mean": -4.2677393,
    "norm_std": 4.5689974,
}

class ASiTTransform:
    def __init__(self):
        self.transform = ASiTTransform.transform

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
        n_frames = fbank.shape[0]
        p = audio_configs['num_frames'] - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:audio_configs['num_frames'], :]

        fbank = (fbank - audio_configs['norm_mean']) / (audio_configs['norm_std'] * 2)

        return fbank
    
    def __call__(self, x):
        # to_db applied in the trainer files
        return self.transform(x)

class StronglyAnnotatedSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feat_params=None,
        forbidden_list=None,

    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.sed_transform = SEDTransform(feat_params)
        self.asit_transform = ASiTTransform()
        tsv_entries = tsv_entries.dropna()
        
        examples = {}
        if forbidden_list is not None:
            forbidden_list = open(forbidden_list, "r").readlines()
            forbidden_list = [f.strip() for f in forbidden_list]
        else:
            forbidden_list = []
        for i, r in tsv_entries.iterrows():
            filename = r["filename"].split(".")[0]
            if filename in forbidden_list:
                continue
            if r["filename"] not in examples.keys():
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": [],
                }
                if not np.isnan(r["onset"]):
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                        }
                    )
            else:
                if not np.isnan(r["onset"]):
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                        }
                    )
        # we construct a dictionary for each example
        self.examples = examples
        self.examples_list = list(examples.keys())
        print("Number of examples: ", len(self.examples_list))
    def __len__(self):
        return len(self.examples_list)

    def __getitem__(self, item):
        c_ex = self.examples[self.examples_list[item]]
        mixture, onset_s, offset_s, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to
        )

        # labels
        labels = c_ex["events"]
        
        # to steps
        labels_df = pd.DataFrame(labels)
        labels_df = process_labels(labels_df, onset_s, offset_s)
        
        # check if labels exists:
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        else:
            strong = self.encoder.encode_strong_df(labels_df)
            strong = torch.from_numpy(strong).float()

        # sed_feat = self.sed_transform(mixture)
        asit_feat = self.asit_transform(mixture)
        out_args = [mixture, asit_feat, strong.transpose(0, 1), padded_indx]
        if self.return_filename:
            out_args.append(c_ex["mixture"])
        return out_args


class WeakSet(Dataset):

    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feat_params=None

    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.sed_transform = SEDTransform(feat_params)
        self.asit_transform = ASiTTransform()
        examples = {}
        for i, r in tsv_entries.iterrows():

            if r["filename"] not in examples.keys():
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": r["event_labels"].split(","),
                }

        self.examples = examples
        self.examples_list = list(examples.keys())
        print(len(self.examples))

    def __len__(self):
        return len(self.examples_list)

    def __getitem__(self, item):
        file = self.examples_list[item]
        c_ex = self.examples[file]

        mixture, _, _, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to
        )
        
        # labels
        labels = c_ex["events"]
        # check if labels exists:
        max_len_targets = self.encoder.n_frames
        weak = torch.zeros(max_len_targets, len(self.encoder.labels))
        if len(labels):
            weak_labels = self.encoder.encode_weak(labels)
            weak[0, :] = torch.from_numpy(weak_labels).float()
        # sed_feat = self.sed_transform(mixture)
        asit_feat = self.asit_transform(mixture)
        out_args = [mixture, asit_feat, weak.transpose(0, 1), padded_indx]

        if self.return_filename:
            out_args.append(c_ex["mixture"])

        return out_args


class UnlabeledSet(Dataset):
    def __init__(
        self,
        unlabeled_folder,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feat_params=None
    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs if pad_to is not None else None 
        self.examples = glob.glob(os.path.join(unlabeled_folder, "*.wav"))
        print(len(self.examples))
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.sed_transform = SEDTransform(feat_params)
        self.asit_transform = ASiTTransform()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        c_ex = self.examples[item]
        mixture, _, _, padded_indx = read_audio(
            c_ex, self.multisrc, self.random_channel, self.pad_to
        )

        max_len_targets = self.encoder.n_frames
        strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        # sed_feat = self.sed_transform(mixture)
        asit_feat = self.asit_transform(mixture)
        out_args = [mixture, asit_feat, strong.transpose(0, 1), padded_indx]

        if self.return_filename:
            out_args.append(c_ex)

        return out_args
