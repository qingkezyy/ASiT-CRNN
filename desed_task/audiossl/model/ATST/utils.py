
from desed_task.audiossl import datasets
from desed_task.audiossl.datasets import Nsynth,Urbansound8k,SpeakerClassifiDataset,IEMOCAPDataset
from pathlib import Path
import os
import torch
import sys
import time
import datetime
import numpy as np





def load_dataset(dataset_name,data_path,fold,transform):
    if dataset_name == "spcv2":
        dataset_train = datasets.LMDBDataset(data_path,split="train",subset=200000,transform=transform) 
        dataset_val = datasets.LMDBDataset(data_path,split="valid",subset=200000,transform=transform) 
        dataset_test = datasets.LMDBDataset(data_path,split="eval",subset=200000,transform=transform) 
    elif dataset_name == "nsynth":
        dataset_train=Nsynth(data_path,split="train",transform=transform)
        dataset_val=Nsynth(data_path,split="valid",transform=transform)
        dataset_test=Nsynth(data_path,split="test",transform=transform)
    elif dataset_name == "us8k":
        dataset_train=Urbansound8k(data_path,split="train",valid_fold=fold,transform=transform)
        dataset_val=Urbansound8k(data_path,split="test",valid_fold=fold,transform=transform)
        dataset_test=Urbansound8k(data_path,split="test",valid_fold=fold,transform=transform)

    elif dataset_name == "voxceleb1":
        dataset_train=SpeakerClassifiDataset("train",
                                                       Path(data_path),
                                                       Path(os.path.join(data_path,"iden_split.txt")),
                                                       transform=transform)
        dataset_val=SpeakerClassifiDataset("dev",
                                                       Path(data_path),
                                                       Path(os.path.join(data_path,"iden_split.txt")),
                                                       transform=transform)
        dataset_test=SpeakerClassifiDataset("test",
                                                       Path(data_path),
                                                       Path(os.path.join(data_path,"iden_split.txt")),
                                                       transform=transform)
    elif dataset_name == "iemocap":
        dataset_train=IEMOCAPDataset(Path(data_path),
                                            Path(os.path.join(data_path,"meta_data","Session{}".format(fold),"train_meta_data.json")),
                                            transform=transform)
        dataset_val=IEMOCAPDataset(Path(data_path),
                                            Path(os.path.join(data_path,"meta_data","Session{}".format(fold),"test_meta_data.json")),
                                            transform=transform)
        dataset_test=dataset_val
    elif dataset_name == "audioset_b":
        dataset_train=datasets.LMDBDataset(data_path,split="train",subset=3000000,transform=transform)
        dataset_val=datasets.LMDBDataset(data_path,split="valid",transform=transform)
        dataset_test=datasets.LMDBDataset(data_path,split="eval",transform=transform)

    elif dataset_name == "audioset":
        dataset_train=datasets.LMDBDataset(data_path,split="train",subset=3000000,transform=transform)
        dataset_val=datasets.LMDBDataset(data_path,split="valid",transform=transform)
        dataset_test=datasets.LMDBDataset(data_path,split="eval",transform=transform)
    else:
        raise NotImplementedError
    return dataset_train,dataset_val,dataset_test


MULTI_LABEL={
    "spcv2":False,
    "nsynth":False,
    "us8k":False,
    "voxceleb1":False,
    "iemocap":False,
    "audioset_b":True,
    "audioset":True
}

NUM_LABELS={
    "spcv2":35,
    "nsynth":11,
    "us8k":10,
    "voxceleb1":1251,
    "iemocap":4,
    "audioset_b":527,
    "audioset":527
}

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing import Any, List, Optional

def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Gather all tensors from several ddp processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.
    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)
    Return:
        gathered_result: list with size equal to the process group where
            ``gathered_result[i]`` corresponds to result tensor from process ``i``
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)


    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)


    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result

import numpy as np
from sklearn import metrics
class Metric:
    preds = []
    targets = []
    def __init__(self,mode="ACC"):
        self.mode=mode
        self.clear()
    def update(self,pred,target):
        self.preds.append(pred)
        self.targets.append(target)
    def clear(self):
        self.preds = []
        self.targets = []
    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        print(preds.shape,targets.shape)
        preds=gather_all_tensors(preds) 
        targets=gather_all_tensors(targets) 
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        print(preds.shape,targets.shape)
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        if self.mode == "mAP":
            mAPs =[]
            for i in range(preds.shape[-1]):
                mAPs.append(metrics.average_precision_score(targets[:,i],preds[:,i],average=None))
            mAPs = np.array(mAPs)
            _mAPs = mAPs[np.isnan(mAPs)]
            print(mAPs)
            print("============{}================= nans".format(len(_mAPs)))
            mAPs = mAPs[~np.isnan(mAPs)]
            mAP = np.mean(mAPs)
            return mAP
        else:
            acc = metrics.top_k_accuracy_score(targets,preds,k=1,labels=np.linspace(0,preds.shape[-1]-1,preds.shape[-1]))
            return acc

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        exit()


