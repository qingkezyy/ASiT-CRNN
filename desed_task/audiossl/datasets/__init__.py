
from pathlib import Path
import os

from .registry import register_dataset,list_all_datasets,get_dataset

from .dcase import DCASEDataset

# [DCASE MARK] add a register for dcase dataset
@register_dataset("dcase", multi_label=True, num_labels=10, num_folds=1)
def create_dcase(config_path, split, transform=None, target_transform=None, unsup=False,return_key=False):
    assert split in ["train", "valid", "test"], "Dataset type: {} is not supported.".format(split)
    return DCASEDataset(config_path, split, transform=transform, target_transform=None, unsup=unsup)

