# ASiT-CRNN
This repo includes the official implementations of "ASiT-CRNN: A method for sound event detection with fine-tuning of self-supervised pre-trained ASiT-based model".

This work is submitted to ????. 

This work is highly related to [ASiT-CRNN](https://????). Please check these works if you want to find out the principles of the ASiT-CRNN.

# Notice

0. **Strong val dataset**: This dataset meta files are now updated to the repo.

1. For better understanding of SED community, our codes are developed based on the [baseline codes](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2022_task4_baseline) of [DCASE2022 challenge task 4](https://dcase.community/). Therefore, the training progress is build under [`pytorch-lightning`](https://lightning.ai/).

2. We report both DESED development set results. The external set is the extra data extracted from the [AudioSet](http://research.google.com/audioset/)/[AudioSetStrong](https://research.google.com/audioset/download_strong.html). Please do not mess it with the 3000+ strongly labelled real audio clips from the AudioSet.




# Get started

0. To reproduce our experiments, please first ensure you have the full DESED dataset (including 3000+ strongly labelled real audio clips from the AudioSet).

1. Ensure you have the correct environment. The environment of this code is the same as the DCASE 2022 baseline, please refer to their [docs/codes](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2022_task4_baseline) to configure your environment.

2. Download the pretrained [ASiT checkpoint (ASiT_16kHz.pth)](https://github.com/ASiT). Noted that this checkpoint is fine-tuned by the AudioSet-2M.

3. Clone the ASiT-CRNN codes by:

```
git clone https://github.com/qingkezyy/ASiT-CRNN.git
```

4. Install our desed_task package by:

```
cd ASiT-CRNN
```

```
pip install -e .
```

5. Change all required paths in `train/local/confs/asit_crnn.yaml` to your own paths. Noted that the pretrained ASiT checkpoint path should be changed in **both** files.

If you want to train start training ssl_crnn, change all required paths in `train/local/confs/ssl_crnn.yaml` to your own paths.

6. Start training asit_crnn by:

```
python train_asit_crnn.py --gpus YOUR_DEVICE_ID,
```

If you want to train start training ssl_crnn by:

```
python train_ssl_crnn.py --gpus YOUR_DEVICE_ID,
```

# Citation

If you want to cite this paper:

```
@article{????,

      title={ASiT-CRNN: A method for sound event detection with fine-tuning of self-supervised pre-trained ASiT-based model}, 

      author={Yueyang Zheng},

      journal={arXiv preprint arXiv:????},

      year={2024}
}

```
