cd ../../../downstream
gpu_id="0"
arch="patchmaeast"

for lr in "1e-1"
do
    echo ${arch}, learning rate: ${lr}
    python train_dcase.py \
    --nproc ${gpu_id}, \
    --learning_rate ${lr} \
    --arch ${arch} \
    --prefix _lr_${lr} \
    --pretrained_ckpt_path "./comparison_models/ckpts/chunk_patch_75_12LayerEncoder.pt" \
    --dcase_conf "./utils_dcase/conf/patch_160.yaml"
done
