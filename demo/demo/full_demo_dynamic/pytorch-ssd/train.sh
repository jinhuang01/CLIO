#!/bin/sh
module load python3/current

DATAROOT="/home/csamplawski/pytorch-ssd/data/"
TRAIN2012="$DATAROOT/voc0712/VOC2012/"
TRAIN2007="$DATAROOT/voc0712/VOC2007"
VAL2007="$DATAROOT/voc0712/test/VOC2007"

#SAVEDIR="/mnt/nfs/scratch1/csamplawski/ssd/224_clamped_simple_nofreeze_128_mean_std"
#SAVEDIR="/mnt/nfs/scratch1/csamplawski/ssd/224_clamped_simple_5freeze_128_mean_std_finetune_person_tv_chair"
#SAVEDIR="/tmp"
CONF='halving_24'
SAVEDIR="/mnt/nfs/scratch1/csamplawski/ssd/dynamic_width_$CONF"
#SAVEDIR="/tmp"

srun --pty --gres=gpu:1 --partition=titanx-long --mem=16GB python3 -u train_ssd.py \
    --dataset_type voc \
    --datasets $TRAIN2007 $TRAIN2012 \
    --validation_dataset $VAL2007 \
    --net multi-headed \
	--checkpoint_folder $SAVEDIR \
    --base_net models/mb2-imagenet-71_8.pth  \
    --scheduler cosine \
    --lr 0.01 \
    --batch_size 32\
    --t_max 200 \
    --validation_epochs 20\
    --num_epochs 1000\
	--widths 2 4 8\
	--shared_layer_conf $CONF\
	#--gpu-id $CUDA_VISIBLE_DEVICES\
