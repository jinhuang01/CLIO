#!/bin/sh
module load python3/current

DATAROOT="/home/csamplawski/pytorch-ssd/data/"
VAL2007="$DATAROOT/voc0712/test/VOC2007"
LABELSFILE="$DATAROOT/voc0712/voc-model-labels.txt"

srun --pty --gres=gpu:1 --partition=1080ti-short --mem=32GB python3 -u eval_ssd.py \
    --dataset_type voc \
    --dataset $VAL2007 \
    --net multi-headed\
	--trained_model $1 \
	--label_file $LABELSFILE \
	--widths 2 4 8\
	--shared_layer_conf 'halving_24'\
