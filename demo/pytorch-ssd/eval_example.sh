#!/bin/sh
#module load python3/current

#DATAROOT="/home/csamplawski/pytorch-ssd/data/"
LABELSFILE="voc-model-labels.txt"
rm frames/*boxed*
python3 -u run_ssd_example.py\
	--net multi-headed\
	--trained_model $1\
	--label_file $LABELSFILE\
	--input $2

#for f in images/frames/outfile*png; do
#done

