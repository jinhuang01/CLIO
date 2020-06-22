#!/bin/sh
LABELSFILE="voc-model-labels.txt"
python3 -u run_from_camera.py\
	--net multi-headed\
	--trained_model models/Original_gray/multi-headed-Epoch-199-Loss-2.9512695020244966.pth\
	--label_file $LABELSFILE\
