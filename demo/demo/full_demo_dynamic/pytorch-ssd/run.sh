#!/bin/sh
LABELSFILE="voc-model-labels.txt"
python3 -u run_from_camera_class.py\
	--net multi-headed\
	--trained_model models/multi-headed-h24-dynamic-Epoch-600.pth \
	--label_file $LABELSFILE\
