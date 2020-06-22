#!/bin/sh
LABELSFILE="voc-model-labels.txt"
python3 -u run_from_camera.py\
	--net multi-headed\
	--trained_model models/multi-headed-Epoch-600-Loss-3.842284660954629.pth\
	--label_file $LABELSFILE\
