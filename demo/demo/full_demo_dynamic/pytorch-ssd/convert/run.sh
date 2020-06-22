#!/bin/sh
python3 torch2onnx.py --onnx_file test.onnx
python3 onnx2pb.py --onnx_file test.onnx --pb_file test.pb
python3 pb2tflite.py --pb_file test.pb --tflite_file test.tflite
python3 test.py --tflite_file test.tflite
