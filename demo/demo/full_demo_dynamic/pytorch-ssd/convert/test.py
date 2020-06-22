# import tensorflow.compat.v1 as tf1
import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np 
import torch
from vision.nn.mobilenet_v2 import MobileNetV2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_file')
    args = parser.parse_args()

    mb_net = MobileNetV2(width_mult=1.0, use_batch_norm=True, onnx_compatible=False).features
    filename = '../models/mb2-imagenet-71_8.pth'
    mb_net.load_state_dict(torch.load(filename, map_location=lambda storage, loc:storage), strict=True)
    shared_layers = mb_net[0:6].eval()
    dummy_input = torch.FloatTensor(10,3,300,300).normal_()
    pytorch_out = shared_layers(dummy_input).detach().numpy()

    #forward pass using TFLite model
    interpreter = tf.lite.Interpreter(model_path=args.tflite_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], dummy_input.numpy())
    interpreter.invoke()
    tf_out = interpreter.get_tensor(output_details[0]['index'])

    abs_diff = np.abs(tf_out - pytorch_out)
    print(f'max abs diff: {abs_diff.max()}')
    print(f'min abs diff: {abs_diff.min()}')
    print(f'avg abs diff: {abs_diff.mean()}')
