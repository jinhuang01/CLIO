import torch
import sys
sys.path.append('..') #needed so we can access vision.nn from this dir
from vision.nn.mobilenet_v2 import MobileNetV2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file')
    args = parser.parse_args()

    mb_net = MobileNetV2(width_mult=1.0, use_batch_norm=True, onnx_compatible=False).features
    filename = '../models/mb2-imagenet-71_8.pth'
    mb_net.load_state_dict(torch.load(filename, map_location=lambda storage, loc:storage), strict=True)
    shared_layers = mb_net[0:6].eval() #need to eval so batchnorm works properly
    dummy_input = torch.FloatTensor(10,3,300,300).normal_()
    
    o_model = torch.onnx._export(
        shared_layers, 
        dummy_input, 
        args.onnx_file,
        export_params=True,
        input_names=['test_input'], 
        output_names=['test_output']
    )
    print('converted pytorch model to onnx')
