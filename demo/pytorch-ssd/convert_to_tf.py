import torch
import torch.nn as nn
import numpy as np
import sys
import argparse
from vision.ssd.multi_headed_ssd import MultiHeadedSSD
from vision.ssd.config import multi_headed_config
import tensorflow as tf
import tensorflow.keras as K

def swap_channels(x, mode='torch2tf'):
    if mode == 'torch2tf':
        order = (0, 2, 3, 1)
    elif mode == 'tf2torch':
        order = (0, 3, 1, 2)
    else:
        print('unsupported mode')
        return x
    return x.transpose(order)

def torch2tf(torch_model, input_shape=(244,244,1)):
    weights = []
    inputs = K.layers.Input(shape=input_shape)
    out = inputs
    #out = K.layers.Lambda(lambda x: (x - 128.0) / 128.0)(inputs)
    # out = K.layers.Lambda(lambda x: x / 128.0)(out)
    for layer in torch_model:
        if isinstance(layer, nn.modules.activation.ReLU):
            out = K.layers.ReLU()(out)
        if isinstance(layer, nn.modules.pooling.MaxPool2d):
            out = K.layers.MaxPool2D(layer.kernel_size)(out)
        if isinstance(layer, nn.modules.conv.Conv2d):
            torch_weights = layer.weight.detach().numpy().transpose((2, 3, 1, 0))
            weights.append(torch_weights)
            tf_layer = K.layers.Conv2D(
                layer.out_channels, 
                kernel_size=layer.kernel_size, 
                strides=layer.stride, 
                padding='same', 
                use_bias=False
            )
            out = tf_layer(out)
    tf_model = K.Model(inputs, out)
    tf_model.set_weights(weights)
    return tf_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model", type=str)
    parser.add_argument("--tflite_file", type=str)
    args = parser.parse_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MultiHeadedSSD(21, width_mult=1.0, config=multi_headed_config, is_test=True, widths=[2,4,8],
            shared_layer_conf='halving_24')
    net.load(args.trained_model)
    net = net.eval()
    net = net.to(DEVICE)
    
    shared_layers = net.shared_layers
    # sd = shared_layers.state_dict()
    # sd['0.weight'] = torch.ones(32,1,3,3) + 0.5
    # shared_layers.load_state_dict(sd)
    tf_model = torch2tf(shared_layers)
    
    #check that the two versions produce the same output
    #torch_input = torch.FloatTensor(1, 1, 244, 244).normal_()
    torch_input = torch.ones(1, 1, 244, 244) * 255
    torch_input = (torch_input - 128.0) / 128.0

    tf_input = swap_channels(torch_input.numpy())
    
    #torch_out = shared_layers((torch_input - 128.0)/128.0).detach().numpy()
    torch_out = shared_layers(torch_input).detach().numpy()

    tf_out = tf_model(tf_input).numpy()
    tf_out = swap_channels(tf_out, 'tf2torch')

    diff = np.abs(tf_out - torch_out)
    print(diff.min(), diff.max(), diff.mean())

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()
    with open(args.tflite_file, 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=args.tflite_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], tf_input)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_details[0]['index'])
    tflite_out = swap_channels(tflite_out, 'tf2torch')

    diff = np.abs(tflite_out - tf_out)
    print(diff.min(), diff.max(), diff.mean())
    diff = np.abs(tflite_out - torch_out)
    print(diff.min(), diff.max(), diff.mean())
