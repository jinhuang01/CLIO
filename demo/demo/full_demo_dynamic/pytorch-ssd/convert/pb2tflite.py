import tensorflow.compat.v1 as tf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_file')
    parser.add_argument('--tflite_file')
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        args.pb_file,
        ['test_input'],
        ['test_output']
    )
    tflite_model = converter.convert()

    with open(args.tflite_file, 'wb') as f:
        f.write(tflite_model)
    print('converted pb file to tflite file')
