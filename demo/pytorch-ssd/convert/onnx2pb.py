import onnx
from onnx_tf.backend import prepare
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file')
    parser.add_argument('--pb_file')
    args = parser.parse_args()

    o_model = onnx.load(args.onnx_file)
    tf_rep = prepare(o_model)
    tf_rep.export_graph(args.pb_file)

    print('converted onnx file to pb file')
