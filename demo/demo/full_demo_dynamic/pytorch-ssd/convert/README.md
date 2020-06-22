# Converting from PyTorch

To get the main run.sh script to work you'll need to install onnx:
pip(3) install onnx

Furthermore we need the onnx-tf package which allows onnx models to run using a Tensorflow backend.
Just a warning that this package is a bit experimental, as it seems that Google hasn't officially added onnx support like in PyTorch.

To install you need to clone the repo:
git clone https://github.com/onnx/onnx-tensorflow.git

Then cd to the onnx-tensorflow dir and run:
pip(3) install .

Everything works for me using Tensorflow 2.1.0. Tensorflow verison definitely seems to matter.

I do get some errors about CUDA not being available.
I also get a flood of UserWarnings converting from onnx to pb.
Neither seem to effect the result.
