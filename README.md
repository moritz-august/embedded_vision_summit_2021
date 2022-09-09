# Deploying PyTorch Models for Real-time Inference On the Edge

This is some example code I provided with my talk on "Deploying PyTorch Models for Real-time Inference On the Edge" at Embedded Vision Summit 2021.

The code illustrates some basic techniques for optimizing and compressing convolutional neural networks as well as preparing them for on-device execution via PyTorch and ONNX Runtime.
While the code showcases the full workflow for deployment for ARM-based platforms with inference on CPU/NPU/DSP, it assumes to be run on an x86 CPU for simplicity.
The code is tested and works with PyTorch 1.12. Note that as of PyTorch 1.12 it is also possible to export quantized pytorch models to ONNX directly.

Find more under:
https://embeddedvisionsummit.com/2021/2021/session/deploying-pytorch-models-for-real-time-inference-on-the-edge/
