from time import time

import onnxruntime as rt
import torch
import torch.nn as nn


def benchmark_onnx_model(model: rt.InferenceSession, n_samples: int = 100) -> float:
    avg_time = 0
    for i in range(n_samples):
        tensor = torch.rand((1, 3, 224, 224)).numpy()
        start = time()
        model.run(None, {'input_image': tensor})
        elapsed = time() - start
        avg_time += elapsed
    avg_time /= n_samples
    return avg_time * 1000


def benchmark_pytorch_model(model: nn.Module, n_samples: int = 100) -> float:
    avg_time = 0
    for i in range(n_samples):
        tensor = torch.rand((1, 3, 224, 224))
        start = time()
        model(tensor)
        elapsed = time() - start
        avg_time += elapsed
    avg_time /= n_samples
    return avg_time * 1000
