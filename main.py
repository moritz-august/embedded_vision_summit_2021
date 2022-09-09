import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional

import onnxruntime as rt
import torch
import torch.nn as nn
import torch.onnx as onnx
from onnxruntime.quantization import quantize_static
from torch.backends._nnapi.prepare import convert_model_to_nnapi
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile

from benchmark import benchmark_onnx_model
from benchmark import benchmark_pytorch_model
from classifier import ToyClassifier
from data import ONNXQuantizationDataReader
from data import ToyDataset

logging.basicConfig(level=logging.INFO)


def script_and_serialize(model: nn.Module, path: str, opt_backend: Optional[str] = None):
    scripted_model = torch.jit.script(model)
    if opt_backend:
        scripted_model = optimize_for_mobile(script_module=scripted_model, backend=opt_backend)
    torch.jit.save(scripted_model, path)


def trace_and_serialize(model: nn.Module, example: torch.Tensor, path: str,
                        opt_backend: Optional[str] = None):
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_inputs=example)
    if opt_backend:
        traced_model = optimize_for_mobile(script_module=traced_model, backend=opt_backend)
    torch.jit.save(traced_model, path)


def training_loop(model: nn.Module, dataloader: DataLoader):
    for batch in dataloader:
        # we don't actually do anything here other than initializing the BatchNorm2d parameters
        # to ensure proper quantization.
        model(batch)


def deploy_float(model: nn.Module, name: str):
    model.eval()

    scripted_path = f'./model_files/{name}_float_scripted.pt'
    script_and_serialize(model, path=scripted_path)
    avg_time = benchmark_pytorch_model(torch.jit.load(scripted_path))
    size = Path(scripted_path).stat().st_size / 1e6
    logging.info(
        f'Benchmarking {scripted_path}: Avg. inference@CPU: {avg_time:3.2f} ms, Size: {size:2.2f} MB')

    traced_path = f'./model_files/{name}_float_traced.pt'
    trace_and_serialize(model, example=torch.rand(1, 3, 224, 224), path=traced_path)
    avg_time = benchmark_pytorch_model(torch.jit.load(traced_path))
    size = Path(traced_path).stat().st_size / 1e6
    logging.info(
        f'Benchmarking {traced_path}: Avg. inference@CPU: {avg_time:3.2f} ms, Size: {size:2.2f} MB')

    # Enable this only if you have built PyTorch with USE_VULKAN=1. Will fail otherwise.

    # vulkan_path = f'./model_files/{name}_float_vulkan_traced.pt'
    # script_and_serialize(model, path=vulkan_path, opt_backend='VULKAN')
    # avg_time = benchmark_model(torch.jit.load(vulkan_path))
    # size = Path(vulkan_path).stat().st_size / 1e6
    # logging.info(
    #     f'Benchmarking {vulkan_path}: Avg. inference@CPU: {avg_time:3.2f} ms, Size: {size:2.2f} MB')


def deploy_quantized(dataloader: DataLoader, model: nn.Module, fuse: bool, name: str,
                     backend: str = 'qnnpack'):
    model = deepcopy(model)
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    path = f'./model_files/{name}_quant'

    model = model.eval()
    if fuse:
        model.fuse()
        path += '_fused'

    model_prepared = torch.quantization.prepare(model)
    for sample in dataloader:
        model_prepared(sample)
    model_quantized = torch.quantization.convert(model_prepared)

    scripted_path = path + '_scripted.pt'
    traced_path = path + '_traced.pt'
    script_and_serialize(model_quantized, path=scripted_path, opt_backend='CPU')
    trace_and_serialize(model_quantized, example=torch.rand(1, 3, 224, 224), path=traced_path,
                        opt_backend='CPU')

    if fuse:
        avg_time = benchmark_pytorch_model(torch.jit.load(scripted_path))
        size = Path(scripted_path).stat().st_size / 1e6
        logging.info(
            f'Benchmarking {scripted_path}: Avg. inference@CPU: {avg_time:3.2f} ms, Size: {size:2.2f} MB')

    avg_time = benchmark_pytorch_model(torch.jit.load(traced_path))
    size = Path(traced_path).stat().st_size / 1e6
    logging.info(
        f'Benchmarking {traced_path}: Avg. inference@CPU: {avg_time:3.2f} ms, Size: {size:2.2f} MB')


def deploy_nnapi(dataloader: DataLoader, model: nn.Module, fuse: bool, name: str,
                 backend: str = 'qnnpack'):
    model = deepcopy(model)
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    path = f'./model_files/{name}_nnapi'

    model = model.eval()
    if fuse:
        model.fuse()
        path += '_fused'

    model_prepared = torch.quantization.prepare(model)
    for sample in dataloader:
        model_prepared(sample)
    model_quantized = torch.quantization.convert(model_prepared)

    input_float = torch.rand(1, 3, 224, 224)

    quantizer = model_quantized.quant
    dequantizer = model_quantized.dequant
    model_quantized.quant = torch.nn.Identity()
    model_quantized.dequant = torch.nn.Identity()
    input_tensor = quantizer(input_float)

    input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    input_tensor.nnapi_nhwc = True

    with torch.no_grad():
        model_quantized_traced = torch.jit.trace(model_quantized, input_tensor)
    nnapi_model = convert_model_to_nnapi(model_quantized_traced, input_tensor)
    nnapi_model_float_interface = torch.jit.script(
        torch.nn.Sequential(quantizer, nnapi_model, dequantizer))

    traced_path = path + '_traced.pt'
    traced_float_path = path + '_float_interface_traced.pt'
    nnapi_model.save(traced_path)
    nnapi_model_float_interface.save(traced_float_path)


def deploy_onnx_quantized(dataloader: DataLoader, model: nn.Module, fuse: bool, name: str):
    model = deepcopy(model)
    path = f'./model_files/{name}'

    model = model.eval()
    if fuse:
        model.fuse()
        path += '_fused'

    float_path = path + '_float.onnx'
    quantized_path = path + '_quant.onnx'
    example_input = torch.rand(1, 3, 224, 224)
    onnx.export(model=model, args=(example_input,), f=float_path, input_names=['input_image'],
                output_names=['logits'], opset_version=12)
    onnx_q_loader = ONNXQuantizationDataReader(quant_loader=dataloader, input_name='input_image')
    quantize_static(model_input=float_path, model_output=quantized_path,
                    calibration_data_reader=onnx_q_loader)
    shutil.move('./augmented_model.onnx', './model_files/augmented_model.onnx')

    avg_time = benchmark_onnx_model(rt.InferenceSession(float_path))
    size = Path(float_path).stat().st_size / 1e6
    logging.info(
        f'Benchmarking {float_path}: Avg. inference@CPU: {avg_time:3.2f} ms, Size: {size:2.2f} MB')
    avg_time = benchmark_onnx_model(rt.InferenceSession(quantized_path))
    size = Path(quantized_path).stat().st_size / 1e6
    logging.info(
        f'Benchmarking {quantized_path}: Avg. inference@CPU: {avg_time:3.2f} ms, Size: {size:2.2f} MB')


def main():
    for optimized, name in [(False, 'classifier'), (True, 'optimized_classifier')]:
        model = ToyClassifier(optimized=optimized)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Model "{name}" has {n_params / 1e6:2.2f} M parameters')
        dataset = ToyDataset()
        dataloader = DataLoader(dataset)

        training_loop(model, dataloader)

        deploy_float(model, name=name)
        deploy_onnx_quantized(dataloader, model, fuse=False, name=name)
        deploy_onnx_quantized(dataloader, model, fuse=True, name=name)
        deploy_quantized(dataloader, model, fuse=False, name=name, backend='fbgemm')
        deploy_quantized(dataloader, model, fuse=True, name=name, backend='fbgemm')
        deploy_nnapi(dataloader, model, fuse=True, name=name)


if __name__ == '__main__':
    main()
