import shutil
from copy import deepcopy
from pathlib import Path
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.onnx as onnx
from onnxruntime.quantization import quantize_static
from torch.backends._nnapi.prepare import convert_model_to_nnapi
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile

from data import ONNXQuantizationDataReader


def deploy_pytorch_float(model: nn.Module, name: str, base_path: Path) -> List[Path]:
    model.eval()

    scripted_path = base_path / f'{name}_float_scripted.pt'
    serialize_script(model, path=scripted_path)

    traced_path = base_path / f'{name}_float_traced.pt'
    serialize_trace(model, example=torch.rand(1, 3, 224, 224), path=traced_path)

    return_paths = [scripted_path, traced_path]

    # Uncomment this only if you have built PyTorch with USE_VULKAN=1. Will fail otherwise.
    # vulkan_path = f'./model_files/{name}_float_vulkan_traced.pt'
    # script_and_serialize(model, path=vulkan_path, opt_backend='VULKAN')
    # return_paths.append(vulkan_path)

    return return_paths


def deploy_pytorch_quantized(dataloader: DataLoader, model: nn.Module, fuse: bool, name: str, base_path: Path) \
        -> List[Path]:
    model = deepcopy(model)
    backend = 'fbgemm'  # this is the appropriate choice for x86 CPUs, 'qnnpack' is the one for ARM architectures
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    model_name = f'{name}_quant'

    model = model.eval()
    if fuse:
        model.fuse()
        model_name += '_fused'

    model_prepared = torch.quantization.prepare(model)
    for sample in dataloader:
        model_prepared(sample)
    model_quantized = torch.quantization.convert(model_prepared)

    scripted_path = base_path / (model_name + '_scripted.pt')
    traced_path = base_path / (model_name + '_traced.pt')
    serialize_script(model_quantized, path=scripted_path, opt_backend='CPU')
    serialize_trace(model_quantized, example=torch.rand(1, 3, 224, 224), path=traced_path,
                    opt_backend='CPU')

    return_paths = [traced_path]
    if fuse:
        return_paths.append(scripted_path)

    return return_paths


def deploy_pytorch_quantized_nnapi(dataloader: DataLoader, model: nn.Module, fuse: bool, name: str, base_path: Path) \
        -> List[Path]:
    model = deepcopy(model)
    backend = 'qnnpack'
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    model_name = f'{name}_nnapi'

    model = model.eval()
    if fuse:
        model.fuse()
        model_name += '_fused'

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

    traced_path = base_path / (model_name + '_traced.pt')
    traced_float_path = base_path / (model_name + '_float_interface_traced.pt')
    nnapi_model.save(traced_path)
    nnapi_model_float_interface.save(traced_float_path)

    return []


def deploy_onnx_quantized(dataloader: DataLoader, model: nn.Module, fuse: bool, name: str, base_path: Path) \
        -> List[Path]:
    model = deepcopy(model)
    model_name = f'{name}'

    model = model.eval()
    if fuse:
        model.fuse()
        model_name += '_fused'

    float_path = base_path / (model_name + '_float.onnx')
    quantized_path = base_path / (model_name + '_quant.onnx')
    example_input = torch.rand(1, 3, 224, 224)
    onnx.export(model=model, args=(example_input,), f=float_path, input_names=['input_image'],
                output_names=['logits'], opset_version=12)
    onnx_q_loader = ONNXQuantizationDataReader(quant_loader=dataloader, input_name='input_image')
    quantize_static(model_input=float_path, model_output=quantized_path,
                    calibration_data_reader=onnx_q_loader)
    shutil.move('./augmented_model.onnx', base_path / 'augmented_model.onnx')

    return [float_path, quantized_path]


def serialize_script(model: nn.Module, path: Path, opt_backend: Optional[str] = None):
    scripted_model = torch.jit.script(model)
    if opt_backend:
        scripted_model = optimize_for_mobile(script_module=scripted_model, backend=opt_backend)
    torch.jit.save(scripted_model, path)


def serialize_trace(model: nn.Module, example: torch.Tensor, path: Path, opt_backend: Optional[str] = None):
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_inputs=example)
    if opt_backend:
        traced_model = optimize_for_mobile(script_module=traced_model, backend=opt_backend)
    torch.jit.save(traced_model, path)


def deploy_all(base_path: Path, dataloader: DataLoader, model: nn.Module, name: str):
    benchmark_models = []
    benchmark_models.extend(
        deploy_pytorch_float(model, name=name, base_path=base_path)
    )
    benchmark_models.extend(
        deploy_pytorch_quantized(dataloader, model, fuse=False, name=name, base_path=base_path)
    )
    benchmark_models.extend(
        deploy_pytorch_quantized(dataloader, model, fuse=True, name=name, base_path=base_path)
    )
    benchmark_models.extend(
        deploy_pytorch_quantized_nnapi(dataloader, model, fuse=True, name=name, base_path=base_path)
    )
    benchmark_models.extend(
        deploy_onnx_quantized(dataloader, model, fuse=False, name=name, base_path=base_path)
    )
    benchmark_models.extend(
        deploy_onnx_quantized(dataloader, model, fuse=True, name=name, base_path=base_path)
    )
    return benchmark_models
