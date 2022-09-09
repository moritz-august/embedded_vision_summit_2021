import logging
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from classifier import ToyClassifier
from data import ToyDataset
from deployment import deploy_onnx_quantized
from deployment import deploy_pytorch_float
from deployment import deploy_pytorch_quantized
from deployment import deploy_pytorch_quantized_nnapi

logging.basicConfig(level=logging.INFO)


def training_loop(model: nn.Module, dataloader: DataLoader):
    for batch in dataloader:
        # we don't actually do anything here other than initializing the BatchNorm2d parameters
        # to ensure proper quantization.
        model(batch)


def main():
    base_model_path = Path('./model_files')
    base_model_path.mkdir(exist_ok=True)

    dataset = ToyDataset()
    dataloader = DataLoader(dataset)

    for optimized, name in [(False, 'classifier'), (True, 'optimized_classifier')]:
        model = ToyClassifier(optimized=optimized)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logging.info(f'Model "{name}" has {n_params:2.2f} M parameters')

        training_loop(model, dataloader)

        deploy_pytorch_float(model, name=name)
        deploy_pytorch_quantized(dataloader, model, fuse=False, name=name, backend='fbgemm')
        deploy_pytorch_quantized(dataloader, model, fuse=True, name=name, backend='fbgemm')
        deploy_pytorch_quantized_nnapi(dataloader, model, fuse=True, name=name)
        deploy_onnx_quantized(dataloader, model, fuse=False, name=name)
        deploy_onnx_quantized(dataloader, model, fuse=True, name=name)


if __name__ == '__main__':
    main()
