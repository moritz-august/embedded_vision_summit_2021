import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import _make_divisible


class ToyClassifier(nn.Module):

    def __init__(self, optimized: bool = False):
        super(ToyClassifier, self).__init__()

        block_params = [
            (3, 18, 1),
            (18, 36, 2),
            (36, 74, 1),
            (74, 146, 2),
            (146, 290, 1),
            (290, 578, 2),
            (578, 1154, 1),
            (1154, 1154, 2)
        ]
        if optimized:
            blocks = [OptimizedConvBNReLU(in_channels=3,
                                          out_channels=_make_divisible(block_params[0][1], 8),
                                          stride=block_params[0][2])]
            for in_channels, out_channels, stride in block_params[1:]:
                blocks.append(OptimizedConvBNReLU(in_channels=_make_divisible(in_channels, 8),
                                                  out_channels=_make_divisible(out_channels, 8),
                                                  stride=stride)
                              )
            in_features = _make_divisible(1154, 8)
        else:
            blocks = [ConvBNReLU(in_channels=in_channels,
                                 out_channels=out_channels,
                                 stride=stride)
                      for (in_channels, out_channels, stride) in block_params]
            in_features = 1154

        self.blocks = nn.ModuleList(blocks)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(in_channels=in_features, out_channels=1000,
                                    kernel_size=(1, 1))

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.contiguous(memory_format=torch.channels_last)
        x = self.quant(x)
        for block in self.blocks:
            x = block(x)
        features = self.pooling(x)
        logits = self.classifier(features)
        logits = self.dequant(logits)
        return logits

    def fuse(self):
        for block in self.blocks:
            block.fuse()


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ConvBNReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                      stride=(stride, stride), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def fuse(self):
        torch.quantization.fuse_modules(self, ['layers.0', 'layers.1', 'layers.2'], inplace=True)


class OptimizedConvBNReLU(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(OptimizedConvBNReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                      stride=(stride, stride), padding=(1, 1), bias=False, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def fuse(self):
        torch.quantization.fuse_modules(self, ['layers.1', 'layers.2', 'layers.3'], inplace=True)
