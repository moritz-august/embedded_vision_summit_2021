import torch
from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ToyDataset(Dataset):

    def __init__(self):
        super(ToyDataset, self).__init__()
        self.len = 10

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int) -> torch.Tensor:
        return torch.rand((3, 224, 224))


class ONNXQuantizationDataReader(CalibrationDataReader):
    def __init__(self,
                 quant_loader: DataLoader,
                 input_name: str):
        self.data = []
        for inputs in quant_loader:
            # Here we unroll batch size as dynamic axis is not supported and
            # batch size is then hardcoded to 1
            for input_frame in inputs:
                self.data.append(input_frame.unsqueeze(0).numpy())

        self.iter = iter([{input_name: d} for d in self.data])

    def get_next(self):
        return next(self.iter, None)
