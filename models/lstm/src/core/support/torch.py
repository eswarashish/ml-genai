import torch
from torch import device
from typing import Optional


class TorchDevice:
    def __init__(self):
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

    @property
    def dev(self)->device:
        return self._dev
       