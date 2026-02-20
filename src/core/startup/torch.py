import torch
from torch import device
from typing import Optional


class TorchDevice:
    def __init__(self,dev: Optional[device]):
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

    @property
    def dev(self):
        return self._dev
       