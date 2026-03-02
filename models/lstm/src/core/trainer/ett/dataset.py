from torch.utils.data import Dataset
from torch import device
from numpy.typing import ArrayLike
from typing import Tuple
import torch

# it is a safer and standard way to have dataset in cpu and load the values when required.
class ETTDataset(Dataset):
    def __init__(self,data: ArrayLike,target: ArrayLike, device: device) -> None:
        self.data = torch.from_numpy(data)
        self.target = torch.from_numpy(target)
        self.device = device

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index)->Tuple:
        return self.data[index].to(device=self.device), self.target[index].to(device=self.device)
    
        
# ett_data = ETTDataset()
