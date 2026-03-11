from torch.utils.data import Dataset
from torch import device
from numpy.typing import ArrayLike
from typing import Tuple
from pathlib import Path
import torch
import numpy as np

# it is a safer and standard way to have dataset in cpu and load the values when required.
class ETTDataset(Dataset):
    def __init__(self,data: ArrayLike,n_samples: int,target: ArrayLike, 
                device: device
                 ) -> None:
        self.data = torch.from_numpy(data)
        self.target = torch.from_numpy(target)
        self.n_samples = n_samples
        self.device = device

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index)->Tuple:
        return self.data[index].to(self.device), self.target[index].to(self.device)
    
        
def get_ettdata(path: Path,device:device)->ETTDataset:
    xy =  np.loadtxt(path,delimiter=",",dtype=np.float32,skiprows=1, usecols=[1,2,3,4,5,6,7])
    return ETTDataset(data=xy[:,0:-1],target=xy[:,[-1]],n_samples=xy.shape[0],device=device)

