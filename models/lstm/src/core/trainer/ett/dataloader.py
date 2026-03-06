from torch.utils.data import DataLoader, random_split
from torch import device
from src.core.trainer.ett.dataset import get_ettdata
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple

class DataloaderRequest(BaseModel):
    device: device
    path:Path
    batch_size:int 
    shuffle: bool
    num_workers:int


def dataloader(request: DataloaderRequest, train_split: float)->Tuple[DataLoader,DataLoader]:
    dataset=get_ettdata(request.path,request.device)
    train_size = int(train_split*len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset,[train_size,val_size])

    return (DataLoader(train_data,
                      batch_size=request.batch_size,
                      shuffle=request.shuffle,
                      num_workers=request.num_workers,
                      pin_memory= (request.device.type == ("cuda" or "mps"))
                      ), 
                      DataLoader(val_data,
                        batch_size=request.batch_size,
                        shuffle=request.shuffle,
                        num_workers=request.num_workers,
                        pin_memory = (request.device.type == ("cuda" or "mps"))
                        ))

