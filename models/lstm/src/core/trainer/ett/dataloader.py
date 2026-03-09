from torch.utils.data import DataLoader, random_split
from src.core.trainer.state import DataloaderRequest
from src.core.trainer.ett.dataset import get_ettdata
from typing import Tuple


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

