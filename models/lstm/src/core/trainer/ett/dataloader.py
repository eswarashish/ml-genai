from torch.utils.data import DataLoader
from torch import device
from src.core.trainer.ett.dataset import get_ettdata
from pathlib import Path
from pydantic import BaseModel

class DataloaderRequest(BaseModel):
    device: device
    path:Path
    batch_size:int 
    shuffle: bool
    num_workers:int


def dataloader(request: DataloaderRequest)->DataLoader:
    return DataLoader(dataset=get_ettdata(request.path,request.device),
                      batch_size=request.batch_size,
                      shuffle=request.shuffle,
                      num_workers=request.num_workers,
                      pin_memory= (request.device.type == ("cuda" or "mps"))
                      )

