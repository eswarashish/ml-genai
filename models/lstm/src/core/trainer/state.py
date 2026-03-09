from torch import device
from pathlib import Path
from pydantic import BaseModel

class DataloaderRequest(BaseModel):
    device: device
    path:Path
    batch_size:int 
    shuffle: bool
    num_workers:int
