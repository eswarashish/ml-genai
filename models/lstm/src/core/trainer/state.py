from torch import device
from pathlib import Path
from pydantic import BaseModel
from dataclasses import dataclass
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

# class DataloaderRequest(BaseModel):
#     device: device
#     path:Path
#     batch_size:int 
#     shuffle: bool
#     num_workers:int


# class TrainerAPIRequest(BaseModel):
#     epochs: int
#     optimizer: Optimizer
#     loss:nn.Module 
#     scheduler:  LRScheduler
#     train_dataloader: DataLoader
#     validate_dataloader: DataLoader 


@dataclass
class TrainerRequest:
    epochs: int
    optimizer: Optimizer
    loss:nn.Module 
    scheduler:  LRScheduler
    train_dataloader: DataLoader
    validate_dataloader: DataLoader 