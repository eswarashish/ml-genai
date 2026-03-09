from src.core.trainer.ett.dataloader import dataloader,DataloaderRequest
from src.utils.logger import getLogger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch

logger = getLogger("Trainer")

class ETTTrainer:
    def __init__(self,request: DataloaderRequest,train_split: float,epochs: int,optimizer: Optimizer,loss:nn.Module, scheduler:  LRScheduler ) -> None:
        self.train_dataloader, self.validate_dataloader = dataloader(request,train_split)
        self.epochs = epochs
        self.optim = optimizer
        self.criterion = loss
        self.lr_scheduler = scheduler
    def train(self,model:nn.Module):
        for epoch in range(self.epochs):
            logger.info(f"Epoch :{epoch+1}/{(self.epochs)} started")
            # Training phase
            model.train() # This will enable the Dropout and BatchNorm
            total_training_loss = 0
            for i,(features, target) in enumerate(self.train_dataloader):

                self.optim.zero_grad()
                output = model(features)
                loss = self.criterion(output,target)
                loss.backward()
                self.optim.step()
                total_training_loss += loss.item()

            model.eval() # This will disable the dropout and batchnorm
            total_val_loss = 0
            with torch.no_grad():
                for i,(features, target) in enumerate(self.validate_dataloader):
                    output = model(features)
                    loss = self.criterion(output,target)
                    total_val_loss += loss.item()

            self.lr_scheduler.step(total_val_loss)

            

