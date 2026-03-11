from src.utils.logger import getLogger
from src.core.trainer.state import TrainerRequest
from torch import nn
import torch

logger = getLogger("Trainer")

class Trainer:
    def __init__(self, request: TrainerRequest) -> None:
        self.train_dataloader, self.validate_dataloader = request.train_dataloader, request.validate_dataloader
        self.epochs = request.epochs
        self.optim = request.optimizer
        self.criterion = request.loss
        self.lr_scheduler = request.scheduler
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

            

