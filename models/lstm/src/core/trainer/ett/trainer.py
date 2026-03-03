from src.core.trainer.ett.dataloader import dataloader,DataloaderRequest
from torch import nn
from torch.optim import Optimizer


class ETTTrainer:
    def __init__(self,request: DataloaderRequest,epochs: int,optimizer: Optimizer,loss:nn.Module) -> None:
        self.dataloader = dataloader(request)
        self.epochs = epochs
        self.optim = optimizer
        self.loss = loss
    def train(self,model:nn.Module):
        for epoch in range(self.epochs):
            for i,(features, target) in enumerate(self.dataloader):
                self.optim.zero_grad()
                output = model(features)
                loss = self.loss(output,target)
                loss.backward()
                self.optim.step()
                
