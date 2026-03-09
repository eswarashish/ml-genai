from src.core.trainer.service import Trainer
from src.core.trainer.state import DataloaderRequest
from torch import nn

async def train(request: DataloaderRequest,trainer: Trainer,model:nn.Module):
    trainer.train(model=model)
    pass
