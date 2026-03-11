from src.core.trainer.service import Trainer
from src.core.trainer.state import TrainerRequest
from torch import nn

async def train(request: TrainerRequest,model:nn.Module):
    trainer = Trainer(request)
    trainer.train(model=model)
    pass
