from src.core.support.torch import TorchDevice
from src.models.lstm import ETDLSTM
from src.api.v1.routers.lstm.startup.state import LSTMState
from torch import float
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI)->AsyncIterator[LSTMState]:
    try:
        device = TorchDevice(dev=None)
        lstm = ETDLSTM(5,4,3,2,0.4,device.dev,float)
        yield LSTMState(device=device,lstm=lstm)
    except Exception as e:
        raise RuntimeError(f'Application startup failed: {e}') from e