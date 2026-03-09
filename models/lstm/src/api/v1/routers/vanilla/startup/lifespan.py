from src.core.support.torch import TorchDevice
from src.models.vanilla import VanillaLSTM
from src.api.v1.routers.vanilla.startup.state import LSTMState
from torch import float
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, Request


@asynccontextmanager
async def lifespan(app: FastAPI)->AsyncIterator[None]:
    try:
        device = TorchDevice()
        lstm = VanillaLSTM(5,4,device.dev,4,0.4,float)
        app.state.lstm_state = LSTMState(device=device, lstm=lstm)
        yield 
        app.state.lstm_state = None
    except Exception as e:
        raise RuntimeError(f'Application startup failed: {e}') from e
    

def getState(request:Request)->LSTMState:
    return request.app.state.lstm_state
