from src.core.support.torch import TorchDevice
from src.models.custom import ETDLSTM
from src.core.trainer.ett.dataset import get_ettdata
from src.utils.logger import getLogger
from src.api.v1.routers.custom.startup.state import LSTMState
from pathlib import Path
from torch import float
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, Request

logger = getLogger('custom_lifespan')
@asynccontextmanager
async def lifespan(app: FastAPI)->AsyncIterator[None]:
    try:
        device = TorchDevice()
        sample_in,sample_out = get_ettdata(path=Path("/Users/eswarashish/Desktop/private/ml-genai/ETTh1.csv"),device=device.dev)[0]
        logger.info(f'{sample_in.shape}')
        lstm = ETDLSTM(5,4,3,4,0.4,device.dev,float)
        app.state.custom_state = LSTMState(device=device, lstm=lstm)
        yield 
        app.state.custom_state = None
    except Exception as e:
        raise RuntimeError(f'Application startup failed: {e}') from e
    

def getState(request:Request)->LSTMState:
    return request.app.state.custom_state
