from src.core.startup.torch import TorchDevice
from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        device = TorchDevice()
        yield device
    except Exception as e:
        raise RuntimeError(f'Application startup failed: {e}') from e