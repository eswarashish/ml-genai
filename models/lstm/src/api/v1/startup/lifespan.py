from src.api.v1.routers.lstm.startup import lifespan as lstm_lifespan
from contextlib import asynccontextmanager, AsyncExitStack
from fastapi import FastAPI
from typing import AsyncIterator

@asynccontextmanager
async def main_lifespan(app: FastAPI)->AsyncIterator[None]:
    yield