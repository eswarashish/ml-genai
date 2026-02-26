from fastapi import FastAPI
from src.api.v1.routers.lstm.module import router as lstm_router
from src.api.v1.routers.lstm.startup.lifespan import lifespan as lstm_lifespan
import asyncio
import uvicorn

app = FastAPI()

# mount the sub-application that defines its own lifespan
app.mount('/lstm', app=lstm_router)
