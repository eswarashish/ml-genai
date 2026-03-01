from src.api.v1.routers import sub_apps
from contextlib import asynccontextmanager, AsyncExitStack
from src.utils.logger import apilogger
from fastapi import FastAPI
from typing import AsyncIterator

@asynccontextmanager
async def main_lifespan(app: FastAPI)->AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        for sub_app in sub_apps:   
              await stack.enter_async_context(sub_app.router.lifespan_context(sub_app))
              apilogger.info(f"Initialised lifespan for sub-app: {sub_app.title}")
        
        yield
        