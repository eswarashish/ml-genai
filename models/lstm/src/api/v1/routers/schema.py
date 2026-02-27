from typing import TypedDict, Optional
from fastapi import FastAPI
from contextlib import AbstractAsyncContextManager

class SubApp(TypedDict):
    sub_app: FastAPI
    lifespan: Optional[AbstractAsyncContextManager]