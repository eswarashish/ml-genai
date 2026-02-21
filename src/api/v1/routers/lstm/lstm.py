from fastapi import FastAPI
from src.api.v1.routers.lstm.startup.lifespan import lifespan
from torch import Tensor
from fastapi import Request,Response


app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health(request: Request):
    return Response(status_code=200,content="Active and Healthy")

@app.post("/lstm")
async def lstm(x: Tensor):
    app.state
    return