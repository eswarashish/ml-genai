from fastapi import FastAPI
from src.api.v1.routers.lstm.startup.lifespan import lifespan, getState
from src.api.v1.routers.schema import SubApp
from src.api.v1.routers.lstm.startup.state import LSTMState
from fastapi import Response, Depends


router = FastAPI(lifespan=lifespan)

@router.get("/health")
async def health(state: LSTMState = Depends(getState)):
    
    return Response(status_code=200,content="Active and Healthy")

@router.post("/device")
async def lstm(state: LSTMState = Depends(getState))->str:
    return state["device"].dev.type