from fastapi import FastAPI
from src.api.v1.routers.lstm.startup.lifespan import lifespan, getState
from src.api.v1.routers.lstm.startup.state import LSTMState
from fastapi import Request,Response, Depends


router = FastAPI(lifespan=lifespan)

@router.get("/health")
async def health(request: Request):
    return Response(status_code=200,content="Active and Healthy")

@router.post("/lstm")
async def lstm(state: LSTMState = Depends(getState) ):
    
    print(state)