from fastapi import FastAPI
from src.api.v1.routers.custom.startup.lifespan import lifespan, getState
from src.api.v1.routers.custom.startup.state import LSTMState
from src.utils.logger import getLogger
from fastapi import Response, Depends,status

router = FastAPI(lifespan=lifespan,title="custom-lstm-router")
logger = getLogger("custom-lstm-router")
@router.get("/lstm-health")
async def health(state: LSTMState = Depends(getState)):
    try:
        state['lstm'].compile()
        
    except Exception as e:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,content=f"An error occured while initiliasing: {e}")

    logger.debug(msg="Torch is compiled")
    return Response(status_code=status.HTTP_200_OK,content="Active and Healthy")

@router.post("/device")
async def lstm(state: LSTMState = Depends(getState))->str:
    return state["device"].dev.type

@router.post("/train")
async def train(state: LSTMState  = Depends(getState)):
    pass