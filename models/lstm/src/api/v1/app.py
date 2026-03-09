from fastapi import FastAPI
from fastapi.requests import Request
from src.api.v1.routers.custom.module import router as lstm_router
from src.api.v1.routers.vanilla.module import router as vanilla_lstm_router
from src.api.v1.startup.lifespan import main_lifespan
from src.utils.logger import apilogger
import uvicorn

app = FastAPI(lifespan=main_lifespan)
# mount the sub-application that defines its own lifespan, and the sub lifespan's initialisation is handled in main lifespan
app.mount('/custom', app=lstm_router)    
app.mount('/vanilla',app=vanilla_lstm_router)


if __name__ == "__main__":
    uvicorn.run("src.api.v1.app:app",reload=True,port=8000,host='0.0.0.0')