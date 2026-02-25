from fastapi import FastAPI
from src.api.v1.routers.lstm.module import  router as lstm_router
import uvicorn

app = FastAPI()
app.mount('/api/v1/lstm',app=lstm_router)

if __name__ == "__main__":
 uvicorn.run("src.api.v1.main:app", reload=True, port=8000, host='0.0.0.0')