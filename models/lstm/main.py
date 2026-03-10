# from src.utils.logger import apilogger
import uvicorn


if __name__ == "__main__":
    uvicorn.run("src.api.v1.app:app",reload=True,port=8000,host='0.0.0.0')