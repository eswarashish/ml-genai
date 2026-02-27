#this would help to write subapps attatched with their 
from typing import List
from src.api.v1.routers.lstm.module import router as lstm_sub_app
from fastapi import FastAPI


sub_apps: List[FastAPI]  = [lstm_sub_app]