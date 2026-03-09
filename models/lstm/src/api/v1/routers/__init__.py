#this would help to write subapps attatched with their 
from typing import List
from src.api.v1.routers.custom.module import router as lstm_sub_app
from src.api.v1.routers.vanilla.module import router as vanilla_sub_app
from fastapi import FastAPI


sub_apps: List[FastAPI]  = [lstm_sub_app,vanilla_sub_app]