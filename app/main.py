# ASGI server
from pydantic import BaseModel
import uvicorn
# A class for working with binary data in memory
from fastapi import FastAPI, File, Response, UploadFile
from typing import Union
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

from const import *
from lib.Processing import *
from lib.Model import Model
from lib.Analysis import *
from utilities.api import *

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class Image(BaseModel):
    img_base64: str

@app.post("/api/analyze/base")
async def analyze_photo_base(file: Image, response: Response):
    input_image = readBy64(file.img_base64)
    base_model = Model(ModelType.BaseModel)
    base_model.load()
    predictions = base_model.api_predict(input_image)

    return {
        "status": "200",
        "predictions": predictions
    }

@app.post("/api/analyze/pso")
async def analyze_photo_PSO(file: Image, response: Response):
    input_image = readBy64(file.img_base64)
    pso_model = Model(ModelType.ParticleSwarm)
    pso_model.load()
    predictions = pso_model.api_predict(input_image)

    return {
        "status": "200",
        "predictions": predictions
    }

@app.post("/api/analyze/abc")
async def analyze_photo_ABC(file: Image, response: Response):
    input_image = readBy64(file.img_base64)
    abc_model = Model(ModelType.ArtificialBee)
    abc_model.load()
    predictions = abc_model.api_predict(input_image)

    return {
        "status": "200",
        "predictions": predictions
    }

@app.post("/api/analyze/aco")
async def analyze_photo_ACO(file: Image, response: Response):
    input_image = readBy64(file.img_base64)
    aco_model = Model(ModelType.AntColony)
    aco_model.load()
    predictions = aco_model.api_predict(input_image)

    return {
        "status": "200",
        "predictions": predictions
    }
