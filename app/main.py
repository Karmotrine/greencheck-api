# ASGI server
import uvicorn
# A class for working with binary data in memory
from fastapi import FastAPI, Response, UploadFile
from typing import Union
app = FastAPI()

from const import *
from lib.Processing import *
from lib.Model import Model
from lib.Analysis import *
from utilities.api import *
"""
"""

@app.post("/api/analyze/base")
async def analyze_photo_base(file: UploadFile, response: Response):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {
            "status": "422",
            "message": "File must be an image with extension: .jpg, .jpeg, .png"
        }
    
    input_image = read_cv2_image(await file.read())
    base_model = Model(ModelType.BaseModel)
    base_model.load()
    predictions = base_model.api_predict(input_image)

    return {
        "status": "200",
        "predictions": predictions
    }

@app.post("/api/analyze/pso")
async def analyze_photo_PSO(file: UploadFile, response: Response):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {
            "status": "422",
            "message": "File must be an image with extension: .jpg, .jpeg, .png"
        }

    input_image = read_cv2_image(await file.read())
    pso_model = Model(ModelType.ParticleSwarm)
    pso_model.load()
    predictions = pso_model.api_predict(input_image)

    return {
        "predictions": predictions
    }

@app.post("/api/analyze/abc")
async def analyze_photo_ABC(file: UploadFile, response: Response):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {
            "status": "422",
            "message": "File must be an image with extension: .jpg, .jpeg, .png"
        }

    input_image = read_cv2_image(await file.read())
    abc_model = Model(ModelType.ArtificialBee)
    abc_model.load()
    predictions = abc_model.api_predict(input_image)

    return {
        "status": "200",
        "predictions": predictions
    }

@app.post("/api/analyze/aco")
async def analyze_photo_ACO(file: UploadFile, response: Response):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {
            "status": "422",
            "message": "File must be an image with extension: .jpg, .jpeg, .png"
        }

    input_image = read_cv2_image(await file.read())
    aco_model = Model(ModelType.AntColony)
    aco_model.load()
    predictions = aco_model.api_predict(input_image)

    return {
        "status": "200",
        "predictions": predictions
    }
