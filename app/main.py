# ASGI server
import uvicorn

# A class for working with binary data in memory
from fastapi import FastAPI, File, UploadFile

from typing import Union

app = FastAPI()


@app.get("/")
async def root():
    return "Hello world"

@app.get("/api/analyze/pso")
async def analyze_photo_PSO(file: UploadFile | None = None):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    # read image file, convert into compatible variable
    # pass image to model for prediction?
    pass

@app.get("/api/analyze/abc")
async def analyze_photo_ABC(file: UploadFile | None = None):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    # read image file, convert into compatible variable
    # pass image to model for prediction?
    pass

@app.get("/api/analyze/aco")
async def analyze_photo_ACO(file: UploadFile | None = None):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    # read image file, convert into compatible variable
    # pass image to model for prediction?
    pass