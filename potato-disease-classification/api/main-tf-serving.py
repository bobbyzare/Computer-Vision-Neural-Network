from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image # type: ignore 
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore 
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
import requests

app = FastAPI()


endpoint = "http://localhost:8502/v1/models/potatoes_model:predict"


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256, 256))  # Resize as needed
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension
    
    json_data = { "instances" :img_batch.tolist()}
    
    responce = requests.post(endpoint, json=json_data)
     
    prediction = np.array(responce.json()["prediction"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    confidence = np.max(prediction)

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    
    

   

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)



