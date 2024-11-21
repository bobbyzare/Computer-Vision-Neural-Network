from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image # type: ignore 
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore 
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model("C:/Users/bobby/potato-disease/models/Potato_model_3.keras") 

CLASS_NAMES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']

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

    predictions = model.predict(img_batch)  # Use 'model' instead of 'MODEL'

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]       
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
        
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)



