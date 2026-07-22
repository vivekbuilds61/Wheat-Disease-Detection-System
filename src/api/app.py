from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from src.api.predict import predict

app = FastAPI(title="Wheat Disease Detection API")

@app.get("/")
def home():
    return {"message": " Wheat Disease Detection API Running"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    return predict(image)
