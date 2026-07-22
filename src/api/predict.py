import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/classification/model.tflite"
LABELS_PATH = "models/classification/labels.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image: Image.Image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(output_data))
    confidence = float(output_data[idx])

    return {
        "disease": LABELS[idx],
        "confidence": round(confidence * 100, 2)
    }
