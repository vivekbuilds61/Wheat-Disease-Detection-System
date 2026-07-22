import tensorflow as tf
import os

H5_MODEL_PATH = "models/classification/mobilenet_v2_wheat.h5"
TFLITE_MODEL_PATH = "models/classification/model.tflite"

def main():
    if not os.path.exists(H5_MODEL_PATH):
        raise FileNotFoundError("Train model first. H5 not found.")

    model = tf.keras.models.load_model(H5_MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f" TFLite model saved: {TFLITE_MODEL_PATH}")

if __name__ == "__main__":
    main()
