import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pandas as pd
from datetime import datetime


MODEL_PATH = "models/classification/model.tflite"
LABELS_PATH = "models/classification/labels.txt"
HISTORY_PATH = "data/prediction_history.csv"

IMG_SIZE = (224, 224)


@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image: Image.Image, interpreter, labels):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]  # shape (2,)
    idx = int(np.argmax(output))
    confidence = float(output[idx])

    return labels[idx], confidence, output

def remedy_text(label: str):
    label_low = label.lower()

    if "rust" in label_low:
        return (
            " **Disease Detected: Rust**\n\n"
            "**Suggested Actions:**\n"
            "- Apply recommended fungicide (as per agriculture officer)\n"
            "- Remove highly infected leaves\n"
            "- Avoid overhead irrigation\n"
            "- Maintain proper plant spacing for airflow\n"
        )
    elif "healthy" in label_low:
        return (
            " **Leaf is Healthy**\n\n"
            "**Suggestions:**\n"
            "- Continue monitoring weekly\n"
            "- Maintain balanced fertilizer\n"
            "- Ensure proper irrigation\n"
        )
    else:
        return "No remedy available for this class."


def save_history(filename, label, conf):
    row = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "prediction": label,
        "confidence_percent": round(conf * 100, 2),
    }

    os.makedirs("data", exist_ok=True)

    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(HISTORY_PATH, index=False)



st.set_page_config(page_title="Wheat Disease Detection", page_icon="🌾", layout="centered")

st.title("🌾 Wheat Disease Detection System")
st.write("Upload wheat leaf image and get disease prediction (Healthy / Rust).")

# Sidebar
st.sidebar.header(" System Info")
st.sidebar.write("**Model:** MobileNetV2 (TFLite)")
st.sidebar.write("**Classes:** Healthy / Rust")
st.sidebar.write("**Input Size:** 224×224")

# Check model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error(f"Labels not found: {LABELS_PATH}")
    st.stop()

interpreter = load_interpreter()
labels = load_labels()

uploaded_file = st.file_uploader(" Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Predicting... Please wait"):
        label, conf, scores = predict(image, interpreter, labels)

    # Save prediction history
    save_history(uploaded_file.name, label, conf)

    st.subheader(" Prediction Result")
    st.success(f"**Disease:** {label}")
    st.info(f"**Confidence:** {conf*100:.2f}%")

    st.progress(min(conf, 1.0))

    st.subheader("🩺 Remedy / Suggestion")
    st.markdown(remedy_text(label))

    # Show raw scores
    with st.expander("See class probability scores"):
        for i, cls in enumerate(labels):
            st.write(f"{cls}: {scores[i]*100:.2f}%")

# Show history
st.subheader(" Prediction History")
if os.path.exists(HISTORY_PATH):
    df_hist = pd.read_csv(HISTORY_PATH)
    st.dataframe(df_hist.tail(10), use_container_width=True)
else:
    st.write("No predictions yet. Upload an image first.")
