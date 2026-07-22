# 🌾 Wheat Disease Detection System (AI/ML + Streamlit)

## 📌 Project Overview
Wheat is affected by many diseases such as Rust, Smut, Powdery mildew and blights. These diseases reduce crop yield and quality.  
This project builds an **AI/ML based Wheat Disease Detection System** that detects wheat leaf disease from an image using Deep Learning (CNN).

The system provides:
✅ Disease prediction (Healthy / Rust)  
✅ Confidence score (%)  
✅ Remedy suggestions  
✅ Web-based interface using Streamlit  

---

## 🎯 Objectives
- Detect wheat leaf disease using image classification.
- Provide early alerts for farmers.
- Reduce crop loss using quick recommendations.
- Provide a simple web application to upload leaf image and get prediction.

---

## 🏗️ System Architecture
1. Farmer uploads/captures wheat leaf image  
2. Image preprocessing (resize, normalize)  
3. Deep Learning model (MobileNetV2) prediction  
4. Output displayed:
   - Disease name
   - Confidence score
   - Remedy suggestion  

---


---

## ✅ Dataset Setup
Place images in:
- `data/raw/healthy/`
- `data/raw/rust_stem/`

---

## ⚙️ Installation
Create virtual env (optional but recommended):
```bash
python -m venv venv
venv\\Scripts\\activate

pip install -r requirements.txt
pip install streamlit


# output

Output Example

The system displays:

Predicted class: Healthy / Rust

Confidence score: xx%

Remedy / Suggestions for farmer