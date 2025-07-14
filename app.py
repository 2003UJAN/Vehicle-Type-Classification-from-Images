import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import io
from PIL import Image

# ---- Hugging Face Raw Model URL ----
MODEL_URL = "https://huggingface.co/ujan2003/vehicle-type-classifier/resolve/main/vehicle_type_classifier.h5"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    model_bytes = io.BytesIO(response.content)
    model = tf.keras.models.load_model(model_bytes)
    return model

# Load the model once
model = load_model()

# Class labels (in the same order as during training)
class_labels = [
    "Big Truck", 
    "City Car", 
    "Multi Purpose Vehicle", 
    "Sedan", 
    "Sport Utility Vehicle", 
    "Truck", 
    "Van"
]

# ---- Streamlit UI ----
st.set_page_config(page_title="Vehicle Type Classifier")
st.title("🚗 Vehicle Type Image Classifier")
st.markdown("Upload an image of a vehicle to classify its type using a deep learning model.")

uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict button
    if st.button("Classify"):
        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]
        st.success(f"🚘 Predicted Vehicle Type: **{predicted_label}**")
