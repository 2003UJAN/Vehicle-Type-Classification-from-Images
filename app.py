import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import io
from PIL import Image

# ---- Hugging Face Model Link ----
MODEL_URL = "https://huggingface.co/your-username/vehicle-type-classifier/resolve/main/vehicle_type_classifier.h5"

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_bytes = requests.get(MODEL_URL).content
    model = tf.keras.models.load_model(io.BytesIO(model_bytes))
    return model

model = load_model()
class_labels = ['Big Truck', 'City Car', 'Multi Purpose Vehicle', 'Sedan', 'Sport Utility Vehicle', 'Truck', 'Van']

# ---- Streamlit UI ----
st.set_page_config(page_title="Vehicle Type Classifier")
st.title("🚘 Vehicle Type Image Classifier")
st.markdown("Upload a vehicle image to classify its type using a trained CNN model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Vehicle Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Vehicle Type"):
        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]
        st.success(f"Predicted Vehicle Type: {predicted_label}")
