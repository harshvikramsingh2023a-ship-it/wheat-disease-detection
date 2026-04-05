import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Fix working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = load_model("wheat_model.h5")
classes = ['healthy', 'mildew', 'rust']

st.title("🌾 Wheat Disease Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    st.success(f"Prediction: {classes[index]}")
    st.write(f"Confidence: {round(prediction[0][index]*100,2)}%")