import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("wheat_model.h5")

classes = ['Healthy', 'Rust', 'Mildew']

st.title("🌾 Wheat Disease Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction)*100

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
