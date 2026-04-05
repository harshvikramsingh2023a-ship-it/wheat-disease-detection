import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['Healthy', 'Mildew', 'Rust']

st.title("🌾 Wheat Disease Detection System")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
