import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

print("✅ RUNNING CORRECT PREDICT FILE")

# Ensure correct path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = load_model("wheat_model.h5")

# IMPORTANT: correct order from training
classes = ['healthy', 'mildew', 'rust']

img = image.load_img("test.png", target_size=(224,224))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
index = np.argmax(prediction)

print("Prediction:", classes[index])
print("Confidence:", round(prediction[0][index]*100, 2), "%")