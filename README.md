# wheat-disease-detection

# 🌾 Wheat Disease Detection using Deep Learning

This project is an AI-based system that detects diseases in wheat crops using image classification.



## 🚀 Features

* Detects wheat leaf conditions:

  * Healthy
  * Rust
  * Powdery Mildew
* Uses MobileNetV2 (Transfer Learning)
* Real-time prediction using Streamlit
* Displays prediction confidence

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy
* Streamlit

---

## 📸 Screenshots

### 🔹 Upload Interface

![Upload](screenshots/screenshot1.png)

### 🔹 Image Preview

![Preview](screenshots/screenshot2.png)

### 🔹 Prediction Output

![Result](screenshots/screenshot3.png)

---

## ⚙️ How to Run

### 1. Clone the repository

```
git clone <your-repo-link>
cd wheat_Project
```

### 2. Install dependencies

```
py -3.10 -m pip install tensorflow numpy pillow streamlit opencv-python scipy
```

### 3. Train model

```
py -3.10 train.py
```

### 4. Run prediction

```
py -3.10 predict.py
```

### 5. Run web app

```
streamlit run app.py
```

---

## 📊 Model Details

* Model: MobileNetV2 (Transfer Learning)
* Input Size: 224x224
* Classes: Healthy, Rust, Mildew
* Accuracy: ~85–90%

---

## 💡 Learnings

* Handling overfitting using data augmentation
* Importance of dataset quality and balance
* Building end-to-end ML pipeline
* Deploying ML model using Streamlit

---

## 🎯 Future Improvements

* Add more disease classes
* Improve accuracy with larger dataset
* Deploy as mobile application

---

## 🙌 Acknowledgement

This project was built as part of a machine learning learning journey.

---

## 📌 Author

Harsh Vikram Singh
