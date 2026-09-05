# 🍎🍊 Apple vs. Orange Image Classifier

A lightweight computer vision project that classifies images of apples and oranges. The underlying neural network was trained using Google Teachable Machine and deployed using a custom Python script. 

This project demonstrates basic image preprocessing, channel conversion (BGR to RGB), and model inference using TensorFlow and OpenCV.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Machine Learning:** TensorFlow / Keras (Legacy Keras 2 format)
* **Computer Vision:** OpenCV (`cv2`)
* **Data Manipulation:** NumPy

## 📂 Project Structure
* `main.py`: The main execution script that loads the model, processes the input image, and prints the prediction.
* `keras_model.h5`: The exported Keras model weights and architecture.
* `labels.txt`: The text file containing the class names (`0 Apple`, `1 Orange`).
* `test_image.jpg`: A sample image used to test the model (replace with your own).

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
