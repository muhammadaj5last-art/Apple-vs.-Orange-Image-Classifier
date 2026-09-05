import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
import cv2
import sys

np.set_printoptions(suppress=True)

def predict_image(image_path):
    # 1. Load model and labels
    try:
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
        with open("labels.txt", "r") as f:
            class_names = f.readlines()
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        return

    # 2. Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # FIX: Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_normalized = (image_array / 127.5) - 1

    # 3. Predict class
    prediction = model.predict(image_normalized)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # 4. Display results
    print(f"Predicted Class: {class_name[2:]}")
    print(f"Confidence Score: {confidence_score:.4f}")

if __name__ == "__main__":
    predict_image("apple4.jpg")