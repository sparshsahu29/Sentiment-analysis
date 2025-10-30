from tensorflow.keras.models import load_model
import os
import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model_path = os.path.join('models', 'sentiment.h5')
new_model = load_model(model_path)

# Read and preprocess the image
img = cv2.imread('sad1.jpg')
if img is None:
    raise ValueError("Image not found. Check the file path or filename.")

# Convert BGR (OpenCV default) to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to the model’s expected input size
resize = tf.image.resize(img, (256, 256))

# Normalize and expand dimensions for prediction
input_img = np.expand_dims(resize / 255.0, axis=0)

# Make prediction
yhat = new_model.predict(input_img)

# Interpret prediction (assuming sigmoid output)
if yhat > 0.5:
    print(f'Predicted class is Sad ({yhat[0][0]:.4f})')
else:
    print(f'Predicted class is Happy ({yhat[0][0]:.4f})')



