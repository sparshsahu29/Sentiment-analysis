from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)
model = load_model(os.path.join('models', 'sentiment.h5'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    # Read image from request
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # Preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(img, (256, 256))
    input_img = np.expand_dims(resize / 255.0, axis=0)

    # Predict
    yhat = model.predict(input_img)
    label = "Sad" if yhat > 0.5 else "Happy"

    return render_template('index.html', prediction=label, image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
