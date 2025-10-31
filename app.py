from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# 🧠 Lazy-load model to prevent Render startup timeout
model = None

def load_sentiment_model():
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        model = load_model(os.path.join('models', 'sentiment.h5'))
        print("✅ Model loaded successfully!")

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

    # Save uploaded file temporarily
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    img_path = os.path.join(upload_dir, file.filename)
    file.save(img_path)

    # 🧠 Load model only when needed
    load_sentiment_model()

    # Preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, (256, 256))
    input_img = np.expand_dims(img / 255.0, axis=0)

    # Predict
    yhat = model.predict(input_img)
    label = "Sad" if yhat > 0.5 else "Happy"

    return render_template('index.html', prediction=label, image_path=img_path)

if __name__ == '__main__':
    # ⚙️ Use Render’s assigned port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
