from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import os
import base64
import io

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/USER/Downloads/model3.h5')

def predict_lung_cancer(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.array([img]).reshape(-1, 256, 256, 1) / 255.0

    # Make prediction
    prediction = model.predict(img)
    prediction_class = np.argmax(prediction)

    # Map prediction to class label
    if prediction_class == 0:
        return "Benign case"
    elif prediction_class == 1:
        return "Malignant case"
    else:
        return "Normal case"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file:
            # Create a unique filename for the uploaded image
            upload_dir = 'uploads'
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, file.filename)
            file.save(image_path)

            # Make prediction using the uploaded image
            prediction = predict_lung_cancer(image_path)

            return render_template('model.html', prediction=prediction, filename=file.filename)

    return render_template('model.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/Home')
def Hello():
    return render_template('Home.html')

@app.route('/Paper')
def Paper():
    return render_template('Paper.html')

@app.route('/Report')
def Report():
    return render_template('report.html')

if __name__ == '__main__':
    app.run(debug=True)
