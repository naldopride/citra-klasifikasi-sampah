import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
import json

# --- Parameter ---
IMG_SIZE = 128
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Fungsi cek ekstensi ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# --- Preprocess HSV ---
def preprocess_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv / 255.0
    return hsv

# --- Load model dan class indices ---
model = load_model('my_model.h5')

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

idx_to_class = {v:k for k,v in class_indices.items()}

# --- Route utama ---
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='Tidak ada file di form')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='Tidak ada file dipilih')

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Baca gambar, preprocess dan prediksi
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_hsv = preprocess_hsv(img)
            img_hsv = np.expand_dims(img_hsv, axis=0)

            pred = model.predict(img_hsv)
            class_idx = np.argmax(pred)
            class_label = idx_to_class.get(class_idx, 'Unknown')

            return render_template('index.html', filename=file.filename, prediction=class_label)

        else:
            return render_template('index.html', message='Format file tidak didukung')

    return render_template('index.html')

# --- Route untuk tampilkan gambar hasil upload ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Jalankan app ---
if __name__ == '__main__':
    app.run(debug=True)
