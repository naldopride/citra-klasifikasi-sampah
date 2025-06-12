from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Inisialisasi Flask
app = Flask(__name__)

# Load model CNN
model = load_model('model/cnn_model.h5')

# Folder upload
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi untuk proses gambar
def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_normalized = hsv / 255.0
    return hsv_normalized.reshape(1, 100, 100, 3), hsv

# Halaman awal
@app.route('/')
def index():
    return render_template('index.html')

# Prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        img_input, hsv_img = process_image(path)
        pred = model.predict(img_input)
        label = 'Organik' if np.argmax(pred) == 0 else 'Anorganik'

        # Histogram HSV
        colors = ('hue', 'saturation', 'value')
        plt.figure(figsize=(8, 3))
        for i, col in enumerate(colors):
            hist = cv2.calcHist([hsv_img], [i], None, [256], [0, 256])
            plt.plot(hist, label=col)
        plt.title('Histogram HSV')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        hist_path = os.path.join(UPLOAD_FOLDER, 'hist.png')
        plt.savefig(hist_path)
        plt.close()

        # Ambil akurasi dari file
        accuracy = 0
        try:
            with open('model/accuracy.txt', 'r') as f:
                accuracy = float(f.read()) * 100
        except:
            pass

        return render_template('result.html', label=label, image=path, hist=hist_path, accuracy=accuracy, cnn_hist='static/cnn_histogram.png')

# Jalankan Flask
if __name__ == '__main__':
    app.run(debug=True)
