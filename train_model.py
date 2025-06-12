import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset dan ubah ke HSV
def load_data(data_dir, img_size=(100, 100)):
    X, y = [], []
    labels = {'organik': 0, 'anorganik': 1}

    for label_name, label in labels.items():
        folder = os.path.join(data_dir, label_name)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                X.append(hsv)
                y.append(label)

    return np.array(X), to_categorical(y)

# Load dan split dataset
X, y = load_data('dataset')
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Buat model CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train dan simpan riwayat
history = model.fit(X_train, y_train, epochs=39, validation_data=(X_test, y_test))

# Simpan model
os.makedirs('model', exist_ok=True)
model.save('model/cnn_model.h5')

# Evaluasi dan simpan akurasi
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi model: {accuracy * 100:.2f}%")
with open('model/accuracy.txt', 'w') as f:
    f.write(str(accuracy))

# Plot histogram akurasi dan loss
plt.figure(figsize=(12, 5))

# Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Akurasi CNN per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss CNN per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Simpan grafik histogram CNN
plt.tight_layout()
plt.savefig('model/cnn_histogram.png')
plt.close()

import shutil
shutil.copy('model/cnn_histogram.png', 'static/cnn_histogram.png')
