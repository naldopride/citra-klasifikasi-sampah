import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Parameter ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'train'   # Folder dengan subfolder per kelas

# --- Custom preprocessing function untuk HSV ---
def preprocess_hsv(img):
    # img: numpy array RGB [0-255]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv / 255.0  # normalisasi
    return hsv

# --- Buat generator dengan preprocessing HSV ---
class HSVImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        # Override method standardize agar output HSV + normalisasi
        # x shape: (height, width, 3) float32 [0-1] dari rescale jika ada
        x_uint8 = (x * 255).astype(np.uint8)
        hsv = cv2.cvtColor(x_uint8, cv2.COLOR_RGB2HSV)
        hsv = hsv / 255.0
        return hsv.astype(np.float32)

# --- Prepare data generators ---
datagen = HSVImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# --- Bangun model CNN sederhana ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Training model ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# --- Plot akurasi dan loss ---
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# --- Contoh prediksi gambar baru ---
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_hsv = preprocess_hsv(img)
    img_hsv = np.expand_dims(img_hsv, axis=0)  # batch size 1
    pred = model.predict(img_hsv)
    class_idx = np.argmax(pred)
    class_label = list(train_generator.class_indices.keys())[class_idx]
    print(f"Prediksi: {class_label}")
    plt.imshow(img)
    plt.title(f"Prediksi: {class_label}")
    plt.axis('off')
    plt.show()

# --- Upload dan prediksi gambar (hanya di Jupyter Notebook) ---
try:
    from IPython.display import display
    import ipywidgets as widgets

    upload_widget = widgets.FileUpload(
        accept='image/*',
        multiple=False
    )

    def on_upload_change(change):
        for filename, fileinfo in upload_widget.value.items():
            with open(filename, 'wb') as f:
                f.write(fileinfo['content'])
            print(f"File '{filename}' berhasil diupload, mulai prediksi...")
            predict_image(filename)
            # Hapus file hasil upload jika mau
            # os.remove(filename)

    upload_widget.observe(on_upload_change, names='value')
    print("Silakan upload gambar untuk diprediksi:")
    display(upload_widget)

except ImportError:
    print("ipywidgets tidak tersedia, fitur upload gambar via widget tidak aktif.")
