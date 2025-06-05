import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Parameter ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'train'  # Folder dataset dengan subfolder per kelas

# --- Preprocess HSV function ---
def preprocess_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv / 255.0
    return hsv

# --- Custom ImageDataGenerator to convert RGB->HSV ---
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class HSVImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        # x shape: (h,w,3) float32 [0-1]
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

# --- Build model ---
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

# --- Train model ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# --- Save model ---
model.save('my_model.h5')
print("Model disimpan di my_model.h5")

# --- Save class indices ---
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Class indices disimpan di class_indices.json")
