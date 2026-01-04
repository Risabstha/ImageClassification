import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import subprocess
import matplotlib.pyplot as plt


# ---------------------------
# 1. Image preprocessing
# ---------------------------
train_dir = 'data/train'
val_dir = 'data/val'

# ---------------------------
# 0. Split dataset if needed
# ---------------------------

# Check if train/val folders exist and are non-empty
def is_nonempty_dir(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0

if not (is_nonempty_dir(train_dir) and is_nonempty_dir(val_dir)):
    print('Splitting dataset into train/val...')
    split_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../split_dataset.py'))
    result = subprocess.run([sys.executable, split_script], check=True)
    print('Dataset split complete.')

# Data augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# ---------------------------
# 2. Build CNN model
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# ---------------------------
# 3. Compile the model
# ---------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# 4. Train the model
# ---------------------------
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator
)

# ---------------------------
# 5. Plot training history
# ---------------------------
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
