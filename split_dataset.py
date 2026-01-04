import os
import shutil
import random
from glob import glob

# Set random seed for reproducibility
random.seed(42)

# Source and target directories
source_dir = 'data'
train_dir = 'data/train'
val_dir = 'data/val'

# Create target directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Check if there are class subfolders
items = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d not in ['train', 'val']]

if items:
    # There are class subfolders
    for class_name in items:
        class_path = os.path.join(source_dir, class_name)
        images = glob(os.path.join(class_path, '*'))
        random.shuffle(images)
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create class subfolders in train/val
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        for img in train_images:
            shutil.copy(img, os.path.join(train_dir, class_name, os.path.basename(img)))
        for img in val_images:
            shutil.copy(img, os.path.join(val_dir, class_name, os.path.basename(img)))
else:
    # No class subfolders, just images
    images = glob(os.path.join(source_dir, '*'))
    random.shuffle(images)
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    for img in train_images:
        shutil.copy(img, os.path.join(train_dir, os.path.basename(img)))
    for img in val_images:
        shutil.copy(img, os.path.join(val_dir, os.path.basename(img)))

print('Dataset split complete!')
