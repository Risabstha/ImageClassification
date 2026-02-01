
## augment each images with all augmentation types once 


import os
import uuid                          # for generating unique filenames created during augmentation
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

flag=False

if flag:
    # Save augmented copies - one for each augmentation type
    # Note: This runs BEFORE normalization, so images are in [0, 255] range
    aug_save_dir = os.path.join(data_dir, "augmented")
    tf.io.gfile.makedirs(aug_save_dir)

    # Ensure class folders exist
    class_names = getattr(data, "class_names", ["bad1", "good1"])
    for cname in class_names:
        tf.io.gfile.makedirs(os.path.join(aug_save_dir, cname))

    saved = 0

    # Process each batch in the dataset
    for batch_images, batch_labels in data:
        # Process each image in the batch
        for img, label in zip(batch_images, batch_labels):
            label_int = int(label)
            img_normalized = img / 255.0
            
            # 1. Vertical flip (deterministic - always flips)
            flip_img = tf.image.flip_up_down(img_normalized)
            flip_img = tf.clip_by_value(flip_img, 0.0, 1.0)
            flip_uint8 = tf.cast(flip_img * 255.0, tf.uint8)
            filename = f"{class_names[label_int]}_flip_{uuid.uuid4().hex[:8]}.jpg"
            out_path = os.path.join(aug_save_dir, class_names[label_int], filename)
            tf.io.write_file(out_path, tf.io.encode_jpeg(flip_uint8))
            saved += 1
            
            # 2. Rotation (deterministic - always rotates)
            rotation_img = tf.image.rot90(img_normalized, k=1)  # 90 degrees rotation
            rotation_img = tf.clip_by_value(rotation_img, 0.0, 1.0)
            rotation_uint8 = tf.cast(rotation_img * 255.0, tf.uint8)
            filename = f"{class_names[label_int]}_rotation_{uuid.uuid4().hex[:8]}.jpg"
            out_path = os.path.join(aug_save_dir, class_names[label_int], filename)
            tf.io.write_file(out_path, tf.io.encode_jpeg(rotation_uint8))
            saved += 1
            
            # 3. Zoom (deterministic - always zooms by cropping center and resizing)
            h, w = img_normalized.shape[0], img_normalized.shape[1]
            crop_size = int(h * 0.7)  # 30% zoom
            start = (h - crop_size) // 2
            zoom_img = img_normalized[start:start+crop_size, start:start+crop_size, :]
            zoom_img = tf.image.resize(zoom_img, [h, w])
            zoom_img = tf.clip_by_value(zoom_img, 0.0, 1.0)
            zoom_uint8 = tf.cast(zoom_img * 255.0, tf.uint8)
            filename = f"{class_names[label_int]}_zoom_{uuid.uuid4().hex[:8]}.jpg"
            out_path = os.path.join(aug_save_dir, class_names[label_int], filename)
            tf.io.write_file(out_path, tf.io.encode_jpeg(zoom_uint8))
            saved += 1

    print(f"Augmented images saved: {saved}")
    print(f"Output dir: {aug_save_dir}")