
from PIL import Image

# ## for single image resizing
# img = Image.open("input.jpg")
# img_resized = img.resize((100, 100), Image.Resampling.LANCZOS)
# img_resized.save("output.jpg")


## for batch image resizing
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "..", "data", "good1")
output_dir = os.path.join(BASE_DIR, "resized_images/good")

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with Image.open(input_path) as img:
            img_resized = img.resize((100, 100), Image.Resampling.LANCZOS)
            img_resized.save(output_path)

print("All images resized to 100x100.")