import os
import cv2 as cv
from stitching import Stitcher

# Folder where your images are located
image_folder = "data/tiles"

# Get all files in the folder and filter only images (jpg, png, etc.)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Sort to keep order consistent (important for stitching)
image_files.sort()

# Create full paths for the images
image_paths = [os.path.join(image_folder, f) for f in image_files]

# Load images into memory
images = [cv.imread(img) for img in image_paths]

# Initialize the stitcher with custom settings (optional)
stitcher = Stitcher(detector="sift", confidence_threshold=0.2)

# Create panorama
panorama = stitcher.stitch(images)

# Save result
cv.imwrite("panorama_result.jpg", panorama)

print("✅ Panorama saved as panorama_result.jpg")
