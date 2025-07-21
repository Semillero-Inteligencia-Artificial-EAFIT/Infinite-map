import cv2
import numpy as np

# Load 5 images (replace with your file paths)
image_paths = ['1.png', '2.png', '3.png', '4.png', '5.png']
images = [cv2.imread(p) for p in image_paths]

# Ensure all images are the same size
target_shape = images[0].shape
resized_images = [cv2.resize(img, (target_shape[1], target_shape[0])) for img in images]

# Average pixel values
combined = np.mean(resized_images, axis=0).astype(np.uint8)

# Save result
cv2.imwrite('combined_average.jpg', combined)