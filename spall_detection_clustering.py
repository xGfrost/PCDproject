import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image and convert to grayscale
image_path = './PositiveSet/PositiveSet/Sample42.bmp'  # Path to your uploaded image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. Apply Median Filter (using OpenCV's built-in function)
image_filtered = cv2.medianBlur(image, 3)

# 3. Apply Gabor Filter (using OpenCV's built-in function)
def apply_gabor_filter(image, theta, lambd, gamma, psi, sigma):
    kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    gabor_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return gabor_image

# Gabor filter parameters
theta = 0  # Gabor filter orientation
lambd = 10  # Wavelength
gamma = 0.5  # Aspect ratio
psi = 0  # Phase offset
sigma = 4  # Standard deviation

# Apply Gabor filter
gabor_image = apply_gabor_filter(image_filtered, theta, lambd, gamma, psi, sigma)

# 4. Apply Otsu Thresholding (using OpenCV's built-in function)
_, binary_image = cv2.threshold(gabor_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. Find contours (using OpenCV's built-in function)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented_image = np.zeros_like(binary_image)
cv2.drawContours(segmented_image, contours, -1, (255), thickness=cv2.FILLED)

# Convert the segmented image to color for visualization
segmented_colored = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)
segmented_colored[:, :, 0] = binary_image  # Red channel for segmented object
segmented_colored[:, :, 2] = 255 - binary_image  # Blue background

# Plot the results
plt.figure(figsize=(15, 5))

# Image sample
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Image sample")
plt.axis('off')

# Gabor filtered image
plt.subplot(1, 3, 2)
plt.imshow(gabor_image, cmap='gray')
plt.title("Gabor filter")
plt.axis('off')

# Segmented image
plt.subplot(1, 3, 3)
plt.imshow(segmented_colored)
plt.title("Segmented image")
plt.axis('off')

plt.tight_layout()
plt.show()
