import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image and convert to grayscale
image = cv2.imread('./PositiveSet/PositiveSet/Sample293.bmp', cv2.IMREAD_GRAYSCALE)

# 2. Apply Median Filter to remove noise
image_filtered = cv2.medianBlur(image, 3)

# 3. Apply Gabor filter to extract features
def gabor_filter(image, theta, lambd, gamma, psi, sigma):
    kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return filtered_img

theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Gabor orientations
gabor_features = []

# Collect Gabor filtered images
for theta in theta_values:
    gabor_img = gabor_filter(image_filtered, theta, 10, 0.5, 0, 4)
    gabor_features.append(gabor_img)

# 4. Binarize the first Gabor filtered image
_, binary_segment = cv2.threshold(gabor_features[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. Find contours for segmented objects
contours, _ = cv2.findContours(binary_segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented_image = np.zeros_like(binary_segment)
for cnt in contours:
    cv2.drawContours(segmented_image, [cnt], -1, 255, thickness=cv2.FILLED)

# Display the results
plt.figure(figsize=(10, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Filtered Image
plt.subplot(2, 2, 2)
plt.imshow(image_filtered, cmap='gray')
plt.title("Filtered Image")
plt.axis('off')

# Gabor Filtered Image
plt.subplot(2, 2, 3)
plt.imshow(gabor_features[0], cmap='gray')
plt.title("Gabor Filtered Image")
plt.axis('off')

# Segmented Image
plt.subplot(2, 2, 4)
plt.imshow(segmented_image, cmap='gray')
plt.title("Segmented Image")
plt.axis('off')

plt.tight_layout()
plt.show()
