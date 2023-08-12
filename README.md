import cv2
import numpy as np

def detect_fire(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper ranges for detecting red/orange color (associated with fire)
    lower_range = np.array([0, 100, 100])
    upper_range = np.array([20, 255, 255])
    
    # Create a mask to isolate fire-like regions
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Apply morphological operations to reduce noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around potential fire regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Filter out small regions
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image

# Path to the input image
input_image_path = 'forest_fire_image.jpg'

# Detect fire in the image
output_image = detect_fire(input_image_path)

# Display the output image
cv2.imshow('Fire Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

