import cv2
import numpy as np
import os


# Define the source and destination folders
source_folder = 'py test pic/Bubble Noise Reduced'
destination_folder = 'py test pic/Bubble Circled Edit'


# Create the destination folders if they don't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

def process_image(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve bubble detection
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Apply threshold to get binary image
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a white background image
    white_background = np.ones_like(image) * 255

    # Draw circles for each contour
    for contour in contours:
        # Calculate the area
        area = cv2.contourArea(contour)
        
        # Calculate the radius of the circle with the same area
        radius = int(np.sqrt(area / np.pi))
        
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Draw the circle at the same coordinates as the contour
        cv2.circle(white_background, (cx, cy), radius, (0, 0, 0), -1)

        cv2.imwrite(output_path, white_background)


# Loop through all images in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        # filtered_destination_path = os.path.join(filtered_destination_folder, filename)
        process_image(source_path, destination_path)

print("Processing complete!")

