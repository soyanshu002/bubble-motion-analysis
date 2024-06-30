import cv2
import numpy as np
import os
from skimage import measure, morphology

# Define the source and destination folders
source_folder = 'py test pic/Bubble Edited Final'
destination_folder = 'py test pic/Bubble Cleaned Edited Final'
filtered_destination_folder = 'py test pic/Bubble Noise Reduced'

# Create the destination folders if they don't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
if not os.path.exists(filtered_destination_folder):
    os.makedirs(filtered_destination_folder)

# Function to remove small objects
def remove_small_objects(binary_image, min_size):
    # Label connected components
    labels = measure.label(binary_image, connectivity=2)
    
    # Create an output image with only large objects
    output_image = np.zeros_like(binary_image)
    
    for region in measure.regionprops(labels):
        if region.area >= min_size:
            for coordinates in region.coords:
                output_image[coordinates[0], coordinates[1]] = 255

    return output_image

# Function to process an image
def process_image(image_path, output_path, filtered_output_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Apply sharpening filter
    sharpening_kernel = np.array([[-1, -1, -1], 
                                  [-1,  9, -1], 
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

    # Remove gray pixels within the range of 120 to 150
    lower_bound = 85
    upper_bound = 180
    mask = (sharpened >= lower_bound) & (sharpened <= upper_bound)
    sharpened[mask] = 0

    # Create a white background image
    white_background = np.ones_like(sharpened) * 255

    # Create a mask for pixels with gray value less than 85
    low_gray_mask = sharpened < 85

    # Copy these pixels to the white background
    white_background[low_gray_mask] = sharpened[low_gray_mask]

    # Save the initial processed image
    cv2.imwrite(output_path, white_background)

    # Convert to binary image for removing small objects
    _, binary_image = cv2.threshold(white_background, 1, 255, cv2.THRESH_BINARY_INV)

    # Convert to boolean for processing
    binary_image_bool = binary_image.astype(bool)

    # Fill small holes inside the bubbles
    min_hole_size = 200  # Adjust this value based on your image characteristics
    filled_image_bool = morphology.remove_small_holes(binary_image_bool, area_threshold=min_hole_size)

    # Convert filled boolean image back to uint8
    filled_image = filled_image_bool.astype(np.uint8) * 255

    # Remove small objects
    filtered_image = remove_small_objects(filled_image, min_size=45)

    # Invert the image back to original format
    filtered_image = cv2.bitwise_not(filtered_image)

    # Save the filtered image
    cv2.imwrite(filtered_output_path, filtered_image)

# Loop through all images in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        filtered_destination_path = os.path.join(filtered_destination_folder, filename)
        process_image(source_path, destination_path, filtered_destination_path)

print("Processing complete!")
