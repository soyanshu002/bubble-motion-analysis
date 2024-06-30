import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage as ndi



img = cv2.imread('py test pic/Test Edits/modified_img.jpg')
img1 = cv2.imread('py test pic/Bubbble_img_2.png')
img2 = cv2.imread('py test pic/Bubble Img Edited/bubbleimg (1).jpg')



blank = np.zeros((500,500), dtype= 'uint8')

# cv.imshow('Bubble Image', img)
# cv.imshow('Bubble Image1', img1)

# cv.imshow('I am Groot', img)
# resized = cv2.resize(img2, (500,500), interpolation= cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Ensure the image is binary by applying a threshold
_, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Convert to boolean for processing
binary_image_bool = binary_image.astype(bool)

# Fill small holes inside the bubbles
min_hole_size = 150  # Adjust this value based on your image characteristics
filled_image_bool = morphology.remove_small_holes(binary_image_bool, area_threshold=min_hole_size)

# Convert filled boolean image back to uint8
filled_image = filled_image_bool.astype(np.uint8) * 255

# Label connected components
labeled_image, num_labels = measure.label(filled_image_bool, connectivity=2, background=0, return_num=True)

# Get properties of labeled regions
regions = measure.regionprops(labeled_image)
print(f"Number of labels found: {num_labels}")

# Convert grayscale filled image to BGR for drawing purposes
colored_labeled_image = cv2.cvtColor(filled_image, cv2.COLOR_GRAY2BGR)

# Invert the colors of the image
inverted_image = cv2.bitwise_not(colored_labeled_image)

# Define size ranges
micro_size_threshold = 45  # New threshold for micro bubbles
small_size_threshold = 150  # Adjust these thresholds based on your image characteristics
medium_size_threshold = 300

# Initialize category filter (0 = micro, 1 = small, 2 = medium, 3 = large)
category_filter = 0

# Function to categorize size
def categorize_size(size):
    if size <= micro_size_threshold:
        return 'micro'
    elif size <= small_size_threshold:
        return 'small'
    elif size <= medium_size_threshold:
        return 'medium'
    else:
        return 'large'

# Function to display bubble sizes and draw rectangles on mouse events
def show_pixel_values(event, x, y, flags, param):
    global image, inverted_image, category_filter

    if event == cv2.EVENT_MOUSEMOVE:
        # Create a copy of the image to draw the rectangle
        img_copy = inverted_image.copy()

        # Check if the cursor is within any bubble
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            if minc <= x <= maxc and minr <= y <= maxr:
                # Draw a rectangle around the bubble
                cv2.rectangle(img_copy, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
                # Calculate the size of the bubble
                size = region.area
                # Categorize the size
                size_category = categorize_size(size)
                # Display bubble size category
                cv2.putText(img_copy, f'Size: {size} ({size_category})', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                break

        # Show the image with rectangle and bubble size category
        cv2.imshow('Image', img_copy)

# Function to clean image based on size category
def clean_image(category_filter):
    global labeled_image, regions

    # Create a mask to keep the desired categories
    mask = np.zeros(labeled_image.shape, dtype=np.uint8)

    for region in regions:
        size = region.area
        size_category = categorize_size(size)
        
        if category_filter == 0 and size_category == 'micro':
            mask[labeled_image == region.label] = 255
        elif category_filter == 1 and size_category == 'small':
            mask[labeled_image == region.label] = 255
        elif category_filter == 2 and size_category == 'medium':
            mask[labeled_image == region.label] = 255
        elif category_filter == 3 and size_category == 'large':
            mask[labeled_image == region.label] = 255

    return mask

# Create a window and set the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', show_pixel_values)

# Display the initial inverted image
cv2.imshow('Image', inverted_image)

# Key press loop to switch between categories and clean image
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('0'):
        category_filter = 0  # Keep micro bubbles
    elif key == ord('1'):
        category_filter = 1  # Keep small bubbles
    elif key == ord('2'):
        category_filter = 2  # Keep medium bubbles
    elif key == ord('3'):
        category_filter = 3  # Keep large bubbles
    elif key == 27:  # ESC key to exit
        break

    # Clean the image based on the selected category
    cleaned_image = clean_image(category_filter)
    cleaned_colored_image = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
    inverted_cleaned_image = cv2.bitwise_not(cleaned_colored_image)

    # Display the cleaned and inverted image
    cv2.imshow('Image', inverted_cleaned_image)

cv2.destroyAllWindows()