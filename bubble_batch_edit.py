import cv2
import os

# Function to resize, crop images from the center, and save serial wise
def resize_and_crop_images(input_folder, output_folder, resize_dim, crop_width, crop_height):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # Check if image was successfully loaded
            if img is None:
                print(f"Failed to load image {filename}. Skipping.")
                continue

            # Resize the image
            resized_img = cv2.resize(img, resize_dim)

            # Get resized image dimensions
            img_height, img_width = resized_img.shape[:2]

            # Calculate the center crop coordinates
            center_x, center_y = img_width // 2, img_height // 2
            x1 = max(center_x - crop_width // 2, 0)
            y1 = max(center_y - crop_height // 2, 0)
            x2 = min(center_x + crop_width // 2, img_width)
            y2 = min(center_y + crop_height // 2, img_height)

            # Crop the image
            cropped_img = resized_img[y1:y2, x1:x2]

            # Display the cropped image
            cv2.imshow('Cropped Image', cropped_img)

            # Wait for user input
            key = cv2.waitKey(0) & 0xFF  # Get lower byte for compatibility

            if key == ord('1'):
                # Save the cropped image to the output folder if '1' is pressed
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_img)
                print(f"Cropped and saved {filename}")
            elif key == 27:  # ESC key
                # Don't save the image if ESC is pressed
                print(f"Skipped saving {filename}")

            # Destroy the window
            cv2.destroyAllWindows()

# Example usage
input_folder = 'py test pic/Bubble Img Frames'
output_folder = 'py test pic/Bubble Img Edited'
resize_dim = (800, 800)  # Resize dimensions (width, height)
crop_width, crop_height = 440, 440  # Crop dimensions (width, height)

resize_and_crop_images(input_folder, output_folder, resize_dim, crop_width, crop_height)
