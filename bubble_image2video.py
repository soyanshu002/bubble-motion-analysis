import cv2
import os

# Define the filtered destination folder and video output path
filtered_destination_folder = 'py test pic/Bubble Circled Edit'
video_output_path = 'output_video.avi'

# Get a list of all the image files in the filtered destination folder
image_files = [f for f in sorted(os.listdir(filtered_destination_folder)) if f.endswith((".jpg", ".png", ".jpeg"))]

# Read the first image to get the frame size
first_image_path = os.path.join(filtered_destination_folder, image_files[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_output_path, fourcc, 100.0, (width, height))

# Write each image to the video
for image_file in image_files:
    image_path = os.path.join(filtered_destination_folder, image_file)
    frame = cv2.imread(image_path)
    video.write(frame)

# Release the VideoWriter object
video.release()

print("Video created successfully!")
