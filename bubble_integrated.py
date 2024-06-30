import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_frame(frame):
    # Ensure the frame is in 8-bit single-channel format
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# Function to detect filled black circles using contours and minEnclosingCircle
def detect_filled_black_circles(preprocess_frame):
    
    _, thresholded = cv2.threshold(preprocess_frame, 50, 255, cv2.THRESH_BINARY_INV)  

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for contour in contours:
        if len(contour) < 5:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        area = cv2.contourArea(contour)
        circle_area = np.pi * (radius ** 2)
        
        # Check if the contour is roughly circular and filled with black
        if 0.7 * circle_area < area < 1.3 * circle_area:  # Adjusted area check to be more lenient
            circles.append((center[0], center[1], radius))
    
    return circles

def classify_bubbles(circles):
    small_bubbles = []
    medium_bubbles = []
    large_bubbles = []
    
    for circle in circles:
        radius = circle[2]
        if radius < 6:  # Example thresholds, adjust as needed
            small_bubbles.append(circle)
        elif radius < 9:
            medium_bubbles.append(circle)
        else:
            large_bubbles.append(circle)
    return small_bubbles, medium_bubbles, large_bubbles

def calculate_centroids(circles):
    centroids = [(int(circle[0]), int(circle[1])) for circle in circles]
    return centroids

def calculate_velocity(centroids, previous_centroids, fps):
    velocities = []
    for (x1, y1), (x2, y2) in zip(centroids, previous_centroids):
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance * fps
        velocities.append(velocity)
    return velocities

def average_velocity(velocities):
    if velocities:
        return sum(velocities) / len(velocities)
    return 0

# Initialize video capture and get FPS
cap = cv2.VideoCapture('output_video.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
output_filename = 'output_bubbles_video.avi'
output_folder = 'py test pic/Video Output'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, output_filename)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize previous centroids
previous_small_centroids, previous_medium_centroids, previous_large_centroids = [], [], []

# Initialize total velocities and frame count
total_avg_small_velocity = 0
total_avg_medium_velocity = 0
total_avg_large_velocity = 0
frame_count = 0

# Initialize dictionaries to store trajectories
trajectories_small = {}
trajectories_medium = {}
trajectories_large = {}



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    preprocessed = preprocess_frame(frame)
    circles = detect_filled_black_circles(preprocessed)
    small_bubbles, medium_bubbles, large_bubbles = classify_bubbles(circles)
    
    small_centroids = calculate_centroids(small_bubbles)
    medium_centroids = calculate_centroids(medium_bubbles)
    large_centroids = calculate_centroids(large_bubbles)
    
    small_velocities = medium_velocities = large_velocities = []

    if previous_small_centroids:
        small_velocities = calculate_velocity(small_centroids, previous_small_centroids, fps)
        for i, centroid in enumerate(small_centroids):
            if i in trajectories_small:
                trajectories_small[i].append(centroid)
            else:
                trajectories_small[i] = [centroid]
    
    if previous_medium_centroids:
        medium_velocities = calculate_velocity(medium_centroids, previous_medium_centroids, fps)
        for i, centroid in enumerate(medium_centroids):
            if i in trajectories_medium:
                trajectories_medium[i].append(centroid)
            else:
                trajectories_medium[i] = [centroid]
    
    if previous_large_centroids:
        large_velocities = calculate_velocity(large_centroids, previous_large_centroids, fps)
        for i, centroid in enumerate(large_centroids):
            if i in trajectories_large:
                trajectories_large[i].append(centroid)
            else:
                trajectories_large[i] = [centroid]
    
    previous_small_centroids = small_centroids
    previous_medium_centroids = medium_centroids
    previous_large_centroids = large_centroids

    avg_small_velocity = average_velocity(small_velocities)
    avg_medium_velocity = average_velocity(medium_velocities)
    avg_large_velocity = average_velocity(large_velocities)

    # Update total velocities and frame count
    total_avg_small_velocity += avg_small_velocity
    total_avg_medium_velocity += avg_medium_velocity
    total_avg_large_velocity += avg_large_velocity
    frame_count += 1

    # Display the average velocities on the frame
    cv2.putText(frame, f'Small Avg Velocity: {avg_small_velocity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Medium Avg Velocity: {avg_medium_velocity:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Large Avg Velocity: {avg_large_velocity:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw the detected circles
    for circle in circles:
        center = (int(circle[0]), int(circle[1]))
        radius = int(circle[2])
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.circle(frame, center, 2, (0, 0, 255), 3)

    # Write the frame to the output video
    out.write(frame)

    # Show the preprocessed frame and the original frame with velocity information
    cv2.imshow('Preprocessed Frame', preprocessed)
    cv2.imshow('Frame with Velocities', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Print the total average velocities divided by the number of frames
print(f'Total Average Small Velocity: {total_avg_small_velocity / frame_count:.2f}')
print(f'Total Average Medium Velocity: {total_avg_medium_velocity / frame_count:.2f}')
print(f'Total Average Large Velocity: {total_avg_large_velocity / frame_count:.2f}')

# Plot the trajectories
def plot_trajectories(trajectories, title):
    plt.figure()
    for trajectory in trajectories.values():
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

plot_trajectories(trajectories_small, 'Small Bubbles Trajectories')
plot_trajectories(trajectories_medium, 'Medium Bubbles Trajectories')
plot_trajectories(trajectories_large, 'Large Bubbles Trajectories')