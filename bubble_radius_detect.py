import cv2
import numpy as np

# Function to detect circles using HoughCircles
def detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles

# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global circles, output
    if event == cv2.EVENT_MOUSEMOVE:
        output = image.copy()
        if circles is not None:
            for (cx, cy, r) in circles:
                distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                if distance <= r:
                    cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
                    cv2.putText(output, f'Radius: {r}', (cx - r, cy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break

# Load the image
image = cv2.imread('py test pic/Bubble Circled Edit/bubbleimg (1).jpg')
output = image.copy()

# Default parameters for circle detection
dp = 1.2
minDist = 30
param1 = 50
param2 = 30
minRadius = 5
maxRadius = 100

# Detect circles in the image with adjustable parameters
circles = detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius)

# Create a window and set the mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

def on_trackbar(val):
    global dp, minDist, param1, param2, minRadius, maxRadius
    dp = cv2.getTrackbarPos('dp', 'image') / 10.0
    minDist = cv2.getTrackbarPos('minDist', 'image')
    param1 = cv2.getTrackbarPos('param1', 'image')
    param2 = cv2.getTrackbarPos('param2', 'image')
    minRadius = cv2.getTrackbarPos('minRadius', 'image')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'image')
    global circles
    circles = detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius)

cv2.createTrackbar('dp', 'image', int(dp * 10), 20, on_trackbar)
cv2.createTrackbar('minDist', 'image', minDist, 100, on_trackbar)
cv2.createTrackbar('param1', 'image', param1, 100, on_trackbar)
cv2.createTrackbar('param2', 'image', param2, 100, on_trackbar)
cv2.createTrackbar('minRadius', 'image', minRadius, 50, on_trackbar)
cv2.createTrackbar('maxRadius', 'image', maxRadius, 200, on_trackbar)

on_trackbar(0)  # Initial call to set up the parameters

while True:
    cv2.imshow('image', output)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
