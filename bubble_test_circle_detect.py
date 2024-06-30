import cv2
import numpy as np

# Function to detect filled black circles using contours and minEnclosingCircle
def detect_filled_black_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for contour in contours:
        if len(contour) < 5  :
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        area = cv2.contourArea(contour)
        circle_area = np.pi * (radius ** 2)
        
        # Check if the contour is roughly circular and filled with black
        if 0.7 * circle_area < area < 1.3 * circle_area:
            circles.append((center[0], center[1], radius))
    
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

# Detect filled black circles in the image
circles = detect_filled_black_circles(image)

# Create a window and set the mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', output)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
