import cv2
from keras.models import load_model
import numpy as np

class_labels = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}

x, y, h, w = 0, 0, 0, 0
cap = cv2.VideoCapture(0)
model = load_model('traffic_sign_model.h5')

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red1 = np.array([160, 10, 10])
    upper_red1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)

    lower_red2 = np.array([0, 10, 10])
    upper_red2 = np.array([9, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    # Combine the masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the frame
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth out the noise
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Apply Canny edge detection to detect edges
    edges_img = cv2.Canny(blur_img, 150, 300)

    # Apply morphological closing to fill in any gaps in the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    closed_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel)

    # Display the video with sign roi rectangle if detected
    if (x != 0 and y != 0 and w != 0 and h != 0):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x = 0
        y = 0
        h = 0
        w = 0

    cv2.imshow('Video', frame)

    contours, _ = cv2.findContours(closed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Loop over the contours to find the speed limit sign
    for cnt in contours:
        # Get the area and perimeter of the contour
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Compute the circularity of the contour
        if (perimeter == 0): continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        # If the contour is circular enough and its area is within a certain range, it's likely a speed limit sign

        if circularity > 0.8 and area > 50:  # and area < 5000
            # Draw a bounding box around the sign
            x, y, w, h = cv2.boundingRect(cnt)
            # Extract the ROI of the sign
            sign_roi = frame[y: y + h, x: x + w]
            sign_roi = cv2.resize(sign_roi, (32, 32))
            sign_roi = sign_roi.reshape(1, 32, 32, 3)

            # Apply model prediction
            prediction = model.predict(sign_roi)
            predicted_class = np.argmax(prediction)
            if predicted_class < 9:
                print("The model predicts: " + class_labels[predicted_class])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
