from keras.models import load_model
import os 
import cv2
import numpy as np
from collections import Counter
import imutils
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

t = time.localtime()

current_time = time.strftime("%H",t)
print("Current Time =", current_time)

path = os.path.dirname(__file__)
Test_folder_path = path + '\\GTSRB\\Test\\'
Meta_folder_path = path + '\\GTSRB\\Meta\\'
wait_time = 1  # Delay in milliseconds

class_predictions = []
frame_counter = 0
average_interval = 3
x,y,h,w = 0,0,0,0
numbers_all=[]
available_times = [('6','19'),('19','6')]

# Load model
model = load_model(path+'\CNN_model_all')

def check_secondary_sign(frame):
    
    # cv2.imshow('Supplementary reference frame', frame)
    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth out the noise
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply Canny edge detection to detect edges
    edges_img = cv2.Canny(blur_img, 100, 200)
    
    # cv2.imshow('Supplementary edges', edges_img)

    # # Apply morphological closing to fill in any gaps in the edges
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1 , 1))
    # closed_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Supplementary closed edges', closed_img)

    contours, _ = cv2.findContours(edges_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours to find the speed limit sign
    for cnt in contours:
        # Approximate the contour to a polygon
        area = cv2.contourArea(cnt)
        perimeter = 0.03 * cv2.arcLength(cnt, True) # 0.03 is the epsilon (error margin)
        approx = cv2.approxPolyDP(cnt, perimeter, True)
        
        # Get the number of sides of the approximated polygon
        if(len(approx)==4) and area>300:
            x_s, y_s, w_s, h_s = cv2.boundingRect(cnt)
            # Extract the ROI of the secondary sign
            s_sign_roi = frame[y_s:y_s + h_s, x_s:x_s + w_s]
            # Convert the ROI to grayscale
            gray_sign_roi = cv2.cvtColor(s_sign_roi, cv2.COLOR_BGR2GRAY)
            # Perform character recognition using Tesseract
            config = '--psm 8 -c tessedit_char_whitelist=-0123456789h'
            text = pytesseract.image_to_string(gray_sign_roi, config=config)
            parts = text.split('-')
            numbers = []
            for part in parts:
                number = ''.join(filter(str.isdigit, part))  # Extract digits from each part
                numbers.append(number)
            numbers_all.append(numbers)
            # cv2.imshow('Supplementary sign roi', s_sign_roi)

# Open the camera
cap = cv2.VideoCapture(0)
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

    # lower_red3 = np.array([150, 30, 30])
    # upper_red3 = np.array([180, 255, 255])
    # mask3 = cv2.inRange(hsv_frame, lower_red3, upper_red3)

    # lower_red4 = np.array([0, 30, 30])
    # upper_red4 = np.array([10, 255, 255])
    # mask4 = cv2.inRange(hsv_frame, lower_red4, upper_red4)


    # Combine the masks
    mask = cv2.bitwise_or(mask1, mask2)
    # mask = cv2.bitwise_or(mask, mask3)
    # mask = cv2.bitwise_or(mask, mask4)

    # Apply the mask to the frame
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth out the noise
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Apply Canny edge detection to detect edges
    edges_img = cv2.Canny(blur_img, 150, 300)
    # edges_img = cv2.adaptiveThreshold    (blur_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,21,-5)

    # Apply morphological closing to fill in any gaps in the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 , 4))
    closed_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel)

    # Display the video with sign roi rectangle if detected 
    if(x!=0 and y !=0 and w!=0 and h!=0):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x = 0
        y = 0
        h = 0
        w = 0

    
    cv2.imshow('Video', frame)
    if(cv2.waitKey(wait_time) & 0xFF == ord('b')):
        break

    cv2.imshow('Video red masking', masked_img)
    if(cv2.waitKey(wait_time) & 0xFF == ord('b')):
        break

    cv2.imshow('Video edges', edges_img)
    if(cv2.waitKey(wait_time) & 0xFF == ord('b')):
        break

    # cv2.imshow('Video closed edges', closed_img)
    # if(cv2.waitKey(wait_time) & 0xFF == ord('b')):
    #     break

    # Find contours in the frame
    contours, _ = cv2.findContours(closed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Loop over the contours to find the speed limit sign
    for cnt in contours:
        # Get the area and perimeter of the contour
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Compute the circularity of the contour
        if(perimeter==0):continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        # If the contour is circular enough and its area is within a certain range, it's likely a speed limit sign
        
        if circularity > 0.8 and area > 50: #and area < 5000
            # Draw a bounding box around the sign
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract possible ROI for secondary sign  
            x_l = x-int(0.5*w)
            x_r = x+w+int(0.5*w)
            y_l = y+int(0.9*h)
            y_u = y+h+h
            # Check if outside of bounds
            if(x_l<0):
                x_l=0
            if(x_r>=960):
                x_r = 959
            if(y_l<0):
                y_l=0
            if(y_u>=540):
                y_u = 539
            check_secondary_sign(frame[y_l:y_u ,x_l:x_r])

            # Extract the ROI of the sign
            sign_roi = frame[y: y + h, x: x + w]
            sign_roi = cv2.resize(sign_roi , (40, 40))
            sign_roi = sign_roi.reshape(1, 40, 40 ,3)  
            
            result = model.predict(sign_roi)   
            class_pred = np.argmax(result,axis=1)
            class_predictions.append(str(class_pred))



            # Check if the average needs to be calculated
            frame_counter += 1
            if frame_counter == average_interval:
                average_pred = Counter(class_predictions).most_common(1)
                final_prediction = average_pred[0][0][1]
                if(numbers_all):
                        list_counter = Counter(tuple(lst) for lst in numbers_all)
                        average_time_pred = list_counter.most_common(1)[0][0]
                        if(len(average_time_pred)==2):
                            if(average_time_pred == available_times[0]):
                                if(int(current_time)>int(average_time_pred[0]) and int(current_time)<int(average_time_pred[1])):
                                    final_prediction = '7'
                                else:
                                    final_prediction = '130'
                            elif(average_time_pred == available_times[1]):
                                if(int(current_time)>int(average_time_pred[0]) and int(current_time)<int(average_time_pred[1])):
                                    final_prediction = '8'
                                else:
                                    final_prediction = '7'           
                if(final_prediction != '6'):
                    class_path = Meta_folder_path+final_prediction+'.png'
                    warning = cv2.imread(class_path)
                    warning = imutils.resize(warning, width=720)                   
                    cv2.imshow("Warning", warning)
                # else:
                #     cv2.imshow("Warning", np.empty_like(frame))
                print(f"Average class prediction: {final_prediction}")
                class_predictions = []
                numbers_all=[]
                frame_counter = 0
