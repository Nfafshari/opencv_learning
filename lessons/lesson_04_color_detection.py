import cv2
import numpy as np

# Capture camera video using cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Get width and height of our video capture (3 is the width, 4 is height)
    width = int(cap.get(3))
    height = int(cap.get(4))

    # convert our color scheme to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # find the color we want to extract. We need 2 values, the lower bound of the color 
    # and the upper bound of the color
    lower_red = np.array([0, 100, 70])  
    upper_red = np.array([10, 255, 255])

    # create a mask
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # apply the mask, pass same image twice and the mask
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # BITWISE AND
    # 1 AND 1 = 1
    # 0 AND 1 = 0
    # 1 AND 0 = 0
    # 0 AND 0 = 0
    # SO
    # if red pixel = pixel is found it will display the red pixel only

    cv2.imshow('frame', result)

    if cv2.waitKey(1) == ord('q'):
        break

# release the video capture so other applications can use it
cap.release()
cv2.destroyAllWindows()