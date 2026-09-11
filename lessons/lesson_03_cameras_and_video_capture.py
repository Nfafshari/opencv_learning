import cv2
import numpy as np

# Capture camera video using cv2
cap = cv2.VideoCapture(0)

# OR use an mp4 video
# cap = cv2.VideoCapture('my_video.mp4')

while True:
    ret, frame = cap.read() # returns the frame (an image), and "ret" tells us whether it worked correctly or not

    # Get width and height of our video capture (3 is the width, 4 is height)
    width = int(cap.get(3))
    height = int(cap.get(4))

    # create a blank image
    image = np.zeros(frame.shape, np.uint8)

    # Resize the frame so that we can copy and paste it 4 times
    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # slice the smaller frame into our blank image
    image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180) # TOP LEFT
    image[height//2:, :width//2] = smaller_frame # BOTTOM LEFT
    image[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180) # TOP RIGHT
    image[height//2:, width//2:] = smaller_frame # BOTTOM RIGHT


    cv2.imshow('frame', image)

    if cv2.waitKey(1) == ord('q'):
        break

# release the video capture so other applications can use it
cap.release()
cv2.destroyAllWindows()