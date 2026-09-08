import cv2
import numpy as np

img = cv2.imread('images/pattern_chessboard.png')

# guard against unknown image paths, which results in img = None
if img is None:
    raise FileNotFoundError('Could not read image')

img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# Convert to grayscale, this simplifies the algorithm detection
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Parametes: 
# * image source, 
# * number of good corners, 
# * minimum corner quality, 
# * minimum euclidean distance
corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.5, 20)
corners = corners.astype(np.intp) # cast values to integers

for corner in corners:
    x, y = corner.ravel() # flatten the array (ie [[[0, 1, 2], ...]] = [0, 1, 2, ...])

    # draw the corner
    cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

# Cropping white space of the image used out
cropped_image = img[0:600, 0:800]

# Draw random lines between each corner
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3))) # map each value as an integer and make it a tuple
        cv2.line(img, corner1, corner2, color, 1)

cv2.imshow('image', cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
