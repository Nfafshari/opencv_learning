import cv2

'''
openCV basic usage for reading, resizing, rotating, and saving an image.

cv2.imread(), cv2.resize(), cv2.rotate(), cv2.imwrite()
'''

img = cv2.imread('images/camera_man_w_noise.jpg', cv2.IMREAD_COLOR)

# guard against unknown image paths, which results in img = None
if img is None:
    raise FileNotFoundError('Could not read images/camera_man_w_noise.jpg')

### RESIZE
# resize an image by pixels
img = cv2.resize(img, (600, 600))
# OR by scale
img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5) # increase the photo by half its size

### ROTATE
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # rotate 90 counter clockwise

### SAVE AN IMAGE
# cv2.imwrite('new_image.jpg', img)

# show the image in a window
cv2.imshow('Image', img)

# close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
