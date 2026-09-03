import cv2
import random

# you can pass -1, 0, 1, for flags IMREAD_UNCHANGED, IMREAD_GRAYSCALE, and IMREAD_COLOR, respectively
img = cv2.imread('images/camera_man_w_noise.jpg', 1)
img2 = cv2.imread('images/camera_man_w_noise.jpg', 1)

# guard against unknown image paths, which results in img = None
if img is None:
    raise FileNotFoundError('Could not read images/camera_man_w_noise.jpg')

if img2 is None:
    raise FileNotFoundError('Could not read images/camera_man_w_noise.jpg')

print(f'''
Note!
imread() returns a numpy array, which is an optimized library for python arrays.
you can get the shape of an array using array.shape which gives you the:
* num of rows
* num of cols
* num of channels

numpy array allows for more array functionality
''')

print(f'0. Rows (height) - {img.shape[0]}\n1. Columns (width) - {img.shape[1]}\n2. Channels (RGB or BGR) - {img.shape[2]}\nFULL: {img.shape}')

print(f'first row of our image (showing only first 25 pixels): {img[0][0:25]}')

print(f'lets change the colors of the pixels!')

for i in range(100):
    for j in range(img.shape[1]):
        img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

print(f'Now lets copy one part of the image to another')

# Numpy array slice allows to slice twice
# INDEX 0 = ROWS
# INDEX 1 = COLUMNS
camera = img[100:200, 50:200]
# REPLACE SECTION
img[0:100, 75:225] = camera

cv2.imshow("Image", img)
cv2.imshow("Image 2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()