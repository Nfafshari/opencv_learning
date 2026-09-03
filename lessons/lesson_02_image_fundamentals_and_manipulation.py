import cv2
import numpy

# you can pass -1, 0, 1, for flags IMREAD_UNCHANGED, IMREAD_GRAYSCALE, and IMREAD_COLOR, respectively
img = cv2.imread('images/camera_man_w_noise.jpg', 1)


print(f"""
Note!
imread() returns a numpy array, which is an optimized library for python arrays.
you can get the shape of an array using array.shape which gives you the:
* num of rows
* num of cols
* num of channels
""")
