import numpy as np
import cv2

# read all images
camera_man_img = cv2.imread('images/camera_man_w_noise.jpg', 0)
salt_and_pepper_img = cv2.imread('images/SaltAndPepperNoise.jpg', 0)
gaussian_noise_img = cv2.imread('images/GaussianNoise.jpg', 0)
uniform_noise_img = cv2.imread('images/UniformNoise.jpg', 0)

for img in (camera_man_img, salt_and_pepper_img, gaussian_noise_img, uniform_noise_img):
    # guard against unknown image paths, which results in img = None
    if img is None:
        raise FileNotFoundError('Could not read image')

    # gaussian blur vs regular blur
    gaussian_blur_5x5 = cv2.GaussianBlur(img, (5, 5), 0)
    gaussian_blur_3x3 = cv2.GaussianBlur(img, (3, 3), 0)
    blur = cv2.blur(img, (5, 5))

    # Show base image
    cv2.imshow('Base Image', img)

    # Show filtered images
    cv2.imshow('Gaussian Blur (5x5)', gaussian_blur_5x5)
    cv2.imshow('Gaussian Blur (3x3)', gaussian_blur_3x3)
    cv2.imshow('Blur Filter (5x5)', blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

for img in (camera_man_img, salt_and_pepper_img, gaussian_noise_img, uniform_noise_img):
    # guard against unknown image paths, which results in img = None
    if img is None:
        raise FileNotFoundError('Could not read image')

    kernel_3x3 = np.ones((3, 3), np.uint8)
    kernel_5x5 = np.ones((5, 5), np.uint8)

    # median filter and n-rank filter
    median_filter_3x3 = cv2.medianBlur(img, 3)
    median_filter_5x5 = cv2.medianBlur(img, 5)
    min_filter_3x3 = cv2.erode(img, kernel_3x3)
    min_filter_5x5 = cv2.erode(img, kernel_5x5)

    # Show base image
    cv2.imshow('Base Image', img)

    # Show filtered images
    cv2.imshow('Median Blur (3x3)', median_filter_3x3)
    cv2.imshow('Median Blur (5x5)', median_filter_5x5)
    cv2.imshow('Minimum Filter (3x3)', min_filter_3x3)
    cv2.imshow('Minimum Filter (5x5)', min_filter_5x5)

    cv2.waitKey(0)
    cv2.destroyAllWindows()