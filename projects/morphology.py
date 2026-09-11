import cv2
import numpy as np

# read all images
camera_man_img = cv2.imread('images/camera_man_w_noise.jpg', 0)
salt_and_pepper_img = cv2.imread('images/SaltAndPepperNoise.jpg', 0)
gaussian_noise_img = cv2.imread('images/GaussianNoise.jpg', 0)
uniform_noise_img = cv2.imread('images/UniformNoise.jpg', 0)

# masks (structuring elements) to run every operation with
kernel_3x3 = np.ones((3, 3), np.uint8)
kernel_5x5 = np.ones((5, 5), np.uint8)

# a mask bigger than any object, used to estimate the background lighting
background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))

# morphology on the segmented images from the thresholding problem
for img in (camera_man_img, salt_and_pepper_img, gaussian_noise_img, uniform_noise_img):
    # guard against unknown image paths, which results in img = None
    if img is None:
        raise FileNotFoundError('Could not read image')

    blurred = cv2.medianBlur(img, 5)

    # black hat subtracts the background lighting so one global threshold works
    # everywhere, the objects come out white on black
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, background_kernel)
    ret, segmented = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for kernel, size in ((kernel_3x3, '3x3'), (kernel_5x5, '5x5')):
        # single operations
        erosion = cv2.erode(segmented, kernel)
        dilation = cv2.dilate(segmented, kernel)
        opening = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)
        gradient = cv2.morphologyEx(segmented, cv2.MORPH_GRADIENT, kernel)

        # combined operations, the order the two are applied in changes the result
        open_then_close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        close_then_open = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # use the cleaned up masks to pull the objects back out of the original image
        open_then_close_result = cv2.bitwise_and(img, img, mask=open_then_close)
        close_then_open_result = cv2.bitwise_and(img, img, mask=close_then_open)

        titles = ['Original Image', 'Black Hat', 'Segmented',
                  'Erosion ' + size, 'Dilation ' + size,
                  'Opening ' + size, 'Closing ' + size,
                  'Gradient ' + size,
                  'Open then Close ' + size, 'Close then Open ' + size,
                  'Open then Close Applied ' + size, 'Close then Open Applied ' + size]
        images = [img, blackhat, segmented,
                  erosion, dilation,
                  opening, closing,
                  gradient,
                  open_then_close, close_then_open,
                  open_then_close_result, close_then_open_result]

        for title, image in zip(titles, images):
            cv2.imshow(title, image)

        # count the objects left in each mask so the orders can be compared with a number
        for title, mask in (('Open then Close ' + size, open_then_close),
                            ('Close then Open ' + size, close_then_open)):
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            objects = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 200]

        cv2.waitKey(0)
        cv2.destroyAllWindows()
