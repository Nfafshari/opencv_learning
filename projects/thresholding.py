import cv2

coins_1_img = cv2.imread('images/coins1.jpg', 0)
coins_2_img = cv2.imread('images/coins2.jpg', 0)
screws_img = cv2.imread('images/screws.jpeg', 0)

# simple thresholdings
for img in (coins_1_img, coins_2_img, screws_img):
    # guard against unknown image paths, which results in img = None
    if img is None:
        raise FileNotFoundError('Could not read image')

    # simple thresh using different methods
    ret,thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    
    for i in range(6):
        cv2.imshow(titles[i], images[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# adaptive thresholdings
for img in (coins_1_img, coins_2_img, screws_img):
    # guard against unknown image paths, which results in img = None
    if img is None:
        raise FileNotFoundError('Could not read image')

    # median blur first so the adaptive windows aren't thrown off by noise
    blurred = cv2.medianBlur(img, 5)

    # adaptive thresh using different methods
    ret,th1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    titles = ['Original Image','Global Thresh (127)','Adaptive Mean','Adaptive Gaussian']
    images = [img, th1, th2, th3]

    for title, image in zip(titles, images):
        cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()