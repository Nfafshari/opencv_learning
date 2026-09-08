import numpy as np
import cv2

img = cv2.imread('images/coins1.jpg', 0)
penny_template = cv2.imread('images/penny.png', 0)
quarter_template = cv2.imread('images/quarter.png', 0)

# guard against unknown image paths, which results in img = None
if img is None or penny_template is None or quarter_template is None:
    raise FileNotFoundError('Could not read image')

penny_h, penny_w = penny_template.shape # only has height and width because we loaded it as grayscale
quarter_h, quarter_w = quarter_template.shape

# main methods of template matching
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# Try with all of them, and see which one works the best and use that
for method in methods:
    img2 = img.copy()

    penny_result = cv2.matchTemplate(img2, penny_template, method) # result retruns = (W - w + 1, H - h + 1)
    quarter_result = cv2.matchTemplate(img2, quarter_template, method)
    penny_min_val, penny_max_val, penny_min_loc, penny_max_loc = cv2.minMaxLoc(penny_result)
    quarter_min_val, quarter_max_val, quarter_min_loc, quarter_max_loc = cv2.minMaxLoc(quarter_result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        penny_location = penny_min_loc
        quarter_location = quarter_min_loc
    else:
        penny_location = penny_max_loc
        quarter_location = quarter_max_loc

    penny_bottom_right = (penny_location[0] + penny_w, penny_location[1] + penny_h)
    quarter_bottom_right = (quarter_location[0] + quarter_w, quarter_location[1] + quarter_h)

    cv2.rectangle(img2, penny_location, penny_bottom_right, 255, 4)
    cv2.rectangle(img2, quarter_location, quarter_bottom_right, 255, 4)

    cv2.imshow('match', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()