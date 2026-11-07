import cv2
import numpy as np
import os

OUT = 'images/aruco'

# 6x6 bits, 250 possible ids - same dictionary the detector will look for
DICTIONARY = cv2.aruco.DICT_6X6_250
MARKER_ID = 23

# big enough that the printer never has to upscale it (blurry edges hurt corner accuracy)
MARKER_PX = 1000

# width of the black border, in bits. 1 is the standard the detector expects
BORDER_BITS = 1

# white margin around the marker - the detector needs contrast outside the black border
QUIET_ZONE_PX = 200


def make_marker():
    ''' draw the marker bits and pad it with a white quiet zone so it prints cleanly '''
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, MARKER_PX, borderBits=BORDER_BITS)

    page = cv2.copyMakeBorder(marker, QUIET_ZONE_PX, QUIET_ZONE_PX, QUIET_ZONE_PX, QUIET_ZONE_PX,
                              cv2.BORDER_CONSTANT, value=255)
    return marker, page


def check_marker(page):
    ''' detect the marker we just made, so we know the file is good before printing it '''
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(page)

    if ids is None or MARKER_ID not in ids.flatten():
        raise ValueError(f'*ERROR - generated marker {MARKER_ID} was not detected')

    print(f'detected id {ids.flatten().tolist()} - marker is valid')


if __name__ == '__main__':
    marker, page = make_marker()
    check_marker(page)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'marker{MARKER_ID}.png')
    cv2.imwrite(path, page)
    print(f'saved {path}')

    cv2.imshow('marker', cv2.resize(page, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
