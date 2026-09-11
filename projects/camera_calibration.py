import cv2
import numpy as np
from pathlib import Path
from typing import NamedTuple

class CalibrationResult(NamedTuple):
    views: int          # how many images actually contributed corners
    rms: float          # RMS reprojection error in pixels, lower is better
    mtx: np.ndarray     # camera matrix, the intrinsics
    dist: np.ndarray    # distortion coefficients
    rvecs: tuple        # per-view rotation vectors
    tvecs: tuple        # per-view translation vectors

def calibrate_camera(pathname: Path, obj_points, img_points, dataset_size: int):
    # Termination criteria for ending our algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points for reading
    objp = np.zeros((7*7, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

    # read all images from /images
    i = 0
    image_size = None
    for path in pathname.rglob("*"):
        if i >= dataset_size:
            break

        img = cv2.imread(path)

         # guard against unknown image paths, which results in img = None
        if img is None:
            raise FileNotFoundError('Could not read image')

        # Scale image down
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        # Convert image color to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # calibrateCamera wants (width, height), shape gives (height, width)
        image_size = gray_img.shape[::-1]

        # Get corners of chessboard in a 7x7 grid
        ret, corners = cv2.findChessboardCorners(gray_img, (7,7), None)

        # Check if corners were found and add object and image points
        if ret == True:
            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

            # draw the refined corners onto a copy so the original stays clean
            preview = img.copy()
            cv2.drawChessboardCorners(preview, (7, 7), corners2, ret)
            cv2.imshow("image", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # increment dataset counter
            i += 1

    # nothing detected means nothing to calibrate from
    if image_size is None or not img_points:
        raise ValueError(f'No chessboard corners found in {pathname}')

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    return CalibrationResult(len(img_points), rms, mtx, dist, rvecs, tvecs)
        

def main():
    # per dataset points
    dataset_1_obj_points = [] # 3D points
    dataset_1_img_points = [] # 2D points

    dataset_2_obj_points = [] # 3D points
    dataset_2_img_points = [] # 2D points

    dataset_3_obj_points = [] # 3D points
    dataset_3_img_points = [] # 2D points

    dataset_4_obj_points = [] # 3D points
    dataset_4_img_points = [] # 2D points

    dataset_5_obj_points = [] # 3D points
    dataset_5_img_points = [] # 2D points

    images_path = Path("images/chessboard_images")

    # pass each dataset size
    dataset_1 = calibrate_camera(images_path, dataset_1_obj_points, dataset_1_img_points, 2)
    dataset_2 = calibrate_camera(images_path, dataset_2_obj_points, dataset_2_img_points, 5)
    dataset_3 = calibrate_camera(images_path, dataset_3_obj_points, dataset_3_img_points, 10)
    dataset_4 = calibrate_camera(images_path, dataset_4_obj_points, dataset_4_img_points, 15)
    dataset_5 = calibrate_camera(images_path, dataset_5_obj_points, dataset_5_img_points, 25)

    # compare reprojection error as the number of views grows
    for result in (dataset_1, dataset_2, dataset_3, dataset_4, dataset_5):
        print(f'{result.views} views = rms {result.rms} px')

if __name__ == "__main__":
    main()
