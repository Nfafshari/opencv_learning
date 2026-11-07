import cv2
import numpy as np
import glob
import os

FOLDER = 'images/hw3'
OUT = 'images/hw3_results'

# the photos are 1500x2000, half size is plenty and keeps everything fast
SCALE = 0.5

# Lowe's ratio test value from the SIFT paper
RATIO = 0.75

# RANSAC inlier tolerance in pixels
RANSAC_PX = 4.0

# figures get scaled down to this width so they fit on a screen
FIGURE_WIDTH = 1600


def show(name, img):
    ''' save the full size result and show a screen sized copy '''
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, name + '.jpg'), img)

    scale = min(1.0, FIGURE_WIDTH / img.shape[1])
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_images():
    ''' part 1 - the overlapping sequence, in file name order (left to right) '''
    paths = sorted(glob.glob(os.path.join(FOLDER, '*.jpeg')))
    images = [cv2.resize(cv2.imread(p), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
              for p in paths]

    if len(images) < 4:
        raise ValueError(f'*ERROR - need at least 4 images in {FOLDER}, found {len(images)}')

    return images


def find_homography(img1, img2):
    ''' SIFT + ratio test + RANSAC. returns H mapping img1 -> img2 and the inlier points '''
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    pairs = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in pairs if m.distance < RATIO * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_PX)
    inliers = mask.ravel() == 1

    return H, pts1[inliers], pts2[inliers]


def project(points, H):
    ''' push Nx2 points through a homography '''
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), H).reshape(-1, 2)


def overlap(img1, img2, H):
    ''' fraction of img2 that img1 covers once warped into it '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    mask = cv2.warpPerspective(np.ones((h1, w1), np.uint8), H, (w2, h2))
    return mask.mean()


def check_figure(img1, img2, pts1, H, count=25):
    '''
    part 2 proof - points picked in img1 (left) are projected through H and drawn
    in img2 (right) with the same color. if H is right they land on the same thing
    '''
    figure = np.hstack([img1, img2])
    offset = img1.shape[1]

    rng = np.random.default_rng(0)
    picked = pts1[rng.choice(len(pts1), min(count, len(pts1)), replace=False)]
    landed = project(picked, H)

    for p, q in zip(picked, landed):
        color = tuple(int(c) for c in rng.integers(60, 256, 3))
        cv2.circle(figure, tuple(np.int32(p)), 8, color, 2, cv2.LINE_AA)
        cv2.circle(figure, tuple(np.int32(q) + [offset, 0]), 8, color, 2, cv2.LINE_AA)
        cv2.line(figure, tuple(np.int32(p)), tuple(np.int32(q) + [offset, 0]), color, 1, cv2.LINE_AA)

    return figure


def homographies_to_reference(images, ref):
    '''
    part 2 - homography from every image into the reference (middle) image.
    each image is matched to its neighbour on the way to the reference and the
    neighbour homographies are chained together
    '''
    to_ref = {ref: np.eye(3)}

    for i in list(range(ref - 1, -1, -1)) + list(range(ref + 1, len(images))):
        step = i + 1 if i < ref else i - 1
        H, pts1, pts2 = find_homography(images[i], images[step])

        error = np.linalg.norm(project(pts1, H) - pts2, axis=1)
        print(f'image {i} -> {step}: {len(pts1)} inliers, reprojection error '
              f'mean {error.mean():.2f}px  max {error.max():.2f}px, '
              f'overlap {overlap(images[i], images[step], H) * 100:.0f}%')

        show(f'2_check_{i}_to_{step}', check_figure(images[i], images[step], pts1, H))

        to_ref[i] = to_ref[step] @ H

    for i in range(len(images)):
        print(f'\nH image {i} -> reference {ref}:\n{np.round(to_ref[i], 4)}')

    return to_ref


def canvas_size(images, homographies):
    '''
    part 3 - push every image's corners into the reference frame and take the
    bounding box. returns the canvas size and the translation that moves the box
    so nothing lands at a negative coordinate
    '''
    corners = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        box = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        corners.append(project(box, H))

    corners = np.vstack(corners)
    x_min, y_min = np.floor(corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(corners.max(axis=0)).astype(int)

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], np.float64)
    return (x_max - x_min, y_max - y_min), T


def warp_image(img, H):
    '''
    part 4 - warp one image onto a blank canvas made just big enough to hold all of it
    '''
    size, T = canvas_size([img], [H])
    warped = cv2.warpPerspective(img, T @ H, size)

    # containment check - every warped corner has to be inside the canvas
    h, w = img.shape[:2]
    box = project(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), T @ H)
    inside = np.all((box >= 0) & (box <= np.array(size)))
    print(f'\nwarp test: canvas {size[0]}x{size[1]}, all corners inside: {inside}')

    # outline where the image edge should be, so it is visible nothing was cut off
    cv2.polylines(warped, [np.int32(box)], True, (0, 0, 255), 3, cv2.LINE_AA)
    return warped


def stitch(images, homographies):
    '''
    part 5 - warp every image onto one canvas and feather blend. each pixel is
    weighted by its distance to its own image's border, so seams fade instead of cut
    '''
    size, T = canvas_size(images, homographies)
    print(f'\npanorama canvas: {size[0]}x{size[1]}')

    total = np.zeros((size[1], size[0], 3), np.float64)
    weights = np.zeros((size[1], size[0]), np.float64)

    for img, H in zip(images, homographies):
        h, w = img.shape[:2]

        # 1 inside, 0 on the one pixel border, so the distance grows toward the middle
        mask = np.zeros((h, w), np.uint8)
        mask[1:-1, 1:-1] = 1
        weight = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        warped = cv2.warpPerspective(img, T @ H, size).astype(np.float64)
        weight = cv2.warpPerspective(weight, T @ H, size)

        total += warped * weight[..., None]
        weights += weight

    return (total / np.maximum(weights, 1e-6)[..., None]).astype(np.uint8)


def main():
    images = load_images()
    ref = len(images) // 2
    print(f'****** {len(images)} images from {FOLDER}, reference image {ref} ******\n')

    to_ref = homographies_to_reference(images, ref)
    homographies = [to_ref[i] for i in range(len(images))]

    size, _ = canvas_size(images, homographies)
    print(f'\npart 3 - panorama needs a {size[0]}x{size[1]} image')

    show('4_warp_test', warp_image(images[0], homographies[0]))
    show('5_panorama', stitch(images, homographies))

    print('**************************')


if __name__ == '__main__':
    main()
