import cv2
import numpy as np
import glob
import os
import re

# photos of the marker, taken in order while the camera slides along a tape measure.
# put the tape reading in each file name so the program can check itself, e.g.
#   00_0cm.jpeg, 01_5cm.jpeg, 02_10cm.jpeg ... 06_30cm.jpeg
FOLDER = 'images/hw4'
OUT = 'images/hw4_results'

# chessboard photos from the same phone, used for the camera intrinsics
CALIB_FOLDER = 'images/chessboard_images'
CHESSBOARD = (7, 7)

# the chessboard photos are 1500x2000, half size is plenty and keeps everything fast
SCALE = 0.5

# must match make_aruco_marker.py
DICTIONARY = cv2.aruco.DICT_6X6_250
MARKER_ID = 23

# outer black square of the printout, measured with a ruler, in meters
MARKER_SIZE_M = 0.198

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


def load(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'*ERROR - could not read {path}')
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def calibrate():
    ''' intrinsics from the chessboard photos, at the same scale as the marker photos.
        the square size doesn't matter here, only K and the distortion are kept '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

    obj_points, img_points = [], []
    size = None
    for path in sorted(glob.glob(os.path.join(CALIB_FOLDER, '*.jpeg'))):
        gray = cv2.cvtColor(load(path), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
        if found:
            obj_points.append(objp)
            img_points.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

    if not img_points:
        raise ValueError(f'*ERROR - no chessboard corners found in {CALIB_FOLDER}')

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(f'calibration - {len(img_points)} views, rms {rms:.3f} px, image {size[0]}x{size[1]}')
    return K, dist, size


def load_frames(calib_size):
    ''' the marker photos in file name order, with the tape reading if the name has one '''
    paths = sorted(glob.glob(os.path.join(FOLDER, '*.jp*g')) + glob.glob(os.path.join(FOLDER, '*.png')))
    if len(paths) < 2:
        raise ValueError(f'*ERROR - need at least 2 images in {FOLDER}, found {len(paths)}')

    frames = []
    for path in paths:
        img = load(path)

        # K only holds for the resolution it was calibrated at. a full res photo from the
        # same camera is just a bigger copy of the same view, so shrinking it to match is fine.
        # a different shape means a crop or another camera mode, and resizing can't fix that
        w, h = img.shape[1::-1]
        if abs(w / h - calib_size[0] / calib_size[1]) > 0.01:
            raise ValueError(f'*ERROR - {path} is {w}x{h}, calibration was '
                             f'{calib_size[0]}x{calib_size[1]} - different shape, check the orientation')
        if (w, h) != calib_size:
            img = cv2.resize(img, calib_size, interpolation=cv2.INTER_AREA)

        name = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'(\d+(?:\.\d+)?)cm', name)
        tape_cm = float(match.group(1)) if match else None
        frames.append((name, img, tape_cm))

    return frames


def marker_corners_3d():
    ''' marker corners in the marker frame, in the order the detector returns them:
        top left, top right, bottom right, bottom left. origin at the center,
        x right, y up, z out of the marker toward the camera '''
    h = MARKER_SIZE_M / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


def marker_pose(img, detector, K, dist):
    ''' 4x4 transform that takes marker coordinates into camera coordinates,
        plus the reprojection error so a bad detection is easy to spot '''
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or MARKER_ID not in ids.flatten():
        return None, None, None

    idx = list(ids.flatten()).index(MARKER_ID)
    image_pts = corners[idx].reshape(4, 2)
    object_pts = marker_corners_3d()

    # IPPE_SQUARE is the solver made for a single square marker
    ok, rvec, tvec = cv2.solvePnP(object_pts, image_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None, None

    projected, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    error = np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_pts) ** 2, axis=1)))

    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.ravel()
    return T, (corners[idx], ids[idx:idx + 1], rvec, tvec), error


def rotation_deg(R):
    ''' angle of a rotation matrix, in degrees '''
    return np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))


def annotate(img, detection, K, dist, lines):
    ''' marker outline and id, its x/y/z axes, and the measurements in the corner '''
    out = img.copy()
    corners, ids, rvec, tvec = detection

    cv2.aruco.drawDetectedMarkers(out, [corners], ids)
    cv2.drawFrameAxes(out, K, dist, rvec, tvec, MARKER_SIZE_M / 2, 4)

    # dark box behind the text so it reads on any background
    cv2.rectangle(out, (0, 0), (out.shape[1], 30 + 32 * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (12, 38 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def top_down_plot(results):
    ''' bird's eye view (x right, z forward) of the camera path in the first camera's frame,
        with the marker drawn where the first camera saw it '''
    size, margin = 800, 80
    plot = np.full((size, size, 3), 255, np.uint8)

    marker = results[0]['T_cm'][:3, 3]
    xs = [r['motion'][0] for r in results] + [marker[0]]
    zs = [r['motion'][2] for r in results] + [marker[2]]

    # same scale on both axes so distances on the plot are honest
    span = max(max(xs) - min(xs), max(zs) - min(zs), 0.1)
    px_per_m = (size - 2 * margin) / span
    cx, cz = (max(xs) + min(xs)) / 2, (max(zs) + min(zs)) / 2

    def to_px(x, z):
        # z forward points up the plot
        return int(size / 2 + (x - cx) * px_per_m), int(size / 2 - (z - cz) * px_per_m)

    # 10 cm grid
    step = 0.10
    for g in np.arange(np.floor((cx - span) / step) * step, cx + span, step):
        cv2.line(plot, to_px(g, cz - span), to_px(g, cz + span), (230, 230, 230), 1)
    for g in np.arange(np.floor((cz - span) / step) * step, cz + span, step):
        cv2.line(plot, to_px(cx - span, g), to_px(cx + span, g), (230, 230, 230), 1)

    # marker as a short line along its own x axis, as seen from the first camera
    R0 = results[0]['T_cm'][:3, :3]
    half = R0[:, 0] * MARKER_SIZE_M / 2
    cv2.line(plot, to_px(marker[0] - half[0], marker[2] - half[2]),
             to_px(marker[0] + half[0], marker[2] + half[2]), (0, 0, 0), 4)
    cv2.putText(plot, f'marker {MARKER_ID}', to_px(marker[0], marker[2] - 0.04),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    points = [to_px(r['motion'][0], r['motion'][2]) for r in results]
    for a, b in zip(points, points[1:]):
        cv2.line(plot, a, b, (200, 120, 0), 2)
    for i, p in enumerate(points):
        cv2.circle(plot, p, 6, (0, 0, 255) if i == 0 else (200, 120, 0), -1)
        cv2.putText(plot, str(i), (p[0] + 8, p[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(plot, 'camera path, top down (x right, z forward), grid 10 cm', (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return plot


def main():
    K, dist, calib_size = calibrate()
    frames = load_frames(calib_size)
    print(f'****** {len(frames)} frames from {FOLDER}, marker {MARKER_ID}, side {MARKER_SIZE_M * 100:.1f} cm ******\n')

    # sub pixel corners - the pose is only as good as these 4 points
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)

    results = []
    for name, img, tape_cm in frames:
        T_cm, detection, error = marker_pose(img, detector, K, dist)
        if T_cm is None:
            print(f'{name}: marker {MARKER_ID} not found, skipped')
            continue
        results.append({'name': name, 'img': img, 'tape_cm': tape_cm,
                        'T_cm': T_cm, 'detection': detection, 'error': error})

    if len(results) < 2:
        raise ValueError('*ERROR - the marker was found in fewer than 2 frames')

    # the marker is fixed, so it links every frame back to the first one:
    #   camera_i -> marker -> camera_0   gives   T_0i = T_0m @ inv(T_im)
    # its translation is where camera i sits, in the first camera's x right / y down / z forward
    T_0m = results[0]['T_cm']
    tape_0 = results[0]['tape_cm']
    for r in results:
        T_0i = T_0m @ np.linalg.inv(r['T_cm'])
        r['motion'] = T_0i[:3, 3]
        r['turn_deg'] = rotation_deg(T_0i[:3, :3])

    header = f'{"frame":<14}{"dx cm":>8}{"dy cm":>8}{"dz cm":>8}{"|d| cm":>9}{"tape cm":>9}{"diff cm":>9}{"turn deg":>9}{"range cm":>10}{"reproj px":>11}'
    print(header)
    print('-' * len(header))

    for i, r in enumerate(results):
        d = r['motion'] * 100
        moved = np.linalg.norm(d)
        distance = np.linalg.norm(r['T_cm'][:3, 3]) * 100

        expected = None
        if r['tape_cm'] is not None and tape_0 is not None:
            expected = abs(r['tape_cm'] - tape_0)

        tape_txt = f'{expected:9.1f}{moved - expected:+9.1f}' if expected is not None else f'{"-":>9}{"-":>9}'
        print(f'{r["name"]:<14}{d[0]:8.1f}{d[1]:8.1f}{d[2]:8.1f}{moved:9.1f}{tape_txt}'
              f'{r["turn_deg"]:9.1f}{distance:10.1f}{r["error"]:11.2f}')

        lines = [f'frame {i}: {r["name"]}',
                 f'camera move from frame 0: x {d[0]:+.1f}  y {d[1]:+.1f}  z {d[2]:+.1f} cm',
                 f'total {moved:.1f} cm' + (f'  (tape {expected:.1f} cm, diff {moved - expected:+.1f})' if expected is not None else ''),
                 f'rotation {r["turn_deg"]:.1f} deg   marker range {distance:.1f} cm']
        show(f'frame_{i:02d}_{r["name"]}', annotate(r['img'], r['detection'], K, dist, lines))

    show('trajectory_top_down', top_down_plot(results))
    print('**************************')


if __name__ == '__main__':
    main()
