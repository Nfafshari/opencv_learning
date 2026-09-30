import cv2
import numpy as np
import math
import time

from feature_detection import fast, anms, NUM_FEATURES, FIGURE_WIDTH

# the patch every descriptor is handed. FAST has no scale of its own, so one fixed
# size is assigned to every keypoint and all three descriptors get the same deal
PATCH_SIZE = 31

# Lowe's ratio test - a match is only kept if the best candidate is clearly better
# than the runner up. 0.75 is the value from the SIFT paper
MATCH_RATIO = 0.75

# how close a match has to land to where the homography says it should, to count
# as correct. 3 px is the usual tolerance in the matching literature
CORRECT_PX = 3.0


def orientation(gray, points, patch=PATCH_SIZE):
    ''' intensity centroid angle per keypoint as used by ORB '''
    half = patch // 2

    # pad so a patch centred on a border keypoint still fits
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REFLECT_101)
    padded = padded.astype(np.float32)

    # circular patch, so the moment does not depend on which way the square is turned
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    mask = (xs ** 2 + ys ** 2) <= half ** 2
    xs, ys = xs[mask], ys[mask]

    angles = np.empty(len(points), np.float32)

    for i, (x, y) in enumerate(points[:, :2].astype(int)):
        block = padded[y:y + patch, x:x + patch][mask]

        # the vector from the corner to the centre of mass of its patch
        angles[i] = math.degrees(math.atan2(float((block * ys).sum()),
                                            float((block * xs).sum()))) % 360

    return angles


def detect(gray):
    '''
    the one detection scheme held fixed for all three descriptors FAST then ANMS

    returns cv2.KeyPoint objects carrying a class_id, which is what lets the three
    descriptors be lined up again afterwards
    '''
    points = anms(fast(gray), NUM_FEATURES)
    angles = orientation(gray, points)

    keypoints = []
    for i, ((x, y, response), angle) in enumerate(zip(points, angles)):
        kp = cv2.KeyPoint(float(x), float(y), float(PATCH_SIZE))
        kp.angle = float(angle)
        kp.response = float(response)
        kp.class_id = i
        keypoints.append(kp)

    return keypoints


def describe(gray, keypoints, extractor):
    ''' run one descriptor over the keypoints, and time just that call '''
    # compute() edits the list it is given, so hand it a copy each time
    copies = []
    for kp in keypoints:
        clone = cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size)
        clone.angle, clone.response, clone.class_id = kp.angle, kp.response, kp.class_id
        copies.append(clone)

    start = time.perf_counter()
    kept, descriptors = extractor.compute(gray, copies)
    elapsed = (time.perf_counter() - start) * 1000

    if descriptors is None:
        return [], np.empty((0, 0), np.uint8), elapsed

    return kept, descriptors, elapsed


def align(described):
    ''' cut every descriptor down to the keypoints all of them managed to describe '''
    shared = None
    for keypoints, _descriptors, _ms in described.values():
        ids = {kp.class_id for kp in keypoints}
        shared = ids if shared is None else (shared & ids)

    aligned = {}
    for name, (keypoints, descriptors, ms) in described.items():
        order = [i for i, kp in enumerate(keypoints) if kp.class_id in shared]
        order.sort(key=lambda i: keypoints[i].class_id)

        aligned[name] = ([keypoints[i] for i in order], descriptors[order], ms)

    return aligned


def candidates(query, train, norm):
    ''' every keypoint's best match in image two, with the ratio that scores it '''
    matcher = cv2.BFMatcher_create(norm, crossCheck=False)

    start = time.perf_counter()
    pairs = matcher.knnMatch(query, train, k=2)
    elapsed = (time.perf_counter() - start) * 1000

    rows = [(p[0].queryIdx, p[0].trainIdx,
             p[0].distance / p[1].distance if p[1].distance > 0 else 0.0)
            for p in pairs if len(p) == 2]

    if not rows:
        return np.empty((0, 2), int), np.empty(0), elapsed

    index = np.array([[i, j] for i, j, _ in rows], int)
    ratios = np.array([r for _, _, r in rows], np.float64)

    return index, ratios, elapsed


def accepted(index, ratios, threshold=MATCH_RATIO):
    ''' the candidates the ratio test lets through at one threshold '''
    return [tuple(pair) for pair in index[ratios < threshold]]


def roc(gaps, index, ratios):
    ''' sweep the ratio threshold and trace true positive rate against false positive rate '''
    if len(index) == 0:
        return np.zeros(1), np.zeros(1), 0.0

    correct = gaps[index[:, 0], index[:, 1]] < CORRECT_PX

    # most confident first, so each step down the list lowers the bar a little
    order = np.argsort(ratios)
    correct = correct[order]

    true_positives = np.cumsum(correct)
    false_positives = np.cumsum(~correct)

    # both rates start at the origin, where the threshold accepts nothing at all
    tpr = np.concatenate([[0.0], true_positives / true_positives[-1]
                          if true_positives[-1] else np.zeros(len(correct))])
    fpr = np.concatenate([[0.0], false_positives / false_positives[-1]
                          if false_positives[-1] else np.zeros(len(correct))])

    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def project(keypoints, homography):
    ''' push image one keypoints through the known homography '''
    xy = np.array([kp.pt for kp in keypoints], np.float64).reshape(-1, 1, 2)

    if len(xy) == 0:
        return np.empty((0, 2))

    return cv2.perspectiveTransform(xy, homography).reshape(-1, 2)


def geometry(query_kps, train_kps, homography, shape):
    ''' where the homography says every image one keypoint should have landed '''
    height, width = shape

    projected = project(query_kps, homography)
    train_xy = np.array([kp.pt for kp in train_kps], np.float64).reshape(-1, 2)

    inside = ((projected[:, 0] >= 0) & (projected[:, 0] < width)
              & (projected[:, 1] >= 0) & (projected[:, 1] < height))

    # distance from each projected point to every detected point in image two
    gaps = np.linalg.norm(projected[:, None, :] - train_xy[None, :, :], axis=2)

    # a correspondence only exists if the detector independently found that corner too
    findable = inside & (gaps.min(axis=1) < CORRECT_PX)

    return gaps, findable


def confusion_matrix(gaps, findable, matches):
    ''' the 2x2 table, scored against the homography the second image was warped by '''

    predicted = np.zeros(len(findable), bool)
    landed_right = np.zeros(len(findable), bool)

    for i, j in matches:
        predicted[i] = True
        landed_right[i] = gaps[i, j] < CORRECT_PX

    true_positive = int((predicted & landed_right).sum())
    false_positive = int((predicted & ~landed_right).sum())
    false_negative = int((~predicted & findable).sum())
    true_negative = int((~predicted & ~findable).sum())

    return true_positive, false_positive, false_negative, true_negative


def scores(true_positive, false_positive, false_negative, true_negative):
    ''' precision, recall and f1 off the confusion counts, guarding the empty cases '''
    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    total = predicted + false_negative + true_negative

    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / total if total else 0.0

    return precision, recall, f1, accuracy


def transforms(shape):
    ''' the known homographies each image gets warped by, hardest last '''
    height, width = shape
    centre = (width / 2, height / 2)

    def turn(degrees, scale=1.0):
        return np.vstack([cv2.getRotationMatrix2D(centre, degrees, scale), [0, 0, 1]])

    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    skewed = np.float32([[width * .08, height * .04], [width * .94, 0],
                         [width, height * .97], [width * .03, height * .92]])

    return (
        ('rotation 15', turn(15)),
        ('rotation 45', turn(45)),
        ('scale 0.7', turn(0, 0.7)),
        ('perspective', cv2.getPerspectiveTransform(corners, skewed)),
    )


DESCRIPTORS = (
    ('SIFT', lambda: cv2.SIFT_create(), cv2.NORM_L2),
    ('BRIEF', lambda: cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32), cv2.NORM_HAMMING),
    ('ORB', lambda: cv2.ORB_create(), cv2.NORM_HAMMING),
)

# one hue per descriptor, fixed - never reassigned when the order or count changes.
# stepped for a dark surface and checked for colourblind separation as a set
SERIES_BGR = {
    'SIFT': (229, 135, 57),     # blue    #3987e5
    'BRIEF': (38, 89, 217),     # orange  #d95926
    'ORB': (112, 158, 25),      # aqua    #199e70
}

SURFACE_BGR = (25, 26, 26)      # #1a1a19
INK_BGR = (255, 255, 255)       # #ffffff
MUTED_BGR = (183, 194, 195)     # #c3c2b7
GRID_BGR = (58, 58, 56)


def vertical_text(figure, text, x, y_centre):
    ''' cv2 can only draw text level, so draw it on a strip and turn the strip '''
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)

    strip = np.full((height + 8, width + 8, 3), SURFACE_BGR, np.uint8)
    cv2.putText(strip, text, (4, height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                MUTED_BGR, 1, cv2.LINE_AA)
    strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)

    top = int(y_centre - strip.shape[0] / 2)
    figure[top:top + strip.shape[0], x:x + strip.shape[1]] = strip


def roc_figure(curves, label, size=620, pad=84):
    ''' the ROC curves for all three descriptors under one transform, on one axis '''
    span = size - 2 * pad
    figure = np.full((size, size + 150, 3), SURFACE_BGR, np.uint8)

    def place(x, y):
        # rates run 0..1, y flips because image rows count downwards
        return int(pad + x * span), int(pad + (1 - y) * span)

    # recessive grid, drawn first so the data sits on top of it
    for step in range(0, 11, 2):
        fraction = step / 10
        cv2.line(figure, place(fraction, 0), place(fraction, 1), GRID_BGR, 1, cv2.LINE_AA)
        cv2.line(figure, place(0, fraction), place(1, fraction), GRID_BGR, 1, cv2.LINE_AA)

        cv2.putText(figure, f'{fraction:.1f}', (place(fraction, 0)[0] - 10, size - pad + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED_BGR, 1, cv2.LINE_AA)
        cv2.putText(figure, f'{fraction:.1f}', (pad - 32, place(0, fraction)[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED_BGR, 1, cv2.LINE_AA)

    # chance line - a descriptor that lands on this has learned nothing
    for step in range(0, span, 12):
        start = place(step / span, step / span)
        end = place(min((step + 6) / span, 1), min((step + 6) / span, 1))
        cv2.line(figure, start, end, GRID_BGR, 1, cv2.LINE_AA)

    for name, (false_rate, true_rate, area) in curves.items():
        points = np.array([place(x, y) for x, y in zip(false_rate, true_rate)], np.int32)
        cv2.polylines(figure, [points], False, SERIES_BGR[name], 2, cv2.LINE_AA)

    # legend carries the identity, so the curves are never colour alone
    for row, (name, (_fpr, _tpr, area)) in enumerate(curves.items()):
        y = pad + 16 + row * 24
        cv2.circle(figure, (size + 18, y - 4), 5, SERIES_BGR[name], -1, cv2.LINE_AA)
        cv2.putText(figure, f'{name}  {area:.3f}', (size + 32, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK_BGR, 1, cv2.LINE_AA)

    cv2.putText(figure, f'ROC - {label}', (pad, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK_BGR, 1, cv2.LINE_AA)
    cv2.putText(figure, 'ratio-test threshold swept 0 to 1, area under curve at right',
                (pad, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED_BGR, 1, cv2.LINE_AA)

    # axis titles, the vertical one turned so it reads up the left edge
    (text_width, _), _ = cv2.getTextSize('false positive rate',
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(figure, 'false positive rate',
                (pad + (span - text_width) // 2, size - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED_BGR, 1, cv2.LINE_AA)
    vertical_text(figure, 'true positive rate', 8, pad + span / 2)

    return figure


def match_figure(left, right, query_kps, train_kps, matches, homography):
    ''' the two frames side by side, correct matches in green and wrong ones in red '''
    figure = np.hstack([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)])
    offset = left.shape[1]

    projected = project(query_kps, homography)

    for i, j in matches:
        start = tuple(np.int32(query_kps[i].pt))
        end = tuple(np.int32(train_kps[j].pt) + [offset, 0])

        correct = np.linalg.norm(projected[i] - train_kps[j].pt) < CORRECT_PX
        cv2.line(figure, start, end, (0, 200, 0) if correct else (0, 0, 235), 1, cv2.LINE_AA)

    scale = FIGURE_WIDTH / figure.shape[1]

    if scale < 1:
        figure = cv2.resize(figure, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return figure


def main():
    path = "images/minecraft_1.jpeg"

    gray = cv2.imread(path, 0)

    if gray is None:
        raise ValueError(f'*ERROR - Could not read {path}')

    height, width = gray.shape

    # the detection scheme is fixed, so image one is detected and described once
    query_kps = detect(gray)

    print(f"****** {path}  ({width}x{height}) ******")
    print(f"detector held fixed: FAST + ANMS, {len(query_kps)} keypoints, "
          f"{PATCH_SIZE}px patches, shared orientations")
    print()

    for label, homography in transforms(gray.shape):
        warped = cv2.warpPerspective(gray, homography, (width, height))
        train_kps = detect(warped)

        described_query = {}
        described_train = {}
        for name, build, _norm in DESCRIPTORS:
            described_query[name] = describe(gray, query_kps, build())
            described_train[name] = describe(warped, train_kps, build())

        # all three descriptors get the identical keypoint set on both sides
        described_query = align(described_query)
        described_train = align(described_train)

        print(f"-- {label} --  {len(next(iter(described_query.values()))[0])} keypoints "
              f"described on both sides")
        print(f"  {'descriptor':<12}{'bytes':>6}{'describe':>11}{'match':>9}"
              f"{'TP':>7}{'FP':>6}{'FN':>6}{'TN':>6}"
              f"{'prec':>8}{'recall':>8}{'F1':>7}{'AUC':>8}")

        # align() leaves every descriptor on the same keypoints, so the geometry is
        # shared and the distance table only gets built once per transform
        shared_kps1 = next(iter(described_query.values()))[0]
        shared_kps2 = next(iter(described_train.values()))[0]
        gaps, findable = geometry(shared_kps1, shared_kps2, homography, gray.shape)

        figures = {}
        curves = {}
        for name, _build, norm in DESCRIPTORS:
            kps1, desc1, describe_ms = described_query[name]
            kps2, desc2, _ = described_train[name]

            index, ratios, match_ms = candidates(desc1, desc2, norm)
            matches = accepted(index, ratios)

            counts = confusion_matrix(gaps, findable, matches)
            precision, recall, f1, _accuracy = scores(*counts)

            curves[name] = roc(gaps, index, ratios)

            # SIFT is 128 float32s, the binary pair are 32 uint8s - report real bytes,
            # because 128 next to 32 reads like SIFT is only four times the size
            size = desc1.shape[1] * desc1.itemsize

            print(f"  {name:<12}{size:>6}{describe_ms:>9.1f}ms{match_ms:>7.1f}ms"
                  f"{counts[0]:>7}{counts[1]:>6}{counts[2]:>6}{counts[3]:>6}"
                  f"{precision:>8.2f}{recall:>8.2f}{f1:>7.2f}{curves[name][2]:>8.3f}")

            figures[name] = match_figure(gray, warped, kps1, kps2, matches, homography)

        print()

        cv2.imshow(f'ROC - {label}', roc_figure(curves, label))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for name, figure in figures.items():
            cv2.imshow(f'{name} - {label} - green correct, red wrong', figure)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("**************************")


if __name__ == "__main__":
    main()
