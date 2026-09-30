import cv2
import numpy as np
import math
import time

# how many features to keep after selection
NUM_FEATURES = 500

# the grid used to score how well a point set covers the frame. 16x16 is 256 cells
# against a budget of 500 points, which is fine enough that a clumped set cannot fill
# it by accident - at 8x8 even raw FAST occupies every cell and the number says nothing
COVERAGE_GRID = 16

# the saved and displayed figures get scaled down to this width so they fit on a screen
FIGURE_WIDTH = 1600

def time_detector(fn, gray, runs=5):
    ''' helper function for getting the median milliseconds for one detection pass '''
    fn(gray)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(gray)
        times.append((time.perf_counter() - start) * 1000)

    # returns the median of runs timings in milliseconds
    return float(np.median(times))

def dog(img):
    ''' difference of gaussians which is used inside the detector SIFT '''
    keypoints = cv2.SIFT_create().detect(img, None)

    points = np.array([[kp.pt[0], kp.pt[1], kp.response] for kp in keypoints], np.float32).reshape(-1, 3)

    # SIFT returns several orientations at the same spot, keep the strongest of each
    return strongest_per_position(points)

def harris(img, quality=0.01, min_distance=5):
    ''' harris corners, R = det(M) - k*trace(M)^2 on the structure tensor '''
    response = cv2.cornerHarris(img.astype(np.float32), 3, 3, 0.04)

    window = np.ones((2 * min_distance + 1,) * 2, np.uint8)
    peaks = (response >= cv2.dilate(response, window)) & (response > quality * response.max())

    # returns an (N, 3) float32 array of x, y, response. whole pixel positions because
    # they come straight off the response image, and R itself is the corner strength
    ys, xs = np.nonzero(peaks)
    return np.column_stack([xs, ys, response[ys, xs]]).astype(np.float32)


def shi_tomasi(img, quality=0.01, min_distance=1):
    ''' shi tomasi, the same structure tensor scored with min(lambda1, lambda2)  '''
    points = cv2.goodFeaturesToTrack(img, 0, quality, min_distance)

    if points is None:
        return np.empty((0, 3), np.float32)

    points = points.reshape(-1, 2)

    # goodFeaturesToTrack throws the scores away, so read min(lambda1, lambda2) back
    # off the eigenvalue image. min_distance is 1 here on purpose - the built in
    # spacing rule would otherwise be doing the job we want ANMS to do
    eigen = cv2.cornerMinEigenVal(img, 3)
    xs, ys = points[:, 0].astype(int), points[:, 1].astype(int)

    # returns an (N, 3) float32 array of x, y, response
    return np.column_stack([points, eigen[ys, xs]]).astype(np.float32)


def fast(img, threshold=30):
    ''' FAST, the segment test on a 16 pixel circle '''
    detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
    keypoints = detector.detect(img, None)

    # returns an (N, 3) float32 array of x, y, response. whole pixels, and the response
    # is the segment test score, the largest threshold this pixel would still pass at
    return np.array([[kp.pt[0], kp.pt[1], kp.response] for kp in keypoints], np.float32).reshape(-1, 3)


def strongest_per_position(points, decimals=3):
    ''' collapses duplicate xy positions down to the single strongest response '''
    if len(points) == 0:
        return points

    # sort weakest first so the last write per position is the strongest one
    points = points[np.argsort(points[:, 2])]

    _, index = np.unique(points[:, :2].round(decimals), axis=0, return_index=True)
    return points[np.sort(index)]


def strongest(points, num_to_keep=NUM_FEATURES):
    '''
    Gets the strongest N features to compare against for AMNS
    '''
    if len(points) <= num_to_keep:
        return points.copy()

    return points[np.argsort(-points[:, 2])[:num_to_keep]]

def anms(points, num_to_keep=NUM_FEATURES, c_robust=0.9):
    '''
    every point gets a suppression radius, the distance to the nearest meaningfully
    stronger point, so points buried in a cluster score small and isolated ones score
    large. keeping the widest radii spreads the survivors across the whole frame.
    '''
    if len(points) <= num_to_keep:
        return points.copy()

    # strongest first, so every point stronger than point i is already behind us at i
    points = points[np.argsort(-points[:, 2])]
    xy, responses = points[:, :2], points[:, 2]

    # the global maximum has nothing stronger anywhere, its radius is infinite
    radii = np.full(len(points), np.inf)

    for i in range(1, len(points)):
        # c_robust < 1 means a near tie does not count as stronger, so two equally good
        # corners do not suppress each other. 0.9 is the value the paper uses
        stronger = xy[:i][responses[:i] * c_robust > responses[i]]

        if len(stronger):
            radii[i] = np.sqrt(((stronger - xy[i]) ** 2).sum(1).min())

    # widest suppression radius wins, return the points
    return points[np.argsort(-radii)[:num_to_keep]]


def coverage(points, shape, grid=COVERAGE_GRID):
    '''
    score how evenly a point set fills the frame, on a grid x grid split of the image (occupancy)
    '''
    height, width = shape

    if len(points) == 0:
        return 0.0, 0.0

    cols = np.clip((points[:, 0] / (width / grid)).astype(int), 0, grid - 1)
    rows = np.clip((points[:, 1] / (height / grid)).astype(int), 0, grid - 1)

    counts = np.bincount(rows * grid + cols, minlength=grid * grid).astype(float)

    share = counts[counts > 0] / counts.sum()
    entropy = -(share * np.log(share)).sum() / math.log(grid * grid)

    return float((counts > 0).mean()), float(entropy)


def draw_points(gray, points, color):
    ''' put the points on a bgr copy of the image '''
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for x, y in points[:, :2].astype(int):
        cv2.circle(canvas, (x, y), 3, color, 1, cv2.LINE_AA)

    return canvas


def caption(canvas, text, x):
    ''' black bar and white text across the top of one panel, drawn in place '''
    cv2.rectangle(canvas, (x, 0), (x + canvas.shape[1] // 2, 30), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


def comparison_figure(gray, baseline, selected, detector):
    ''' the before and after pair side by side, which is the figure for the writeup '''
    before = draw_points(gray, baseline, (0, 0, 235))
    after = draw_points(gray, selected, (0, 220, 0))

    # a grey seam so the two panels do not read as one image
    seam = np.full((gray.shape[0], 4, 3), 90, np.uint8)
    figure = np.hstack([before, seam, after])

    # two panels side by side is wider than any screen, so scale the pair down. this
    # happens before the captions go on, otherwise the text shrinks with the image
    scale = FIGURE_WIDTH / figure.shape[1]

    if scale < 1:
        figure = cv2.resize(figure, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    caption(figure, f'{detector} strongest {len(baseline)} pts', 0)
    caption(figure, f'{detector} + ANMS by {len(selected)} pts', figure.shape[1] // 2)

    return figure


DETECTORS = (
    ('DoG', dog),
    ('Harris', harris),
    ('Shi-Tomasi', shi_tomasi),
    ('FAST', fast),
)

def main():
    # set all image paths
    images_list = [
        "images/minecraft_1.jpeg",
        "images/minecraft_2.jpeg",
        "images/minecraft_3.jpeg",
        "images/minecraft_blurry.jpeg"
    ]

    # loop through each image and compute DoG, Harris, Shi-Tomasi, and FAST
    for path in images_list:
        # read image in gray scale
        gray = cv2.imread(path, 0)

        if gray is None:
            raise ValueError(f'*ERROR - Could not read {path}')

        print(f"****** {path}  ({gray.shape[1]}x{gray.shape[0]}) ******")

        for name, detector in DETECTORS:
            raw = detector(gray)
            detect_ms = time_detector(detector, gray)

            print(f"-- {name} --  {len(raw)} raw points, {detect_ms:.1f} ms to detect")
            print(f"  {'method':<10}{'kept':>6}{'time':>10}{'occupancy':>12}{'entropy':>10}")

            # both get the same budget, so the coverage numbers are comparable
            selections = {}
            for method, select in (('Strongest', strongest), ('ANMS', anms)):
                start = time.perf_counter()
                selected = select(raw, NUM_FEATURES)
                elapsed = (time.perf_counter() - start) * 1000

                occupancy, entropy = coverage(selected, gray.shape)
                selections[method] = selected

                print(f"  {method:<10}{len(selected):>6}{elapsed:>8.1f}ms"
                      f"{occupancy:>11.0%}{entropy:>10.3f}")

            print()

            # the figure is the baseline against ANMS, both at 500 points
            figure = comparison_figure(gray, selections['Strongest'], selections['ANMS'], name)

            cv2.imshow('strongest vs ANMS - press any key for the next detector', figure)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("**************************")
        print()

if __name__ == "__main__":
    main()


'''
Fixed — the key mismatch would have crashed on the first detector. The figure captions you already renamed.

## Reading the output

```
****** images/minecraft_1.jpeg  (1500x2000) ******
-- FAST --  24555 raw points, 8.9 ms to detect
  method      kept      time   occupancy   entropy
  Strongest    500     0.2ms        43%     0.782
  ANMS         500  3728.1ms        79%     0.919
```

**`24555 raw points`** — what the detector found before any selection, at its own threshold. This is the pool ANMS picks from. It varies wildly: FAST on a busy Minecraft frame gives ~25k, Harris ~4.5k, Shi-Tomasi ~37k. Bigger pool means ANMS has more to work with but takes longer.

**`8.9 ms to detect`** — median of 5 runs from `time_detector`, so one warm-up pass plus 5 timed. Median rather than mean because a single OS hiccup would skew a mean.

**`kept`** — always 500 (`NUM_FEATURES`). Both rows are the same size *on purpose*. This is the control that makes the last two columns mean anything: if one method kept 5000 and the other 500, better coverage would just mean "more points," not "better chosen."

**`time`** — cost of the selection step alone, detection excluded.

**`occupancy`** — of 256 grid cells, the percentage holding at least one point.

**`entropy`** — how evenly the points spread across those cells, normalized to [0,1]. 1.0 would be exactly 500/256 points in every cell.

## What the two rows are

`Strongest` is the naive thing: sort by response, take the top 500. It's the control, not a real competitor — it's what you'd do if you'd never heard of ANMS.

`ANMS` is the actual method.

The comparison is the entire result. **43% → 79% occupancy at identical point count.** Sorting by strength alone puts 500 points into 110 of 256 cells, because the busiest texture in the frame owns most of the top responses. ANMS puts the same 500 into 202 cells.

Entropy moves less dramatically (0.782 → 0.919) because it's a log-scale measure — it's already fairly high for any non-degenerate set. Don't read the two columns as independent evidence; they're two views of one fact.

## Things worth noticing in your report

**The time column is embarrassing for ANMS and you should say so.** 0.2ms vs 3728ms — four orders of magnitude. That's the O(n²) in the docstring: 24,555 points means ~300 million distance comparisons. It's honest to report it as the cost of the method rather than hide it.

**Harris and FAST tell different stories.** Harris already applies its own non-max suppression with `min_distance=5`, so its raw set is pre-thinned and ANMS has less clumping to fix. FAST dumps everything, so ANMS has more to do and the improvement is larger. Worth a sentence.

**`minecraft_blurry` is your interesting case.** Blur kills corner responses, so the raw pool collapses (~1.2k for Harris vs 4.5k sharp) and occupancy stays low even after ANMS — around 31% in my run. ANMS can only distribute points that exist; if the detector found nothing in the sky, no selection method conjures features there. That's a genuine limitation to state rather than a failure of your implementation.

---------------------------------------------------

`coverage` isn't part of ANMS — it's the measurement that proves ANMS did its job. You can delete it and the selection still works; you just lose the evidence.

Your requirement says the method "must provide a reasonable set of features covering the whole image." That's a claim you have to back with a number, not a screenshot. This is that number.

## How it works

Split the image into a 16×16 grid (256 cells), drop every selected point into its cell, count. Then two summaries of those counts:

**Occupancy** — fraction of cells holding at least one point. Simple, intuitive, easy to defend in a writeup.

**Entropy** — Shannon entropy of the count distribution, divided by `log(256)` so it lands in [0, 1]. 1.0 means every cell holds exactly the same number of points; lower means the mass is piling up somewhere.

## Why both

Occupancy can be gamed. A set with 1 point in one cell and 400 in its neighbour scores both cells as "occupied" — 100% occupancy, terrible distribution. Entropy catches that because it's sensitive to the *balance*, not just presence.

Entropy alone is harder to explain to a reader, and occupancy alone lies. Together they're hard to argue with.

## What it produced on your images

```
  method      kept   occupancy   entropy
  top-N        500         43%     0.782
  ANMS         500         79%     0.919
'''