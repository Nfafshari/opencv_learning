import cv2
import numpy as np

from utils.safe_imread import safe_imread
from lessons.base import Lesson, Preview

IMAGE_NAME = 'pattern_chessboard.png'


def source() -> cv2.typing.MatLike:
    ''' The chessboard at half size - the full 1830x1330 is more pixels than we need. '''
    return cv2.resize(safe_imread(IMAGE_NAME), (0, 0), fx=0.5, fy=0.5)


def find_corners(img: cv2.typing.MatLike) -> np.ndarray:
    ''' Shi-Tomasi corner detection, cast to integers so the points can index pixels. '''

    # Convert to grayscale, this simplifies the algorithm detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parametes:
    # * image source,
    # * number of good corners,
    # * minimum corner quality,
    # * minimum euclidean distance
    corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.5, 20)

    return corners.astype(np.intp)  # cast values to integers


def draw_corners(img: cv2.typing.MatLike, corners: np.ndarray, radius: int = 4) -> cv2.typing.MatLike:
    '''
    Mark every detected corner with a filled blue dot.

    `radius` is a parameter only so the TUI can ask for fatter dots. A 4px dot on a
    915px wide image is well under one character cell once it is scaled down to text.
    '''

    for corner in corners:
        x, y = corner.ravel()  # flatten the array (ie [[[0, 1, 2], ...]] = [0, 1, 2, ...])

        # draw the corner
        cv2.circle(img, (x, y), radius, (255, 0, 0), -1)

    return img


def draw_connections(img: cv2.typing.MatLike, corners: np.ndarray) -> cv2.typing.MatLike:
    ''' Draw a randomly colored line between every pair of corners. '''

    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            corner1 = tuple(corners[i][0])
            corner2 = tuple(corners[j][0])

            # map each value as an integer and make it a tuple
            color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
            cv2.line(img, corner1, corner2, color, 1)

    return img


def corners_only(radius: int = 4) -> cv2.typing.MatLike:
    img = source()
    return draw_corners(img, find_corners(img), radius)


def corners_and_lines() -> cv2.typing.MatLike:
    img = source()
    corners = find_corners(img)

    draw_corners(img, corners)

    # Cropping white space of the image used out.
    # NOTE: this is a VIEW into img, not a copy, so the lines drawn below still
    # show up here even though they are drawn after the crop.
    cropped_image = img[0:600, 0:800]

    # Draw random lines between each corner
    draw_connections(img, corners)

    return cropped_image


def run() -> None:
    '''
    Find corners with Shi-Tomasi, then draw them and connect them.

    cv2.goodFeaturesToTrack(), cv2.circle(), cv2.line()
    '''

    cv2.imshow('image', corners_and_lines())

    cv2.waitKey(0)
    cv2.destroyAllWindows()


CONCEPTS = '''\
# Corner detection

## Why corners

A flat patch looks the same no matter which way you nudge it. An edge looks the same
if you slide *along* it. A **corner** changes under a nudge in every direction, which
makes it the one feature you can re-find reliably in another frame - the basis for
tracking, stitching, and camera calibration.

## goodFeaturesToTrack

```python
corners = cv2.goodFeaturesToTrack(gray, 100, 0.5, 20)
```

| Argument | Meaning |
| --- | --- |
| `gray` | **must** be single channel - convert first |
| `100` | keep at most the 100 strongest corners |
| `0.5` | quality level: discard anything below 50% of the best corner's score |
| `20` | minimum pixel distance between two kept corners |

Quality level is relative, not absolute. On a low-contrast image the "best" corner is
weak, so `0.5` of it is weaker still and you get junk. The distance argument is what
stops 30 detections from piling onto one corner.

## Shape of the result

`goodFeaturesToTrack` returns `(N, 1, 2)` float32 - an extra middle axis you almost
never want. `corner.ravel()` flattens one entry to `x, y`, and `.astype(np.intp)`
makes them usable as pixel coordinates. Drawing functions reject floats.

## Drawing

- `cv2.circle(img, (x, y), radius, color, -1)` - thickness `-1` means **filled**.
- Colors are `(B, G, R)`, so `(255, 0, 0)` is blue, not red.
- These functions draw **in place** and return nothing useful.

## The view-vs-copy trap in this lesson

`cropped_image = img[0:600, 0:800]` is a slice, so it is a *view* onto the same
buffer. The connecting lines are drawn onto `img` **after** that line runs - and they
still appear in the crop. Add `.copy()` and they would not.
'''

LESSON = Lesson(
    title="Lesson 05 - Corner Detection",
    summary="Shi-Tomasi corners, drawing circles and lines",
    run=run,
    concepts=CONCEPTS,
    previews=(
        Preview(
            label='Source',
            build=source,
            caption='Chessboard pattern resized to half scale.',
        ),
        Preview(
            label='Corners',
            build=lambda: corners_only(radius=14),
            caption='Up to 100 corners, quality 0.5, min 20px apart. Dots enlarged to survive the downscale.',
        ),
        Preview(
            label='Connected (cropped)',
            build=corners_and_lines,
            caption='Every pair joined by a random color line, then cropped to 800x600.',
        ),
    ),
)

if __name__ == "__main__":
    run()
