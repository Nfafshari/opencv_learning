import cv2
import numpy as np

from utils.safe_imread import safe_imread
from lessons.base import Lesson, Preview

# find the color we want to extract. We need 2 values, the lower bound of the color
# and the upper bound of the color
LOWER_RED = np.array([0, 100, 70])
UPPER_RED = np.array([10, 255, 255])

PHOTO = 'coins1.jpg'


def red_mask(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    ''' The raw mask: white where the pixel is in range, black everywhere else. '''

    # convert our color scheme to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # create a mask
    return cv2.inRange(hsv, LOWER_RED, UPPER_RED)


def red_only(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    ''' The original frame with everything outside the mask blacked out. '''

    # apply the mask, pass same image twice and the mask
    return cv2.bitwise_and(frame, frame, mask=red_mask(frame))

    # BITWISE AND
    # 1 AND 1 = 1
    # 0 AND 1 = 0
    # 1 AND 0 = 0
    # 0 AND 0 = 0
    # SO
    # if red pixel = pixel is found it will display the red pixel only


def hue_chart(width: int = 480, height: int = 220) -> cv2.typing.MatLike:
    '''
    A synthetic test card: hue sweeps left to right, brightness fades top to bottom.

    Built directly in HSV so the mask bounds are easy to reason about - the lower
    bound of [0, 100, 70] should clip both the far right (wrong hue) and the
    bottom strip (too dark).
    '''

    hsv = np.zeros((height, width, 3), np.uint8)

    hsv[..., 0] = np.linspace(0, 179, width, dtype=np.uint8)          # hue across x
    hsv[..., 1] = 255                                                 # full saturation
    hsv[..., 2] = np.linspace(255, 0, height, dtype=np.uint8)[:, None]  # value down y

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def run() -> None:
    '''
    Isolate one color from a live camera feed using an HSV range mask.

    cv2.cvtColor(), cv2.inRange(), cv2.bitwise_and()
    '''

    # Capture camera video using cv2
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError('*ERROR - Could not open camera 0. Is another app using the webcam?')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('frame', red_only(frame))

        if cv2.waitKey(1) == ord('q'):
            break

    # release the video capture so other applications can use it
    cap.release()
    cv2.destroyAllWindows()


CONCEPTS = '''\
# Color detection with HSV

## Why convert away from BGR

In BGR, "red" is spread across all three channels and every one of them moves when
the lighting changes. **HSV** splits it apart:

- **H**ue - which color it is (this is the part you actually want to threshold)
- **S**aturation - how vivid, 0 = gray
- **V**alue - how bright, 0 = black

Thresholding hue alone survives shadows and dim rooms far better than an RGB box.

## OpenCV's ranges are not the textbook ranges

| Channel | Textbook | OpenCV (8-bit) |
| --- | --- | --- |
| Hue | 0-360 | **0-179** (halved to fit a byte) |
| Saturation | 0-100% | 0-255 |
| Value | 0-100% | 0-255 |

## The mask

```python
mask = cv2.inRange(hsv, lower, upper)          # 8-bit, 255 = in range, 0 = out
result = cv2.bitwise_and(frame, frame, mask=mask)
```

`inRange` returns a single-channel image, so it shows up as black and white.
`bitwise_and` with the frame passed twice is the idiom for "keep the original
pixels, but only where the mask is set".

## The red wrap-around problem

Red sits at **both ends** of the hue circle - roughly 0-10 *and* 170-179. The bounds
here only catch the low half, so a deep red can go undetected. Catching all of it
means two `inRange` calls combined with `cv2.bitwise_or`.

Raising the lower S and V bounds (`[0, 100, 70]`) is what excludes washed-out grays
and near-black pixels, which technically have a red hue but do not look red at all.
'''

LESSON = Lesson(
    title="Lesson 04 - Color Detection",
    summary="HSV conversion, inRange masks, and bitwise_and",
    run=run,
    concepts=CONCEPTS,
    previews=(
        Preview(
            label='Hue test card',
            build=hue_chart,
            caption='Built in HSV: hue 0-179 across x, value 255-0 down y.',
        ),
        Preview(
            label='Mask (test card)',
            build=lambda: red_mask(hue_chart()),
            caption='inRange() output - white is in range. Note the dark bottom is rejected on V, not hue.',
        ),
        Preview(
            label='Masked test card',
            build=lambda: red_only(hue_chart()),
            caption='bitwise_and keeps the original pixels only where the mask is set.',
        ),
        Preview(
            label='Real photo',
            build=lambda: red_only(safe_imread(PHOTO)),
            caption='The same bounds on coins - copper survives, silver drops out. Press r for live video.',
        ),
    ),
)

if __name__ == "__main__":
    run()
