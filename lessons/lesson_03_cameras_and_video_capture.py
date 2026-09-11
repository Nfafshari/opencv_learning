import cv2
import numpy as np

from utils.safe_imread import safe_imread
from lessons.base import Lesson, Preview

STILL_IMAGE = 'camera_man_w_noise.jpg'


def mosaic(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Tile a frame into itself four times, with the top two rotated 180 degrees.

    Reads the size off `frame.shape` rather than `cap.get(3)/cap.get(4)` so the
    exact same function works on a webcam frame and on a still image.
    '''

    height, width = frame.shape[:2]

    # create a blank image
    image = np.zeros(frame.shape, np.uint8)

    # Resize the frame so that we can copy and paste it 4 times
    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_h, small_w = smaller_frame.shape[:2]

    upside_down = cv2.rotate(smaller_frame, cv2.ROTATE_180)

    # Slice the smaller frame into our blank image. The destination slices are sized
    # from the *resized* frame rather than from height//2, because an odd dimension
    # rounds down on the resize: a 225px frame halves to 112, so `image[112:]` would
    # be 113 rows and NumPy refuses to broadcast 112 into 113. Anchoring the bottom
    # and right halves to the edges leaves at most a one pixel seam down the middle.
    image[:small_h, :small_w] = upside_down                    # TOP LEFT
    image[height - small_h:, :small_w] = smaller_frame         # BOTTOM LEFT
    image[:small_h, width - small_w:] = upside_down            # TOP RIGHT
    image[height - small_h:, width - small_w:] = smaller_frame  # BOTTOM RIGHT

    return image


def run() -> None:
    '''
    Live webcam capture, and building a new frame out of slices of the old one.

    cv2.VideoCapture(), cap.read(), cv2.waitKey(1)
    '''

    # Capture camera video using cv2
    cap = cv2.VideoCapture(0)

    # OR use an mp4 video
    # cap = cv2.VideoCapture('my_video.mp4')

    if not cap.isOpened():
        raise RuntimeError('*ERROR - Could not open camera 0. Is another app using the webcam?')

    while True:
        # returns the frame (an image), and "ret" tells us whether it worked correctly or not
        ret, frame = cap.read()

        # A dropped frame gives ret=False and frame=None, which crashes everything downstream
        if not ret:
            break

        cv2.imshow('frame', mosaic(frame))

        if cv2.waitKey(1) == ord('q'):
            break

    # release the video capture so other applications can use it
    cap.release()
    cv2.destroyAllWindows()


CONCEPTS = '''\
# Cameras and VideoCapture

A video is just a loop that pulls one image at a time.

## The capture loop

```python
cap = cv2.VideoCapture(0)     # 0 = default camera; a path opens a file instead
while True:
    ret, frame = cap.read()   # ret is False when the camera drops a frame or the file ends
    if not ret:
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()                 # without this the webcam stays locked to this process
```

## waitKey is doing two jobs

- `cv2.waitKey(1)` waits **1 millisecond** for a keypress, then returns. That pause
  is also what gives the window time to repaint - drop it and you get a frozen window.
- `cv2.waitKey(0)` waits **forever**, which is what you want for a still image and
  never what you want in a video loop.

## Frame properties

`cap.get(3)` and `cap.get(4)` are width and height (`CAP_PROP_FRAME_WIDTH` /
`CAP_PROP_FRAME_HEIGHT`). Reading `frame.shape` is usually better: it is the size
of the frame you actually got, not the size the driver claims.

## Building a frame from slices

`np.zeros(frame.shape, np.uint8)` makes a black canvas of the same size, then slice
assignment drops the four quarters in. `dtype=np.uint8` matters - a float array
renders as a white rectangle because OpenCV expects 0-255 integers.
'''

LESSON = Lesson(
    title="Lesson 03 - Cameras & Video",
    summary="VideoCapture loops and rebuilding frames from slices",
    run=run,
    concepts=CONCEPTS,
    previews=(
        Preview(
            label='Single frame',
            build=lambda: safe_imread(STILL_IMAGE),
            caption='Stand-in for one cap.read() frame - a frame is just an image.',
        ),
        Preview(
            label='4-up mosaic',
            build=lambda: mosaic(safe_imread(STILL_IMAGE)),
            caption='The same mosaic() the webcam loop applies, run on a still. Press r for live video.',
        ),
    ),
)

if __name__ == "__main__":
    run()
