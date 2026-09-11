import cv2

from utils.safe_imread import safe_imread
from lessons.base import Lesson, Preview

IMAGE_NAME = 'camera_man_w_noise.jpg'


def resized() -> cv2.typing.MatLike:
    ''' Resize by an exact pixel size, then again by a scale factor. '''
    img = safe_imread(IMAGE_NAME)

    # resize an image by pixels
    img = cv2.resize(img, (600, 600))
    # OR by scale
    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)  # increase the photo by half its size

    return img


def rotated() -> cv2.typing.MatLike:
    ''' Rotate the resized image a quarter turn counter clockwise. '''
    return cv2.rotate(resized(), cv2.ROTATE_90_COUNTERCLOCKWISE)


def run() -> None:
    '''
    openCV basic usage for reading, resizing, rotating, and saving an image.

    cv2.imread(), cv2.resize(), cv2.rotate(), cv2.imwrite()
    '''

    img = rotated()

    ### SAVE AN IMAGE
    # cv2.imwrite('new_image.jpg', img)

    # show the image in a window
    cv2.imshow('Image', img)

    # close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


CONCEPTS = '''\
# Reading, resizing, rotating, saving

The four calls that show up in almost every OpenCV script.

## Key calls

- `cv2.imread(path, flag)` - returns a **NumPy array** in **BGR** order, or `None`
  if the path is wrong. It does not raise, so always guard the result.
- `cv2.resize(img, (w, h))` - note the size is `(width, height)`, the opposite
  order from `img.shape`, which is `(rows, cols, channels)`.
- `cv2.resize(img, (0, 0), fx=1.5, fy=1.5)` - pass `(0, 0)` to scale by a factor
  instead of an absolute size.
- `cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)` - only 90 degree steps. For an
  arbitrary angle you need `cv2.warpAffine` with a rotation matrix.
- `cv2.imwrite('out.jpg', img)` - the extension decides the encoder.

## Gotchas

- Rotating 90 degrees **swaps width and height**, so anything downstream that
  assumed the old shape breaks.
- `cv2.imshow` needs `cv2.waitKey(0)` after it or the window never paints - the
  wait loop is what pumps the GUI event queue.
- Relative paths resolve against the *working directory*, not the script, which is
  why this repo uses `utils/safe_imread.py`.
'''

LESSON = Lesson(
    title="Lesson 01 - OpenCV Basics",
    summary="Resize, Rotate, and Save images",
    run=run,
    concepts=CONCEPTS,
    previews=(
        Preview(
            label='Original',
            build=lambda: safe_imread(IMAGE_NAME),
            caption='225 x 225, read straight off disk with imread().',
        ),
        Preview(
            label='Resized',
            build=resized,
            caption='resize() to 600x600, then again by fx=1.5 / fy=1.5.',
        ),
        Preview(
            label='Rotated',
            build=rotated,
            caption='rotate() a quarter turn counter clockwise.',
        ),
    ),
)

if __name__ == "__main__":
    run()
