import random

import cv2

from utils.safe_imread import safe_imread
from lessons.base import Lesson, Preview

IMAGE_NAME = 'camera_man_w_noise.jpg'


def scrambled_rows(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Overwrite the top 100 rows with random pixels, one pixel at a time.

    Edits `img` in place - a NumPy array is mutable and is *not* copied when
    passed into a function.
    '''

    for i in range(100):
        for j in range(img.shape[1]):
            img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    return img


def copied_region(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    ''' Copy one rectangle of the image over another rectangle. '''

    # Numpy array slice allows to slice twice
    # INDEX 0 = ROWS
    # INDEX 1 = COLUMNS
    camera = img[100:200, 50:200]
    # REPLACE SECTION
    img[0:100, 75:225] = camera

    return img


def manipulated() -> cv2.typing.MatLike:
    ''' Both edits applied to a fresh copy of the source image. '''
    return copied_region(scrambled_rows(safe_imread(IMAGE_NAME)))


def run() -> None:
    '''
    Images are NumPy arrays: shape, indexing, slicing, and pasting regions.

    img.shape, pixel assignment, slice assignment
    '''

    # you can pass -1, 0, 1, for flags IMREAD_UNCHANGED, IMREAD_GRAYSCALE, and IMREAD_COLOR, respectively
    img = safe_imread(IMAGE_NAME, 1)
    img2 = safe_imread(IMAGE_NAME, 1)

    print(f'''
Note!
imread() returns a numpy array, which is an optimized library for python arrays.
you can get the shape of an array using array.shape which gives you the:
* num of rows
* num of cols
* num of channels

numpy array allows for more array functionality
''')

    print(f'0. Rows (height) - {img.shape[0]}\n1. Columns (width) - {img.shape[1]}\n2. Channels (RGB or BGR) - {img.shape[2]}\nFULL: {img.shape}')

    print(f'first row of our image (showing only first 25 pixels): {img[0][0:25]}')

    print(f'lets change the colors of the pixels!')
    img = scrambled_rows(img)

    print(f'Now lets copy one part of the image to another')
    img = copied_region(img)

    cv2.imshow("Image", img)
    cv2.imshow("Image 2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


CONCEPTS = '''\
# An image is just a NumPy array

Everything OpenCV hands back is an `ndarray`, so NumPy indexing *is* the image API.

## Shape

`img.shape` is `(rows, cols, channels)` - height first, width second:

| Index | Meaning |
| --- | --- |
| `shape[0]` | rows / height |
| `shape[1]` | columns / width |
| `shape[2]` | channels (3 for BGR) |

A grayscale image has **no** third element, so `h, w = img.shape` works there but
blows up on a color image.

## imread flags

`-1` `IMREAD_UNCHANGED` - keeps an alpha channel
`0` `IMREAD_GRAYSCALE` - 2D array
`1` `IMREAD_COLOR` - 3D BGR array (the default)

## Indexing

- `img[row][col]` is a single pixel: `[B, G, R]`, **not** RGB.
- `img[100:200, 50:200]` slices rows then columns - the reverse of `(x, y)`.
- Slice assignment pastes a region: `img[0:100, 75:225] = patch`. The shapes have
  to match exactly or NumPy raises.

## The gotcha worth remembering

A slice is a **view**, not a copy. `patch = img[100:200, 50:200]` still points at
the original buffer, so later edits to `img` show up in `patch`. Use `.copy()` when
you want a real snapshot - that is why this lesson reads a second `img2` from disk
instead of reusing the first.
'''

LESSON = Lesson(
    title="Lesson 02 - Image Fundamentals",
    summary="Images as NumPy arrays: shape, indexing, slicing",
    run=run,
    concepts=CONCEPTS,
    previews=(
        Preview(
            label='Original',
            build=lambda: safe_imread(IMAGE_NAME),
            caption='shape = (225, 225, 3) - rows, columns, BGR channels.',
        ),
        Preview(
            label='Grayscale flag',
            build=lambda: safe_imread(IMAGE_NAME, cv2.IMREAD_GRAYSCALE),
            caption='imread(..., 0) drops to a 2D array - no channel axis at all.',
        ),
        Preview(
            label='Scrambled rows',
            build=lambda: scrambled_rows(safe_imread(IMAGE_NAME)),
            caption='Top 100 rows overwritten pixel by pixel with random BGR values.',
        ),
        Preview(
            label='Region copied',
            build=manipulated,
            caption='img[0:100, 75:225] = img[100:200, 50:200] - slice assignment.',
        ),
    ),
)

if __name__ == "__main__":
    run()
