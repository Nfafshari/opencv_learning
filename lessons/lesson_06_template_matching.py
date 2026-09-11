import cv2

from utils.safe_imread import safe_imread
from lessons.base import Lesson, Preview

# main methods of template matching
METHODS = (
    ('TM_CCOEFF', cv2.TM_CCOEFF),
    ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
    ('TM_CCORR', cv2.TM_CCORR),
    ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
    ('TM_SQDIFF', cv2.TM_SQDIFF),
    ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED),
)

# For SQDIFF a *low* score is a good match; for everything else a high score is.
LOWER_IS_BETTER = (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)


def best_match(image: cv2.typing.MatLike, template: cv2.typing.MatLike, method: int) -> cv2.typing.Point:
    ''' Slide the template over the image and return the top-left corner of the best hit. '''

    # result returns = (W - w + 1, H - h + 1)
    result = cv2.matchTemplate(image, template, method)
    _min_val, _max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return min_loc if method in LOWER_IS_BETTER else max_loc


def match_with(method: int) -> cv2.typing.MatLike:
    ''' Box the penny and the quarter using one matching method. '''

    img = safe_imread('coins1.jpg', cv2.IMREAD_GRAYSCALE)
    penny_template = safe_imread('penny.png', cv2.IMREAD_GRAYSCALE)
    quarter_template = safe_imread('quarter.png', cv2.IMREAD_GRAYSCALE)

    # only has height and width because we loaded it as grayscale
    penny_h, penny_w = penny_template.shape
    quarter_h, quarter_w = quarter_template.shape

    penny_location = best_match(img, penny_template, method)
    quarter_location = best_match(img, quarter_template, method)

    penny_bottom_right = (penny_location[0] + penny_w, penny_location[1] + penny_h)
    quarter_bottom_right = (quarter_location[0] + quarter_w, quarter_location[1] + quarter_h)

    cv2.rectangle(img, penny_location, penny_bottom_right, 255, 4)
    cv2.rectangle(img, quarter_location, quarter_bottom_right, 255, 4)

    return img


def run() -> None:
    '''
    Locate a small template inside a larger image, once per matching method.

    cv2.matchTemplate(), cv2.minMaxLoc(), cv2.rectangle()
    '''

    # Try with all of them, and see which one works the best and use that
    for name, method in METHODS:
        print(f'matching with {name} - press any key for the next method')

        cv2.imshow('match', match_with(method))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


CONCEPTS = '''\
# Template matching

Slide a small image over a big one and score the overlap at every position.

## The two calls

```python
result = cv2.matchTemplate(image, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
```

`result` is **not** an image of the scene - it is a score map, one pixel per
position the template could sit at. Its size is `(W - w + 1, H - h + 1)`, so it is
slightly smaller than the source. `minMaxLoc` finds the best and worst spots in it.

## Which end of the score do you want

| Method | Best match is |
| --- | --- |
| `TM_SQDIFF`, `TM_SQDIFF_NORMED` | the **minimum** - it measures difference |
| everything else | the **maximum** - it measures correlation |

Picking the wrong end of that is the classic bug: the box lands somewhere random and
the code looks fine. The `_NORMED` variants divide out overall brightness, which
makes scores comparable between images and much less sensitive to exposure. Plain
`TM_CCORR` in particular tends to just find the brightest patch.

## Drawing the box

`matchTemplate` gives you the **top-left** corner. The bottom-right is
`(x + template_width, y + template_height)`, which is why the template's `shape`
gets unpacked first. Loaded grayscale, that unpack is `h, w = template.shape` - a
color template would need `shape[:2]`.

## Where it breaks

There is no scale or rotation tolerance at all. Resize the coin by 20% and the match
collapses. It also **always** returns a best position, even when the object is not in
the image - you have to threshold the score yourself to decide whether the hit is
real. For anything beyond exact-size repeats, feature matching is the better tool.
'''

LESSON = Lesson(
    title="Lesson 06 - Template Matching",
    summary="matchTemplate, minMaxLoc, and picking a method",
    run=run,
    concepts=CONCEPTS,
    previews=(
        Preview(
            label='Scene',
            build=lambda: safe_imread('coins1.jpg', cv2.IMREAD_GRAYSCALE),
            caption='The haystack: coins1.jpg in grayscale.',
        ),
        Preview(
            label='Penny template',
            build=lambda: safe_imread('penny.png', cv2.IMREAD_GRAYSCALE),
            caption='The needle: 46x42 crop. Same scale as the scene or matching fails.',
        ),
        Preview(
            label='Quarter template',
            build=lambda: safe_imread('quarter.png', cv2.IMREAD_GRAYSCALE),
            caption='The second needle: 52x55.',
        ),
        *(
            Preview(
                label=name,
                build=lambda method=method: match_with(method),
                caption=(
                    f'{name} - best match is the '
                    f'{"minimum" if method in LOWER_IS_BETTER else "maximum"} score. '
                    'Cycle through all six to see which ones actually land on the coins.'
                ),
            )
            for name, method in METHODS
        ),
    ),
)

if __name__ == "__main__":
    run()
