'''
Draw OpenCV images with nothing but colored text, so they can live inside the TUI.

A terminal cell is roughly twice as tall as it is wide, so each cell is treated as
two stacked pixels: the upper half block ('▀') is painted in the *top* pixel's color
and the cell background shows through as the *bottom* pixel's color. That gets us
square-ish pixels and 24-bit color on any truecolor terminal without pulling in a
sixel/kitty image library.
'''

from __future__ import annotations

import cv2
import numpy as np
from rich.console import Console, ConsoleOptions, RenderResult
from rich.color import Color
from rich.segment import Segment
from rich.style import Style

# Top half of the cell is the glyph, bottom half is the background.
UPPER_HALF_BLOCK = '▀'


class TerminalImage:
    '''
    A Rich renderable that scales an image to fit a character grid.

    `width` and `height` are in *cells*. The usable pixel box is therefore
    (width x height * 2), since every cell carries two pixel rows.
    '''

    def __init__(self, image: np.ndarray, width: int, height: int) -> None:
        self.image = image
        self.width = max(1, width)
        self.height = max(1, height)

    def _to_rgb(self) -> np.ndarray:
        '''Normalize grayscale / BGR / BGRA input down to plain RGB.'''
        img = self.image

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _fit(self) -> np.ndarray:
        '''Scale to fit the cell grid, preserving aspect ratio.'''
        img = self._to_rgb()
        src_h, src_w = img.shape[:2]

        # Never upscale past the box; min() on both axes is what keeps the aspect ratio.
        scale = min(self.width / src_w, (self.height * 2) / src_h)

        out_w = max(1, round(src_w * scale))
        out_h = max(1, round(src_h * scale))

        # Two pixel rows per cell, so an odd height would leave a half-drawn row.
        out_h = max(2, out_h - (out_h % 2))

        # INTER_AREA is the right filter for shrinking; it averages instead of dropping pixels.
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        return cv2.resize(img, (out_w, out_h), interpolation=interpolation)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        pixels = self._fit()
        out_h, out_w = pixels.shape[:2]

        # Center the image in whatever space the widget gave us.
        left_pad = max(0, (self.width - out_w) // 2)
        top_pad = max(0, (self.height - out_h // 2) // 2)

        padding = Segment(' ' * left_pad) if left_pad else None
        newline = Segment('\n')

        # Building a Style object per pixel is the slow path, so cache by color pair.
        # Photos have large flat regions, which makes the hit rate high.
        styles: dict[tuple[int, ...], Style] = {}

        def style_for(key: tuple[int, ...]) -> Style:
            style = styles.get(key)
            if style is None:
                style = Style(
                    color=Color.from_rgb(key[0], key[1], key[2]),
                    bgcolor=Color.from_rgb(key[3], key[4], key[5]),
                )
                styles[key] = style
            return style

        for _ in range(top_pad):
            yield newline

        for row in range(0, out_h, 2):
            top = pixels[row]
            bottom = pixels[row + 1]

            if padding is not None:
                yield padding

            # Run-length merge identical neighbours into one Segment. A 100-cell row of
            # sky becomes one segment instead of 100, which keeps redraws cheap.
            previous: tuple[int, ...] | None = None
            run = 0

            for col in range(out_w):
                current = (
                    int(top[col][0]), int(top[col][1]), int(top[col][2]),
                    int(bottom[col][0]), int(bottom[col][1]), int(bottom[col][2]),
                )

                if current == previous:
                    run += 1
                    continue

                if previous is not None:
                    yield Segment(UPPER_HALF_BLOCK * run, style_for(previous))

                previous = current
                run = 1

            if previous is not None:
                yield Segment(UPPER_HALF_BLOCK * run, style_for(previous))

            yield newline
