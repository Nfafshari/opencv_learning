from __future__ import annotations

import numpy as np
from rich.align import Align
from rich.console import RenderableType
from rich.text import Text
from textual.app import RenderResult
from textual.widget import Widget

from utils.terminal_image import TerminalImage


class ImageView(Widget):
    '''
    Shows a single OpenCV image (BGR numpy array) inside the TUI.

    The image is *not* a reactive attribute on purpose: Textual compares the old
    and new value to decide whether to refresh, and `old_array != new_array` on a
    numpy array returns an array rather than a bool, which raises. So the image is
    a plain attribute and `show()` asks for the redraw explicitly.
    '''

    def __init__(self, placeholder: str = 'No preview.', **kwargs) -> None:
        super().__init__(**kwargs)
        self.image: np.ndarray | None = None
        self.placeholder = placeholder

    def show(self, image: np.ndarray | None, placeholder: str | None = None) -> None:
        '''Swap in a new image (or None to fall back to the placeholder text).'''
        self.image = image

        if placeholder is not None:
            self.placeholder = placeholder

        self.refresh()

    def on_resize(self) -> None:
        # The image is fitted to the cell grid at render time, so any size change
        # means the cached render is the wrong shape.
        self.refresh()

    def render(self) -> RenderResult:
        if self.image is None:
            message: RenderableType = Text(self.placeholder, style='dim italic')
            return Align.center(message, vertical='middle')

        return TerminalImage(self.image, self.size.width, self.size.height)
