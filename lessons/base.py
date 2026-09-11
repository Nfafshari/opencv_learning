from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np


@dataclass(frozen=True)
class Preview:
    '''
    One image the TUI can draw for a lesson.

    `build` is a callable rather than an image so nothing is computed until the
    user actually arrows onto that preview - otherwise every lesson in the
    registry would run its OpenCV work at import time.
    '''
    label: str
    build: Callable[[], np.ndarray]
    caption: str = ''


@dataclass(frozen=True)
class Lesson:
    ''' A class for storing lesson information '''
    title: str
    summary: str
    run: Callable[[], None]
    concepts: str = ''
    previews: tuple[Preview, ...] = field(default_factory=tuple)
