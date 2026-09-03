from dataclasses import dataclass
from collections.abc import Callable

@dataclass(frozen=True)
class Lesson:
    ''' A class for storing lesson information '''
    title: str
    summary: str
    run: Callable[[], None]
