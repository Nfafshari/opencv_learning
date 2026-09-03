import sys

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, RichLog, Static

from lessons.base import Lesson
from lessons.registry import LESSONS


class LessonItem(ListItem):
    '''
    A single row in the lesson list.

    Subclassing ListItem so each row carries its own Lesson object. The
    alternative is indexing into LESSONS by row number, which silently breaks
    the moment display order and registry order drift apart.
    '''

    def __init__(self, lesson: Lesson) -> None:
        super().__init__()
        self.lesson = lesson

    def compose(self) -> ComposeResult:
        yield Label(self.lesson.title)


class OpenCvApp(App):
    CSS_PATH = "globals.tcss"
    TITLE = "OpenCV Tutorials"
    SUB_TITLE = "Fundamental OpenCV practice lessons"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="body"):
            with Vertical(id='lesson-section'):
                yield Static("Lessons:", id='lessons-title')
                yield ListView(
                    *(LessonItem(lesson) for lesson in LESSONS),
                    id="lesson-list",
                )
            with Vertical(id="detail"):
                yield Static("Select a lesson.", id="summary")
                yield RichLog(id="log", markup=True)

        yield Footer()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        '''Fires as the user arrows through the list.'''
        # TODO:
        #   1. Grab the highlighted row from `event.item`.
        #      Guard against None - it is None when the list is empty.
        #   2. That row is a LessonItem, so it carries `.lesson`.
        #   3. Find the summary pane with self.query_one("#summary", Static)
        #      and push the lesson's summary into it with .update().
        pass

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        '''Fires when the user presses Enter on a row.'''
        # TODO:
        #   1. Same first two steps as above - read `event.item`, guard, get `.lesson`.
        #   2. Find the log with self.query_one("#log", RichLog) and .write() the title.
        #
        # Do NOT call lesson.run() here yet. That is Milestone 3, and it has to be
        # wrapped in `with self.suspend():` or the OpenCV window and the TUI will
        # fight over the terminal.
        pass


if __name__ == "__main__":
    app = OpenCvApp()
    app.run()
    sys.exit(app.return_code or 0)
