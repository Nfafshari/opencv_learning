import sys

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Label, ListItem, ListView, Markdown, Static

from lessons.base import Lesson
from lessons.registry import LESSONS
from widgets import ImageView


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
        ("n", "next_preview", "Next image"),
        ("p", "previous_preview", "Prev image"),
        ("r", "run_lesson", "Run in OpenCV"),
        ("f", "toggle_full_image", "Full image"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.lesson: Lesson | None = None
        self.preview_index = 0

        # Previews are built on demand and kept here, keyed by (lesson title, index).
        # Template matching runs six passes over the image, so arrowing back and
        # forth would otherwise redo that work every single time.
        self._preview_cache: dict[tuple[str, int], object] = {}

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
                with Vertical(id="preview-pane"):
                    yield Static("", id="preview-title")
                    yield ImageView(placeholder="Select a lesson.", id="image")
                    yield Static("", id="preview-caption")
                with VerticalScroll(id="concepts-pane"):
                    yield Markdown("", id="concepts")

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#lesson-list", ListView).focus()

    # ------------------------------------------------------------------ list

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        '''Fires as the user arrows through the list.'''
        # event.item is None when the list is empty or the highlight is cleared.
        if not isinstance(event.item, LessonItem):
            return

        self.lesson = event.item.lesson
        self.preview_index = 0

        self.query_one("#concepts", Markdown).update(
            self.lesson.concepts or f'# {self.lesson.title}\n\n{self.lesson.summary}'
        )
        self.query_one("#concepts-pane", VerticalScroll).scroll_home(animate=False)

        self._show_preview()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        '''Fires when the user presses Enter on a row.'''
        self.action_run_lesson()

    # -------------------------------------------------------------- previews

    def _show_preview(self) -> None:
        '''Draw the current preview, building it the first time it is asked for.'''
        image_view = self.query_one("#image", ImageView)
        title = self.query_one("#preview-title", Static)
        caption = self.query_one("#preview-caption", Static)

        if self.lesson is None:
            image_view.show(None, "Select a lesson.")
            return

        previews = self.lesson.previews

        if not previews:
            title.update(self.lesson.title)
            caption.update("")
            image_view.show(None, "This lesson has no still preview - press r to run it.")
            return

        preview = previews[self.preview_index]
        title.update(
            f"[b]{preview.label}[/b]  [dim]{self.preview_index + 1}/{len(previews)}[/dim]"
        )

        key = (self.lesson.title, self.preview_index)

        if key not in self._preview_cache:
            # A preview is arbitrary OpenCV code, so a missing file or a bad call
            # should land in the caption instead of taking the whole app down.
            try:
                self._preview_cache[key] = preview.build()
            except Exception as exc:
                self._preview_cache[key] = exc

        result = self._preview_cache[key]

        if isinstance(result, Exception):
            image_view.show(None, f"Could not build this preview:\n{result}")
            caption.update(f"[red]{type(result).__name__}[/red]")
            return

        image_view.show(result)
        caption.update(f"[dim]{preview.caption}[/dim]")

    def _step_preview(self, step: int) -> None:
        if self.lesson is None or not self.lesson.previews:
            return

        # Modulo wraps in both directions, so n past the end lands back on the first.
        self.preview_index = (self.preview_index + step) % len(self.lesson.previews)
        self._show_preview()

    def action_next_preview(self) -> None:
        self._step_preview(1)

    def action_previous_preview(self) -> None:
        self._step_preview(-1)

    def action_toggle_full_image(self) -> None:
        '''Hide the notes so the image gets the whole right-hand column.'''
        self.query_one("#detail").toggle_class("image-only")

    # ------------------------------------------------------------------- run

    def action_run_lesson(self) -> None:
        '''Hand the terminal over to the real script and its OpenCV windows.'''
        if self.lesson is None:
            return

        lesson = self.lesson

        try:
            # Without suspend(), OpenCV's window and the TUI both fight over the
            # terminal - suspend drops out of the alternate screen until run() returns.
            with self.suspend():
                print(f'--- {lesson.title} ---')
                print('The OpenCV window may open behind this terminal.')
                lesson.run()
        except Exception as exc:
            self.notify(
                f'{type(exc).__name__}: {exc}',
                title=f'{lesson.title} failed',
                severity='error',
                timeout=10,
            )
        else:
            self.notify(f'{lesson.title} finished.', timeout=4)


if __name__ == "__main__":
    app = OpenCvApp()
    app.run()
    sys.exit(app.return_code or 0)
