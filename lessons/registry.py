from lessons import (
    lesson_01_basics,
    lesson_02_image_fundamentals_and_manipulation,
    lesson_03_cameras_and_video_capture,
    lesson_04_color_detection,
    lesson_05_corner_detection,
    lesson_06_template_matching,
)

# Display order for the TUI. Every module here has to expose a `LESSON` and keep its
# OpenCV work inside functions - anything at module scope would fire on import and
# pop a window open behind the TUI.
LESSONS = (
    lesson_01_basics.LESSON,
    lesson_02_image_fundamentals_and_manipulation.LESSON,
    lesson_03_cameras_and_video_capture.LESSON,
    lesson_04_color_detection.LESSON,
    lesson_05_corner_detection.LESSON,
    lesson_06_template_matching.LESSON,
)
