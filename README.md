# opencv_learning

A personal sandbox for learning the basics of **OpenCV** and **machine vision**.

## Contents

- `main.py` - a Textual TUI that browses the lessons, renders their result images
  **inside the terminal**, and shows the notes for each one.
- `lessons/` - one module per lesson, numbered in the order I worked through them.
  Each exposes a `LESSON` with its notes and still previews, plus a `run()` that
  opens the real OpenCV windows.
- `projects/` - small end-to-end projects (noise filtering, thresholding,
  morphology, camera calibration).
- `images/` - sample images: noise variants (Gaussian, salt & pepper, uniform),
  coins, screws, chessboards, and koala.
- `utils/terminal_image.py` - draws an OpenCV image as colored half blocks so it
  can be displayed in the TUI.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e .
```

## Run

```bash
python main.py              # the lesson browser
python -m lessons.lesson_01_basics   # or any single lesson on its own
```

### Keys

| Key | Action |
| --- | --- |
| up / down | move between lessons |
| `n` / `p` | next / previous preview image |
| `r` or Enter | run the lesson for real, in an OpenCV window |
| `f` | give the whole pane to the image |
| `q` | quit |

Images are drawn with the upper half block character, one cell per two pixels, so
a **truecolor terminal** (Windows Terminal, VS Code, iTerm2) gives the best result.

## Topics being explored

- Image I/O and basic transforms (resize, rotate, save)
- Image fundamentals and manipulation
- Cameras and VideoCapture
- Drawing lines, images, circles, and text
- Colors and color detection
- Corner detection
- Template matching (Object detection)
- Face and eye detection

>[!NOTE]
> All credit to **Tech With Tim** on YouTube who made great video tutorials about openCV.
> This repo is for me to learn OpenCV better and practice with some fun small projects.
