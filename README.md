# opencv_learning

A personal sandbox for learning the basics of **OpenCV** and **machine vision**.

## Contents

Scripts are numbered in the order I worked through them:
- `images/` - sample images for practice: noise variants (Gaussian, salt & pepper, uniform), coins, screws, and koala.
- `1_opencv_basics.py` - reading, resizing, rotating, saving, and displaying an image.
- `2_image_fundamentals_and_manipulation.py` - images as NumPy arrays (rows, cols, channels) and `imread` flags.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install opencv-python numpy
```

## Run

```bash
python 1_opencv_basics.py
```

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
