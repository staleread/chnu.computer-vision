import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    import marimo as mo  # comment the import if not using marimo notebook editor
    import cv2 as cv
    import dlib
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from time import perf_counter
    from typing import NamedTuple

    # Some useful types
    Color = tuple[int, int, int]
    RgbImage = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
    GrayImage = np.ndarray[tuple[int, int], np.dtype[np.uint8]]

    class Image(NamedTuple):
        name: str
        rgb: RgbImage
        gray: GrayImage


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CV Lab 9 — Face Detection Demo

    Select an image, toggle grayscale, and adjust the scale to compare dlib and Viola-Jones side by side.
    """)
    return


@app.cell
def _():
    # Define some utility functions

    def load_image(name: str) -> Image:
        bgr = cv.imread(f'data/{name}.jpg')
        rgb: RgbImage = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        gray: GrayImage = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        return Image(name, rgb, gray)

    def random_color() -> Color:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return load_image, random_color


@app.cell
def _(load_image):
    # Load the images for demo
    images: dict[str, Image] = {
        name: load_image(name) for name in [
            "in-glasses", "paint", "bike", "cave",
            "group", "masks", "run", "beatles", "drawing",
        ]
    }
    return (images,)


@app.cell
def _():
    detector = dlib.get_frontal_face_detector()
    cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    return cascade, detector


@app.cell
def _(cascade, detector, random_color):
    def detect_dlib(img: Image, *, gray: bool = False, scale: float = 1.0, thickness: int = 3) -> tuple[np.ndarray, dict]:
        source = np.copy(img.gray if gray else img.rgb)

        if scale != 1.0:
            source = cv.resize(source, (0, 0), fx=scale, fy=scale)

        # Detect faces rectangles with measuring the elapsed time
        start = perf_counter()
        rects = detector(source, 1)
        elapsed = perf_counter() - start

        # Draw rectangles on the source
        for rect in rects:
            cv.rectangle(source, (rect.left(), rect.top()), (rect.right(), rect.bottom()), random_color(), thickness)

        return source, {'faces': len(rects), 'time': elapsed}

    def detect_vj(img: Image, *, gray: bool = False, scale: float = 1.0, thickness: int = 3) -> tuple[np.ndarray, dict]:
        source = np.copy(img.gray if gray else img.rgb)

        if scale != 1.0:
            source = cv.resize(source, (0, 0), fx=scale, fy=scale)

        # Detect faces with measuring the elapsed time
        start = perf_counter()
        faces = cascade.detectMultiScale(source, scaleFactor=1.1, minNeighbors=10, flags=cv.CASCADE_SCALE_IMAGE)
        elapsed = perf_counter() - start

        # Draw rectangles on the source
        for (x, y, w, h) in faces:
            cv.rectangle(source, (x, y), (x + w, y + h), random_color(), thickness)

        return source, {'faces': len(faces), 'time': elapsed}

    return detect_dlib, detect_vj


@app.cell
def _(images: dict[str, Image]):
    image_select = mo.ui.dropdown(list(images.keys()), value="in-glasses", label="Image")
    gray_toggle = mo.ui.switch(label="Grayscale")
    scale_slider = mo.ui.slider(0.1, 100.0, step=0.1, value=100.0, label="Scale (%)")
    thickness_input = mo.ui.number(1, 30, step=1, value=3, label="Thickness")
    return gray_toggle, image_select, scale_slider, thickness_input


@app.cell
def _(
    gray_toggle,
    image_select,
    images: dict[str, Image],
    scale_slider,
    thickness_input,
):
    _controls = mo.hstack([image_select, gray_toggle, scale_slider, thickness_input], justify="start", gap=2)

    # Show image resolution change if downscale was applied
    if scale_slider.value < 100.0:
        _scale = scale_slider.value / 100.0
        _h, _w = images[image_select.value].rgb.shape[:2]
        _info = mo.md(f"Resolution: **{_w}×{_h}** → **{int(_w * _scale)}×{int(_h * _scale)}**")
        _output = mo.vstack([_controls, _info])
    else:
        _output = _controls

    _output
    return


@app.cell(hide_code=True)
def _(
    detect_dlib,
    detect_vj,
    gray_toggle,
    image_select,
    images: dict[str, Image],
    scale_slider,
    thickness_input,
):
    # Gather values from UI elements
    _img = images[image_select.value]
    _gray = gray_toggle.value
    _scale = scale_slider.value / 100.0
    _thickness = int(thickness_input.value)

    # Run the dectection using both methods
    _dlib_img, _dlib_m = detect_dlib(_img, gray=_gray, scale=_scale, thickness=_thickness)
    _vj_img, _vj_m = detect_vj(_img, gray=_gray, scale=_scale, thickness=_thickness)

    _cmap = "gray" if _gray else None
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Plot dlib result
    _ax1.imshow(_dlib_img, cmap=_cmap)
    _ax1.set_title(f"dlib — {_dlib_m['faces']} faces  {_dlib_m['time']:.3f} s", fontsize=13)
    _ax1.axis('off')

    # Plot VJ result
    _ax2.imshow(_vj_img, cmap=_cmap)
    _ax2.set_title(f"Viola-Jones — {_vj_m['faces']} faces  {_vj_m['time']:.3f} s", fontsize=13)
    _ax2.axis('off')

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
