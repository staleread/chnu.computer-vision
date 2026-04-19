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


@app.cell
def _():
    # Some utility functions

    def load_image(name: str, filepath: str | None = None) -> Image:
        bgr = cv.imread(filepath if filepath else f'data/{name}.jpg')
        rgb: RgbImage = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        gray: GrayImage = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

        return Image(name, rgb, gray)

    def load_images_by_names(names: list[str]) -> list[Image]:
        return [load_image(name) for name in names]

    def random_color() -> tuple[int, int, int]:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return load_images_by_names, random_color


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CV lab 9. Face detection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First of all, let's take a look at the photos selected for the experiment
    """)
    return


@app.cell
def _(load_images_by_names):
    images: list[Image] = load_images_by_names([
        "in-glasses",
        "paint",
        "bike",
        "cave",
        "group",
        "masks",
        "run",  # or "fans"
        "beatles",
        "drawing"
        ])
    return (images,)


@app.cell
def _(images: list[Image]):
    # Plotting images in a 3x3 grid

    _fig, _axes = plt.subplots(3, 3, figsize=(12, 9))

    for i, (_ax, _img) in enumerate(zip(_axes.flatten(), images)):
        _ax.imshow(_img.rgb)
        _ax.set_title(f"{i}: {_img.name}")
        _ax.axis('off')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we'll load the `dlib` **frontal** face detector and define a utility function for plotting the photo with rectangles on detected faces
    """)
    return


@app.cell
def _():
    detector = dlib.get_frontal_face_detector()
    return (detector,)


@app.cell
def _(detector, random_color):
    def show_detected_faces_dlib(
        img: Image,
        *,
        ax,
        thickness: int = 2,
        color: Color | None = None,
        gray: bool = False,
        scale: float = 1.0,
    ):
        # Work on a copy for visualization
        result = np.copy(img.gray if gray else img.rgb)

        # Resize the image on demand
        if scale != 1.0:
            h_old, w_old = result.shape[:2]
            result = cv.resize(result, (0, 0), fx=scale, fy=scale)
            h_new, w_new = result.shape[:2]

            print(f'[DLIB] scaled the image from {w_old}x{h_old} to {w_new}x{h_new}')

        # Detect faces (upsampling = 1)
        start = perf_counter()
        rects = detector(result, 1)
        end = perf_counter()

        elapsed_time = end - start

        print('[DLIB] Number of detected faces:', len(rects), 'Elapsed time (s)', elapsed_time)

        # Draw rectangles around faces
        for rect in rects:
            cv.rectangle(
                result,
                (rect.left(), rect.top()),
                (rect.right(), rect.bottom()),
                color or random_color(),
                thickness)

        ax.imshow(result, cmap="gray" if gray else None)
        ax.axis('off')

    return (show_detected_faces_dlib,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Detecting single face

    Let's test some solo photos in RGB mode
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_dlib(images[0], ax=_axes_flat[0], thickness=20)
    show_detected_faces_dlib(images[1], ax=_axes_flat[1], thickness=10)
    show_detected_faces_dlib(images[2], ax=_axes_flat[2], thickness=10)
    show_detected_faces_dlib(images[3], ax=_axes_flat[3], thickness=5)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Method | Image | Faces | Image size | Time (s) |
    |---|---|---|---|---|
    | dlib | 0 | 1 | 1808x3216 | 1.362 |
    | dlib | 1 | 1 | 1200x1600 | 0.451 |
    | dlib | 2 | 1 | 1741x1741 | 0.696 |
    | dlib | 3 | 1 | 901x1600 | 0.333 |

    And in gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_dlib(images[0], ax=_axes_flat[0], thickness=20, gray=True)
    show_detected_faces_dlib(images[1], ax=_axes_flat[1], thickness=10, gray=True)
    show_detected_faces_dlib(images[2], ax=_axes_flat[2], thickness=10, gray=True)
    show_detected_faces_dlib(images[3], ax=_axes_flat[3], thickness=5, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the table comparing RGB vs Grayscale processing times:

    | Method | Image | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|
    | dlib | 0 | 1.362 | 0.893 | 34.4% ⚡ |
    | dlib | 1 | 0.451 | 0.298 | 33.9% ⚡ |
    | dlib | 2 | 0.696 | 0.465 | 33.2% ⚡ |
    | dlib | 3 | 0.333 | 0.224 | 32.7% ⚡ |

    Consistent **~33% speed improvement** across all images regardless of size.

    Now let's experiment with lower image resolution!
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_dlib(images[0], ax=_axes_flat[0], thickness=1, scale=0.03)
    show_detected_faces_dlib(images[1], ax=_axes_flat[1], thickness=1, scale=0.095)
    show_detected_faces_dlib(images[2], ax=_axes_flat[2], thickness=2, scale=0.158)
    show_detected_faces_dlib(images[3], ax=_axes_flat[3], thickness=2, scale=0.45)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Method | Image | Scale | Compressed size | Time (s) | Times faster |
    |---|---|---|---|---|---|
    | dlib | 0 | 0.03 | 54x96 | 0.00140 | 973x 🚀 |
    | dlib | 1 | 0.095 | 114x152 | 0.00460 | 98x 🔥 |
    | dlib | 2 | 0.158 | 275x275 | 0.01822 | 38x |
    | dlib | 3 | 0.45 | 405x720 | 0.06978 | 4.8x |

    The closer the face, the more we can compress and the faster detection runs.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's try to process the compressed pic using gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_dlib(images[0], ax=_axes_flat[0], thickness=1, scale=0.03, gray=True)
    show_detected_faces_dlib(images[1], ax=_axes_flat[1], thickness=1, scale=0.095, gray=True)
    show_detected_faces_dlib(images[2], ax=_axes_flat[2], thickness=2, scale=0.158, gray=True)
    show_detected_faces_dlib(images[3], ax=_axes_flat[3], thickness=2, scale=0.45, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The algorithm detects faces correctly in grayscale at the same compression rate.

    | Method | Image | Compressed size | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|---|
    | dlib | 0 | 54x96 | 0.00140 | 0.00116 | 17.3% ⚡ |
    | dlib | 1 | 114x152 | 0.00460 | 0.00368 | 19.9% ⚡ |
    | dlib | 2 | 275x275 | 0.01822 | 0.01299 | 28.7% ⚡ |
    | dlib | 3 | 405x720 | 0.06978 | 0.05176 | 25.8% ⚡ |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Detecting multiple faces
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(15, 15))

    show_detected_faces_dlib(images[4], ax=_axes['a'], thickness=15)
    show_detected_faces_dlib(images[5], ax=_axes['b'], thickness=5)
    show_detected_faces_dlib(images[7], ax=_axes['c'], thickness=10)
    show_detected_faces_dlib(images[8], ax=_axes['d'], thickness=10)
    show_detected_faces_dlib(images[6], ax=_axes['e'], thickness=10)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Method | Image | Faces | Time (s) |
    |---|---|---|---|
    | dlib | 4 | 9 | 2.116 |
    | dlib | 5 | 4 | 0.048 |
    | dlib | 7 | 3 | 0.278 |
    | dlib | 8 | 3 | 0.157 |
    | dlib | 6 | 77 | 5.681 |

    And in gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(15, 15))

    show_detected_faces_dlib(images[4], ax=_axes['a'], thickness=12, gray=True)
    show_detected_faces_dlib(images[5], ax=_axes['b'], thickness=5, gray=True)
    show_detected_faces_dlib(images[7], ax=_axes['c'], thickness=10, gray=True)
    show_detected_faces_dlib(images[8], ax=_axes['d'], thickness=10, gray=True)
    show_detected_faces_dlib(images[6], ax=_axes['e'], thickness=10, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Method | Image | RGB faces | Gray faces | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|---|---|
    | dlib | 4 | 9 | 10 | 2.116 | 1.438 | 32.0% ⚡ |
    | dlib | 5 | 4 | 4 | 0.048 | 0.035 | 27.1% |
    | dlib | 7 | 3 | 3 | 0.278 | 0.190 | 31.7% ⚡ |
    | dlib | 8 | 3 | 3 | 0.157 | 0.107 | 31.8% ⚡ |
    | dlib | 6 | 77 | 76 | 5.681 | 3.863 | 32.0% ⚡ |
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))

    show_detected_faces_dlib(images[7], ax=_axes[0], thickness=7, scale=0.7)
    show_detected_faces_dlib(images[7], ax=_axes[1], thickness=5, scale=0.5)
    show_detected_faces_dlib(images[7], ax=_axes[2], thickness=3, scale=0.4)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Scale precision vs. speed for image 7 (beatles, 1204x995 original, 3 faces):

    | Method | Image | Resolution | Scale | Faces | Time (s) | Speed improvement |
    |---|---|---|---|---|---|---|
    | dlib | 7 | 1204x995 | 1.0 | 3 | 0.278 | baseline |
    | dlib | 7 | 843x696 | 0.7 | 3 | 0.137 | 2.0x faster ⚡ |
    | dlib | 7 | 602x498 | 0.5 | 4 | 0.070 | 4.0x faster ⚡ |
    | dlib | 7 | 482x398 | 0.4 | 3 | 0.045 | 6.2x faster 🚀 |
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))

    show_detected_faces_dlib(images[6], ax=_axes[0], thickness=7, scale=0.4)
    show_detected_faces_dlib(images[6], ax=_axes[1], thickness=5, scale=0.33)
    show_detected_faces_dlib(images[6], ax=_axes[2], thickness=3, scale=0.3)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Scale precision vs. speed for image 6 (run, 6048x4024 original, 77 faces):

    | Method | Image | Resolution | Scale | Faces | Face loss | Time (s) | Speed improvement |
    |---|---|---|---|---|---|---|---|
    | dlib | 6 | 6048x4024 | 1.0 | 77 | — | 5.681 | baseline |
    | dlib | 6 | 2419x1610 | 0.4 | 75 | 2 (2.6%) | 0.886 | 6.4x faster ⚡ |
    | dlib | 6 | 1996x1328 | 0.33 | 58 | 19 (24.7%) | 0.606 | 9.4x faster 🔥 |
    | dlib | 6 | 1814x1207 | 0.3 | 48 | 29 (37.7%) | 0.498 | 11.4x faster 🚀 |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Viola-Jones face detection
    """)
    return


@app.cell
def _():
    face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    return (face_cascade,)


@app.cell
def _(face_cascade, random_color):
    def show_detected_faces_vj(
        img: Image,
        *,
        ax,
        thickness: int = 2,
        color: Color | None = None,
        gray: bool = False,
        scale: float = 1.0,
    ):
        # Work on a copy for visualization
        result = np.copy(img.gray if gray else img.rgb)

        # Resize the image if needed
        if scale != 1.0:
            h_old, w_old = result.shape[:2]
            result = cv.resize(result, (0, 0), fx=scale, fy=scale)
            h_new, w_new = result.shape[:2]

            print(f'[VJ] scaled the image from {w_old}x{h_old} to {w_new}x{h_new}')

        start = perf_counter()
        faces = face_cascade.detectMultiScale(result,
                                             scaleFactor=1.1,
                                             minNeighbors=10,
                                             flags=cv.CASCADE_SCALE_IMAGE)
        end = perf_counter()
        elapsed_time = end - start

        print('[VJ] Number of detected faces:', len(faces), 'Elapsed time (s)', elapsed_time)

        # Draw rectangles around faces
        for (x, y, w, h) in faces: 
            cv.rectangle(result, (x, y), (x+w, y+h), color or random_color(), thickness)

        ax.imshow(result, cmap="gray" if gray else None)
        ax.axis('off')

    return (show_detected_faces_vj,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Detecting single face

    Let's test some solo photos in RGB mode
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_vj(images[0], ax=_axes_flat[0], thickness=20)
    show_detected_faces_vj(images[1], ax=_axes_flat[1], thickness=10)
    show_detected_faces_vj(images[2], ax=_axes_flat[2], thickness=10)
    show_detected_faces_vj(images[3], ax=_axes_flat[3], thickness=10)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Method | Image | Faces | Image size | Time (s) |
    |---|---|---|---|---|
    | VJ | 0 | 1 | 1808x3216 | 0.138 |
    | VJ | 1 | 2 | 1200x1600 | 0.112 |
    | VJ | 2 | 1 | 1741x1741 | 0.433 |
    | VJ | 3 | 2 | 901x1600 | 0.166 |

    And in gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_vj(images[0], ax=_axes_flat[0], thickness=20, gray=True)
    show_detected_faces_vj(images[1], ax=_axes_flat[1], thickness=10, gray=True)
    show_detected_faces_vj(images[2], ax=_axes_flat[2], thickness=10, gray=True)
    show_detected_faces_vj(images[3], ax=_axes_flat[3], thickness=10, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Unlike dlib, VJ does not benefit consistently from grayscale — results are mixed:

    | Method | Image | RGB faces | Gray faces | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|---|---|
    | VJ | 0 | 1 | 1 | 0.138 | 0.144 | -4.3% 🔴 |
    | VJ | 1 | 2 | 4 | 0.112 | 0.127 | -13.4% 🔴 |
    | VJ | 2 | 1 | 1 | 0.433 | 0.380 | 12.2% ⚡ |
    | VJ | 3 | 2 | 2 | 0.166 | 0.163 | 1.8% |

    Now let's experiment with lower image resolution!
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_vj(images[0], ax=_axes_flat[0], thickness=1, scale=0.03)
    show_detected_faces_vj(images[1], ax=_axes_flat[1], thickness=2, scale=0.14)
    show_detected_faces_vj(images[2], ax=_axes_flat[2], thickness=2, scale=0.158)
    show_detected_faces_vj(images[3], ax=_axes_flat[3], thickness=2, scale=0.45)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Method | Image | Scale | Compressed size | Time (s) | Times faster |
    |---|---|---|---|---|---|
    | VJ | 0 | 0.03 | 54x96 | 0.00135 | 102x 🚀 |
    | VJ | 1 | 0.14 | 168x224 | 0.00787 | 14x ⚡ |
    | VJ | 2 | 0.158 | 275x275 | 0.00983 | 44x 🔥 |
    | VJ | 3 | 0.45 | 405x720 | 0.03793 | 4.4x |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's try to process the compressed pic using gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_vj(images[0], ax=_axes_flat[0], thickness=1, scale=0.03, gray=True)
    show_detected_faces_vj(images[1], ax=_axes_flat[1], thickness=1, scale=0.14, gray=True)
    show_detected_faces_vj(images[2], ax=_axes_flat[2], thickness=2, scale=0.158, gray=True)
    show_detected_faces_vj(images[3], ax=_axes_flat[3], thickness=2, scale=0.45, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    VJ in gray mode is slower at compressed sizes — the opposite of dlib:

    | Method | Image | Compressed size | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|---|
    | VJ | 0 | 54x96 | 0.00135 | 0.00133 | 1.4% |
    | VJ | 1 | 168x224 | 0.00787 | 0.01558 | -98.1% 🔴 |
    | VJ | 2 | 275x275 | 0.00983 | 0.01838 | -87.0% 🔴 |
    | VJ | 3 | 405x720 | 0.03793 | 0.06837 | -80.3% 🔴 |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Detecting multiple faces
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(15, 15))

    show_detected_faces_vj(images[4], ax=_axes['a'], thickness=14)
    show_detected_faces_vj(images[5], ax=_axes['b'], thickness=5)
    show_detected_faces_vj(images[7], ax=_axes['c'], thickness=10)
    show_detected_faces_vj(images[8], ax=_axes['d'], thickness=10)
    show_detected_faces_vj(images[6], ax=_axes['e'], thickness=10)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Method | Image | Faces | Time (s) |
    |---|---|---|---|
    | VJ | 4 | 11 | 0.623 |
    | VJ | 5 | 1 | 0.022 |
    | VJ | 7 | 4 | 0.058 |
    | VJ | 8 | 2 | 0.060 |
    | VJ | 6 | 76 | 2.325 |

    And in gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(15, 15))

    show_detected_faces_vj(images[4], ax=_axes['a'], thickness=14, gray=True)
    show_detected_faces_vj(images[5], ax=_axes['b'], thickness=5, gray=True)
    show_detected_faces_vj(images[7], ax=_axes['c'], thickness=10, gray=True)
    show_detected_faces_vj(images[8], ax=_axes['d'], thickness=10, gray=True)
    show_detected_faces_vj(images[6], ax=_axes['e'], thickness=10, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Method | Image | RGB faces | Gray faces | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|---|---|
    | VJ | 4 | 11 | 12 | 0.623 | 0.626 | -0.4% |
    | VJ | 5 | 1 | 2 | 0.022 | 0.023 | -5.4% |
    | VJ | 7 | 4 | 4 | 0.058 | 0.055 | 5.3% |
    | VJ | 8 | 2 | 2 | 0.060 | 0.053 | 11.7% ⚡ |
    | VJ | 6 | 76 | 74 | 2.325 | 2.317 | 0.3% |

    Unlike dlib's consistent ~32% gain, VJ shows negligible or even negative effect from grayscale on multi-face images.
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))

    show_detected_faces_vj(images[6], ax=_axes[0], thickness=12, scale=0.4)
    show_detected_faces_vj(images[6], ax=_axes[1], thickness=8, scale=0.33)
    show_detected_faces_vj(images[6], ax=_axes[2], thickness=5, scale=0.3)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Scale precision vs. speed for image 6 (run, 6048x4024 original, 76 faces):

    | Method | Image | Resolution | Scale | Faces | Face loss | Time (s) | Speed improvement |
    |---|---|---|---|---|---|---|---|
    | VJ | 6 | 6048x4024 | 1.0 | 76 | — | 2.325 | baseline |
    | VJ | 6 | 2419x1610 | 0.4 | 59 | 17 (22.4%) | 0.448 | 5.2x faster ⚡ |
    | VJ | 6 | 1996x1328 | 0.33 | 49 | 27 (35.5%) | 0.328 | 7.1x faster ⚡ |
    | VJ | 6 | 1814x1207 | 0.3 | 45 | 31 (40.8%) | 0.297 | 7.8x faster 🚀 |
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 9))

    show_detected_faces_vj(images[7], ax=_axes[0], thickness=2, scale=0.3)
    show_detected_faces_vj(images[7], ax=_axes[1], thickness=2, scale=0.25)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Scale precision vs. speed for image 7 (beatles, 1204x995 original, 4 faces):

    | Method | Image | Resolution | Scale | Faces | Face loss | Time (s) | Speed improvement |
    |---|---|---|---|---|---|---|---|
    | VJ | 7 | 1204x995 | 1.0 | 4 | — | 0.058 | baseline |
    | VJ | 7 | 361x298 | 0.3 | 4 | 0 (0%) | 0.011 | 5.1x faster ⚡ |
    | VJ | 7 | 301x249 | 0.25 | 3 | 1 (25%) | 0.009 | 6.2x faster 🚀 |
    """)
    return


if __name__ == "__main__":
    app.run()
