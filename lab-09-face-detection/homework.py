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
    # CV Lab 9 — Face Detection

    This lab compares two classical face detectors — **dlib HOG+SVM** and **Viola-Jones (Haar cascade)** — across single-face and multi-face photographs. For each detector we measure:

    - baseline performance on full-resolution RGB images
    - the effect of switching to **grayscale** input
    - the speed/recall trade-off of **downscaling** the image before detection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dataset

    Nine photos split into two groups:

    | Group | Indices | Description |
    |---|---|---|
    | Single face | 0–3 | One subject at increasing distances: selfie, portrait, ~2 m, ~5 m |
    | Multiple faces | 4–8 | Groups from ~3 faces (7, 8) up to a crowd of 77 (6, "run") |

    Image 8 ("drawing") is a sketched scene — an edge case for a frontal-face model trained on photos.
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
    ## 1. dlib HOG + SVM detector
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
    ### 1.1 Single-face images (0–3)

    Full-resolution RGB — establishes the baseline per-image processing time.
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

    All four detected correctly. Time scales roughly with pixel count. Same images in grayscale:
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
    | Method | Image | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|
    | dlib | 0 | 1.362 | 0.893 | 34.4% ⚡ |
    | dlib | 1 | 0.451 | 0.298 | 33.9% ⚡ |
    | dlib | 2 | 0.696 | 0.465 | 33.2% ⚡ |
    | dlib | 3 | 0.333 | 0.224 | 32.7% ⚡ |

    **~33% consistent speedup** across all resolutions and distances, independent of image size.

    Next: how far can we shrink the image before detection fails?
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
    Each scale was tuned to the minimum at which detection still succeeds. The face's pixel size in the *compressed* image is what matters, not the original resolution.

    | Method | Image | Scale | Compressed size | Time (s) | Times faster |
    |---|---|---|---|---|---|
    | dlib | 0 | 0.03 | 54x96 | 0.00140 | 973x 🚀 |
    | dlib | 1 | 0.095 | 114x152 | 0.00460 | 98x 🔥 |
    | dlib | 2 | 0.158 | 275x275 | 0.01822 | 38x |
    | dlib | 3 | 0.45 | 405x720 | 0.06978 | 4.8x |

    The selfie (image 0) at 3% of its original size still detects fine — because the face fills most of the frame. Image 3 (subject at ~5 m) needs 45% of the original to keep the face large enough, so the speedup is modest.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Same scales, grayscale input — does the gray speedup hold when images are already tiny?
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
    Detection succeeds in all cases. The gray speedup is smaller here than at full resolution:

    | Method | Image | Compressed size | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|---|
    | dlib | 0 | 54x96 | 0.00140 | 0.00116 | 17.3% ⚡ |
    | dlib | 1 | 114x152 | 0.00460 | 0.00368 | 19.9% ⚡ |
    | dlib | 2 | 275x275 | 0.01822 | 0.01299 | 28.7% ⚡ |
    | dlib | 3 | 405x720 | 0.06978 | 0.05176 | 25.8% ⚡ |

    At tiny sizes (54×96) the gray conversion cost is a larger slice of total time, so the benefit is real but proportionally smaller (**17–29%** vs. the ~33% seen at full resolution). The practical recommendation is to **combine both**: compress first, then pass grayscale.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.2 Multi-face images (4–8)

    Full-resolution RGB baseline.
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

    Image 5 ("masks") returns only 4 detections despite showing a group — medical masks occlude the lower face, breaking the HOG frontal template. Image 6 takes 5.7 s due to its 6048×4024 resolution. In grayscale:
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

    The **~32% speedup** holds across all multi-face images, confirming it is independent of scene complexity or face count. Detection counts are virtually identical — grayscale does not reduce recall.

    #### Scale sensitivity

    Since image 6 is the most expensive (6048×4024, 77 faces), and image 7 is a compact multi-face photo, we test how aggressively each can be downscaled.
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
    Image 7 ("beatles") — 4 people, original 1204×995. Notable: scale 0.5 is the only one that finds all four faces. Scales 0.7 and 0.4 each miss a *different* person (bottom and rightmost respectively), meaning the full-resolution pass also misses one. The 0.5 scale happens to produce the best recall here.

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
    Image 6 ("run") — crowd of 77 faces, original 6048×4024. At 0.4 scale almost nothing is lost; scaling lower causes face count to drop dramatically as distant faces fall below the detector's minimum detectable size.

    | Method | Image | Resolution | Scale | Faces | Face loss | Time (s) | Speed improvement |
    |---|---|---|---|---|---|---|---|
    | dlib | 6 | 6048x4024 | 1.0 | 77 | — | 5.681 | baseline |
    | dlib | 6 | 2419x1610 | 0.4 | 75 | 2 (2.6%) | 0.886 | 6.4x faster ⚡ |
    | dlib | 6 | 1996x1328 | 0.33 | 58 | 19 (24.7%) | 0.606 | 9.4x faster 🔥 |
    | dlib | 6 | 1814x1207 | 0.3 | 48 | 29 (37.7%) | 0.498 | 11.4x faster 🚀 |

    **0.4 scale is the practical limit** for this image: 6.4× faster, only 2.6% loss.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Viola-Jones (Haar cascade) detector

    `cv2.CascadeClassifier` with `haarcascade_frontalface_default.xml`. Parameters: `scaleFactor=1.1`, `minNeighbors=10`.
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
    ### 2.1 Single-face images (0–3)

    Full-resolution RGB baseline.
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

    VJ is **5–10× faster than dlib** at full resolution. Images 1 and 3 return 2 detections — false positives are possible at `minNeighbors=10`. In grayscale:
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
    | Method | Image | RGB faces | Gray faces | RGB time (s) | Gray time (s) | Speed improvement |
    |---|---|---|---|---|---|---|
    | VJ | 0 | 1 | 1 | 0.138 | 0.144 | -4.3% 🔴 |
    | VJ | 1 | 2 | 4 | 0.112 | 0.127 | -13.4% 🔴 |
    | VJ | 2 | 1 | 1 | 0.433 | 0.380 | 12.2% ⚡ |
    | VJ | 3 | 2 | 2 | 0.166 | 0.163 | 1.8% |

    Unlike dlib's stable ~33% gain, VJ shows **no consistent benefit from grayscale** — results range from −13% to +12%. Notably, image 1 in grayscale returns 4 detections vs. 2 in RGB, showing that colour information suppresses some false positives. Next: downscaling.
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
    Same minimum-detectable scales as dlib, for a direct comparison:

    | Method | Image | Scale | Compressed size | Time (s) | Times faster |
    |---|---|---|---|---|---|
    | VJ | 0 | 0.03 | 54x96 | 0.00135 | 102x 🚀 |
    | VJ | 1 | 0.14 | 168x224 | 0.00787 | 14x ⚡ |
    | VJ | 2 | 0.158 | 275x275 | 0.00983 | 44x 🔥 |
    | VJ | 3 | 0.45 | 405x720 | 0.03793 | 4.4x |

    VJ gains less from compression than dlib (102× vs. 973× for image 0) because it was already much faster at full resolution. In grayscale:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Same scales, grayscale input:
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
    ### 2.2 Multi-face images (4–8)

    Full-resolution RGB baseline.
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

    VJ finds more detections than dlib on image 4 (11 vs. 9): one extra is a partially-occluded face dlib misses, one is a false positive. On image 5 VJ finds only 1 face while dlib finds 4, missing 3.

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

    At scale 0.4, VJ already loses 22.4% of faces — compare with dlib's 2.6% at the same scale. Downscaling hurts VJ's recall much faster.
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

    Scale 0.3 is the sweet spot: 5.1× speedup with no face loss. Below that the detector misses one face.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusions

    | | dlib HOG+SVM | Viola-Jones (Haar) |
    |---|---|---|
    | Single-face recall | 4/4 at full res | 4/4 at full res |
    | Multi-face recall (image 6) | 77 faces | 76 faces |
    | Grayscale speedup | ~33% consistent | negligible / inconsistent |
    | Grayscale at compressed sizes | faster | slower (−80–98%) |
    | Face loss at 0.4 scale (image 6) | 2.6% | 22.4% |
    | Base speed (image 6, full res) | 5.681 s | 2.325 s |

    **dlib** is slower at full resolution but degrades gracefully when downscaled, making it practical for high-resolution images with a tuned scale factor.

    **Viola-Jones** is faster out of the box and matches dlib's recall on most images, but suffers much steeper face loss when downscaled and does not benefit from grayscale input.

    For single-face or small-group photos either detector works well. For dense crowds at high resolution, dlib with modest downscaling (0.4×) offers the better speed/recall trade-off.
    """)
    return


if __name__ == "__main__":
    app.run()
