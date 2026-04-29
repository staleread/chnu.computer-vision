import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    import marimo as mo # comment thi if not using Marimo notebook editor
    import cv2
    import os
    import tempfile
    import imageio
    import matplotlib.pyplot as plt


@app.function
def get_tracker(tracker_type: str):
    if tracker_type == "MIL":
        return cv2.TrackerMIL.create()
    if tracker_type == "TLD":
        return cv2.legacy.TrackerTLD.create()
    return None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CV Lab 10 — Object Tracking Demo

    This notebook demonstrates object tracking using OpenCV trackers.

    1. **Select a video** and **tracker**.
    2. **Configure the region of interest** using sliders.
    3. **Run tracking** to see the result.
    """)
    return


@app.cell
def _():
    # Get videos file paths
    _video_files = [f"data/{file}" for file in os.listdir("data") if file.endswith(('.mp4', '.gif'))]

    video_select = mo.ui.dropdown(
        options=_video_files, 
        value=_video_files[0] if _video_files else None, 
        label="Select Video"
    )
    tracker_select = mo.ui.dropdown(
        options=["MIL", "TLD"], 
        value="MIL", 
        label="Select Tracker"
    )

    mo.vstack([video_select, tracker_select])
    return tracker_select, video_select


@app.cell
def _(video_select):
    mo.stop(video_select.value is None)

    _cap = cv2.VideoCapture(video_select.value)
    _ret, first_frame = _cap.read()
    _cap.release()

    if not _ret:
        mo.md("Failed to load video.")
        mo.stop(True)
    return (first_frame,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Configure the region of interest (ROI) using sliders
    """)
    return


@app.cell
def _(first_frame):
    _h, _w = first_frame.shape[:2]

    x_slider = mo.ui.slider(start=0, stop=_w-1, step=1, value=_w//4, label="X", full_width=True)
    y_slider = mo.ui.slider(start=0, stop=_h-1, step=1, value=_h//4, label="Y", full_width=True)
    w_slider = mo.ui.slider(start=1, stop=_w, step=1, value=_w//2, label="Width", full_width=True)
    h_slider = mo.ui.slider(start=1, stop=_h, step=1, value=_h//2, label="Height", full_width=True)

    mo.vstack([x_slider, y_slider, w_slider, h_slider])
    return h_slider, w_slider, x_slider, y_slider


@app.cell
def _(first_frame, h_slider, w_slider, x_slider, y_slider):
    # Preview selection
    _preview = first_frame.copy()
    _x, _y, _bw, _bh = x_slider.value, y_slider.value, w_slider.value, h_slider.value

    # Ensure bbox is within display_frame bounds
    _h_disp, _w_disp = first_frame.shape[:2]
    _bw = min(_bw, _w_disp - _x)
    _bh = min(_bh, _h_disp - _y)

    cv2.rectangle(_preview, (_x, _y), (_x + _bw, _y + _bh), (0, 255, 0), 2)

    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(_preview, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("ROI Preview")
    return


@app.cell
def _():
    run_button = mo.ui.run_button(label="Start Tracking")

    run_button
    return (run_button,)


@app.cell
def _(
    first_frame,
    h_slider,
    run_button,
    tracker_select,
    video_select,
    w_slider,
    x_slider,
    y_slider,
):
    mo.stop(not run_button.value)

    _x, _y, _bw, _bh = x_slider.value, y_slider.value, w_slider.value, h_slider.value

    # Final bounds check on original resolution
    _h_orig, _w_orig = first_frame.shape[:2]
    _x = max(0, min(_x, _w_orig - 1))
    _y = max(0, min(_y, _h_orig - 1))
    _bw = max(1, min(_bw, _w_orig - _x))
    _bh = max(1, min(_bh, _h_orig - _y))

    _orig_bbox = (_x, _y, _bw, _bh)

    _tracker = get_tracker(tracker_select.value)
    _tracker.init(first_frame, _orig_bbox)

    _cap = cv2.VideoCapture(video_select.value)
    _total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _max_frames = min(150, _total_frames) if _total_frames > 0 else 150

    _frames = []

    # Correct way to use marimo progress bar with a loop
    for _ in mo.status.progress_bar(range(_max_frames), title="Tracking..."):
        _success, _frame = _cap.read()
        if not _success:
            break

        _success, _trk_bbox = _tracker.update(_frame)

        if _success:
            (_bx, _by, _btw, _bth) = [int(_v) for _v in _trk_bbox]
            cv2.rectangle(_frame, (_bx, _by), (_bx + _btw, _by + _bth), (0, 255, 0), 2)
            cv2.putText(_frame, tracker_select.value, (_bx, _by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(_frame, "Tracking failure", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        _frames.append(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB))

    _cap.release()

    tracking_video = None
    if _frames:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as _tmp:
            imageio.mimsave(_tmp.name, _frames, fps=30, format='FFMPEG', codec='libx264', macro_block_size=None)
            tracking_video = mo.video(_tmp.name)
    return (tracking_video,)


@app.cell
def _(tracking_video):
    mo.stop(tracking_video is None)
    mo.vstack([
        mo.md("### Tracking Result"),
        tracking_video
    ])
    return


if __name__ == "__main__":
    app.run()
