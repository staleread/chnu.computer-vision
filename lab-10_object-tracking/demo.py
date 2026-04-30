import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    import marimo as mo  # comment this if not using Marimo notebook editor
    import cv2
    import os
    import json
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


@app.cell
def _():
    # Profile management
    # The lab requires the analysis of different trackers on different
    # scenarios like object scaling, background or lightning changes.
    # In order to be make the setup persistent a JSON file is used as
    # a storage. The state management of marimo makes the "profile" creation
    # easy with possibly many setups for a single video by using a unique label

    config_path = "config.json"

    def load_profiles():
        import os
        import json

        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    get_profiles, set_profiles = mo.state(load_profiles())
    get_active_profile, set_active_profile = mo.state({"name": "Custom", "rois": []})
    get_roi_index, set_roi_index = mo.state(0)
    return (
        config_path,
        get_active_profile,
        get_profiles,
        get_roi_index,
        set_active_profile,
        set_profiles,
        set_roi_index,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Profile Management

    Save and load your tracking configurations:
    - **Select Profile**: Choose an existing configuration to instantly restore all settings.
    - **Profile Label**: Add a custom tag to your current setup before saving.
    - **Save Profile**: Click to store your current video, range, and ROI settings.

    *Note: Manually changing any setting will switch the profile back to "Custom".*
    """)
    return


@app.cell
def _(get_active_profile, get_profiles, set_active_profile):
    def on_profile_change(v):
        if v == "Custom":
            set_active_profile({"name": "Custom"})
        else:
            _profile_data = get_profiles().get(v, {})
            set_active_profile({**_profile_data, "name": v})

    _options = ["Custom"] + list(get_profiles().keys())
    _current_profile = get_active_profile()
    _current_name = _current_profile.get("name", "Custom")

    if _current_name not in _options:
        _current_name = "Custom"

    profile_dropdown = mo.ui.dropdown(
        options=_options,
        value=_current_name,
        label="Select Profile",
        on_change=on_profile_change,
    )

    profile_dropdown
    return


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
def _(get_active_profile, set_active_profile):
    # Get video file paths
    _video_files = [
        f"data/{file}" for file in os.listdir("data") if file.endswith((".mp4", ".gif"))
    ]

    _active = get_active_profile()
    _default_video = _active.get("video")
    if _default_video not in _video_files:
        _default_video = _video_files[0] if _video_files else None

    video_select = mo.ui.dropdown(
        options=_video_files,
        value=_default_video,
        label="Select Video",
        on_change=lambda v: set_active_profile({**get_active_profile(), "name": "Custom", "video": v}),
    )

    video_select
    return (video_select,)


@app.cell
def _(video_select):
    mo.stop(video_select.value is None)

    _cap = cv2.VideoCapture(video_select.value)
    total_frames: int = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    return (total_frames,)


@app.cell
def _(get_active_profile, set_active_profile, total_frames: int):
    _active = get_active_profile()

    _stop = max(0, total_frames - 1)
    _default_range = _active.get("frame_range", [0, min(150, _stop)])

    # Validate and clamp range
    if (
        not isinstance(_default_range, list)
        or len(_default_range) != 2
        or not all(isinstance(x, (int, float)) for x in _default_range)
    ):
        _default_range = [0, min(150, _stop)]
    else:
        # Ensure start <= end and both are within [0, _stop]
        _r_start = max(0, min(_default_range[0], _stop))
        _r_end = max(_r_start, min(_default_range[1], _stop))
        _default_range = [_r_start, _r_end]

    frame_range = mo.ui.range_slider(
        start=0,
        stop=_stop,
        step=1,
        value=_default_range,
        label="Frame Range",
        full_width=True,
        on_change=lambda v: set_active_profile({**get_active_profile(), "name": "Custom", "frame_range": v}),
    )

    frame_range
    return (frame_range,)


@app.cell
def _(frame_range, video_select):
    mo.stop(video_select.value is None)

    _cap = cv2.VideoCapture(video_select.value)
    _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range.value[0])
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
def _(
    first_frame,
    get_active_profile,
    get_roi_index,
    set_active_profile,
    set_roi_index,
):
    h, w = first_frame.shape[:2]
    active = get_active_profile()
    rois = active.get("rois", [])

    # Ensure at least one ROI exists
    if not rois:
        rois = [{"x": w // 4, "y": h // 4, "w": w // 2, "h": h // 2}]

    # Local index clamped to current list
    idx = min(get_roi_index(), len(rois) - 1)
    current_roi = rois[idx]

    def _get_val(key, default, hi, lo=0):
        return max(lo, min(current_roi.get(key, default), hi))

    def sync_roi(new_parts):
        current = get_active_profile()
        current_rois = list(current.get("rois", rois))
        # Important: use current index from state to avoid closure stale idx
        _c_idx = min(get_roi_index(), len(current_rois) - 1)
        current_rois[_c_idx] = {**current_rois[_c_idx], **new_parts}
        set_active_profile({**current, "name": "Custom", "rois": current_rois})

    def add_roi(_):
        current = get_active_profile()
        current_rois = list(current.get("rois", rois))
        current_rois.append({"x": w // 4, "y": h // 4, "w": w // 2, "h": h // 2})
        set_active_profile({**current, "name": "Custom", "rois": current_rois})
        set_roi_index(len(current_rois) - 1)

    def remove_roi(_):
        current = get_active_profile()
        current_rois = list(current.get("rois", rois))
        if len(current_rois) > 1:
            _c_idx = min(get_roi_index(), len(current_rois) - 1)
            current_rois.pop(_c_idx)
            set_active_profile({**current, "name": "Custom", "rois": current_rois})
            set_roi_index(max(0, _c_idx - 1))

    # UI for switching between ROIs
    roi_selector = mo.ui.tabs(
        {f"ROI {i+1}": mo.md("") for i in range(len(rois))},
        value=f"ROI {idx+1}",
        on_change=lambda v: set_roi_index(int(v.split()[-1]) - 1)
    )

    add_button = mo.ui.button(label="Add ROI", on_click=add_roi)
    remove_button = mo.ui.button(label="Remove ROI", on_click=remove_roi, disabled=len(rois) <= 1)

    profile_label = mo.ui.text(label="Profile Label", placeholder="e.g. wheels")
    save_button = mo.ui.run_button(
        label="Save Profile",
        disabled=(active.get("name") != "Custom"),
    )

    # Sliders for ROI selection
    x_slider = mo.ui.slider(
        start=0, stop=w - 1, step=1, value=_get_val("x", w // 4, w - 1),
        label="X", full_width=True, on_change=lambda v: sync_roi({"x": v}),
    )
    y_slider = mo.ui.slider(
        start=0, stop=h - 1, step=1, value=_get_val("y", h // 4, h - 1),
        label="Y", full_width=True, on_change=lambda v: sync_roi({"y": v}),
    )
    w_slider = mo.ui.slider(
        start=1, stop=w, step=1, value=_get_val("w", w // 2, w, lo=1),
        label="Width", full_width=True, on_change=lambda v: sync_roi({"w": v}),
    )
    h_slider = mo.ui.slider(
        start=1, stop=h, step=1, value=_get_val("h", h // 2, h, lo=1),
        label="Height", full_width=True, on_change=lambda v: sync_roi({"h": v}),
    )

    roi_controls = mo.vstack([
        mo.hstack([roi_selector, add_button, remove_button], justify="start"),
        mo.hstack([profile_label, save_button], justify="start", align="end"),
        x_slider, y_slider, w_slider, h_slider
    ])
    return (
        h_slider,
        profile_label,
        roi_controls,
        rois,
        save_button,
        w_slider,
        x_slider,
        y_slider,
    )


@app.cell
def _(h_slider, w_slider, x_slider, y_slider):
    # Get slider values
    x, y, bw, bh = x_slider.value, y_slider.value, w_slider.value, h_slider.value
    return bh, bw, x, y


@app.cell
def _(roi_controls):
    roi_controls
    return


@app.cell
def _(bh, bw, first_frame, rois, x, y):
    # Preview ROI selection
    _preview = first_frame.copy()

    # Draw all ROIs, highlight the current one in Red
    for _r in rois:
        _bx, _by, _bw, _bh = _r["x"], _r["y"], _r["w"], _r["h"]
        _color = (0, 255, 0) # Green for others
        # Check if this is the one currently being edited
        if _bx == x and _by == y and _bw == bw and _bh == bh:
            _color = (0, 0, 255) # Red for active
        cv2.rectangle(_preview, (_bx, _by), (_bx + _bw, _by + _bh), _color, 2)

    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(_preview, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ROI Preview (Current ROI in Red)")
    return


@app.cell
def _():
    # UI button for starting tracking
    tracker_select = mo.ui.dropdown(
        options=["MIL", "TLD"],
        value="MIL",
        label="Select Tracker"
    )
    run_button = mo.ui.run_button(label="Start Tracking")

    mo.hstack([tracker_select, run_button])
    return run_button, tracker_select


@app.cell
def _(
    config_path,
    frame_range,
    get_profiles,
    profile_label,
    rois,
    save_button,
    set_active_profile,
    set_profiles,
    video_select,
):

    if save_button.value:
        # Construct profile name
        _base = os.path.basename(video_select.value).replace(".", "_")
        _label = profile_label.value.strip()
        _name = f"{_base}_{_label}" if _label else _base

        _new_profile = {
            "video": video_select.value,
            "frame_range": frame_range.value,
            "rois": rois,
        }

        _profiles = get_profiles()
        _profiles[_name] = _new_profile

        with open(config_path, "w") as _f:
            json.dump(_profiles, _f, indent=4)

        set_profiles(_profiles)
        set_active_profile({**_new_profile, "name": _name})
    return


@app.cell
def _(
    first_frame,
    frame_range,
    rois,
    run_button,
    tracker_select,
    video_select,
):
    mo.stop(not run_button.value)

    _trackers = []
    for _r in rois:
        _t = get_tracker(tracker_select.value)
        _t.init(first_frame, (_r["x"], _r["y"], _r["w"], _r["h"]))
        _trackers.append(_t)

    _cap = cv2.VideoCapture(video_select.value)
    _start, _end = frame_range.value
    _cap.set(cv2.CAP_PROP_POS_FRAMES, _start)
    _max_frames = _end - _start + 1

    _frames = []

    # Process each frame in the selected range
    for i in mo.status.progress_bar(range(int(_max_frames)), title="Tracking..."):
        _success, _frame = _cap.read()
        if not _success:
            break

        for _idx, _tracker in enumerate(_trackers):
            if i == 0:
                # First frame is the initialization frame: show manual ROI
                _r = rois[_idx]
                _bx, _by, _btw, _bth = _r["x"], _r["y"], _r["w"], _r["h"]
                _color = (255, 0, 0)  # Blue for manual
                _label = f"Box {_idx+1} Initial"
                _show_box = True
            else:
                # Subsequent frames: show tracker result
                _success_trk, _trk_bbox = _tracker.update(_frame)
                if _success_trk:
                    (_bx, _by, _btw, _bth) = [int(_v) for _v in _trk_bbox]
                    _color = (0, 255, 0)  # Green for tracker
                    _label = f"Box {_idx+1} ({tracker_select.value})"
                    _show_box = True
                else:
                    _show_box = False

            if _show_box:
                cv2.rectangle(_frame, (_bx, _by), (_bx + _btw, _by + _bth), _color, 2)
                cv2.putText(
                    _frame,
                    _label,
                    (_bx, _by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    _color,
                    2,
                )
            else:
                cv2.putText(
                    _frame,
                    f"Box {_idx+1} failure",
                    (50, 50 + (_idx * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

        # Ensure dimensions are even for FFMPEG (libx264 yuv420p)
        _h, _w = _frame.shape[:2]
        if _h % 2 != 0 or _w % 2 != 0:
            _frame = _frame[: _h - (_h % 2), : _w - (_w % 2)]

        _frames.append(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB))

    _cap.release()

    tracking_video = None

    if _frames:
        # Save frames with rectanges to a temporary file so we can show it with marimo
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as _tmp:
            imageio.mimsave(
                _tmp.name,
                _frames,
                fps=30,
                format="FFMPEG",
                codec="libx264",
                macro_block_size=None,
            )
            tracking_video = mo.video(_tmp.name)
    return (tracking_video,)


@app.cell
def _(tracking_video):
    mo.stop(tracking_video is None)
    mo.vstack([mo.md("### Tracking Result"), tracking_video])
    return


if __name__ == "__main__":
    app.run()
