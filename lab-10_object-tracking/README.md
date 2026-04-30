# CV Lab 10 — Object Tracking Demo

An interactive object tracking application built with **marimo** and **OpenCV**. This tool allows users to perform multi-object tracking on videos and GIFs, with persistent configuration management.

## Features

- **Multi-Object Tracking**: Track multiple regions of interest (ROIs) simultaneously in a single video.
- **Profile Management**: Save and load tracking configurations (Video, Frame Range, and ROIs) to a `config.json` file for persistence.
- **OpenCV Trackers**: Support for multiple tracking algorithms (MIL, TLD).
- **Interactive ROI Selection**: Fine-tune bounding boxes using sliders and a tabbed interface for multiple objects.
- **Custom Frame Ranges**: Select specific segments of a video to analyze.
- **Visual Feedback**: Real-time ROI preview and rendered tracking results with distinct labels for manual detection vs. algorithmic tracking.

## Setup

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
uv sync
uv run marimo edit demo.py
```

## Usage

1. **Select a Video**: Choose from the available files in the `data/` directory.
2. **Set Frame Range**: Use the range slider to define the segment of the video you want to track.
3. **Configure ROIs**: 
   - Add one or more "ROIs" using the "Add ROI" button.
   - Adjust the X, Y, Width, and Height of each box using the sliders.
   - Switch between boxes using the tabbed interface.
4. **Save Profile** (Optional): Provide a label and save your current configuration to quickly restore it later.
5. **Run Tracking**: Select your preferred tracker (e.g., MIL or TLD) and click **Start Tracking**. 
6. **View Result**: Once processing is complete, the tracking video will be displayed in the notebook.
