# CV Lab 10 - Object Tracking

## 1. Objectives
Research and comparison of object tracking algorithms (MIL and TLD) across various video scenarios. Analysis of their robustness to changes in scale, lighting, occlusions, and background dynamics.

## 2. Trackers Overview
*   **MIL (Multiple Instance Learning):** Brief principle (use of positive and negative examples in "bags" to handle tracker drift).
*   **TLD (Tracking-Learning-Detection):** Brief principle (combination of a tracker, detector, and learning via P-N experts for long-term tracking).

## 3. Experiment #1: Bus and Pedestrian (data/bus.gif)
*   **Tracking Object:** Bus.
*   **Scenario:** The bus slowly moves away from the camera (downscaling), while a pedestrian crosses between the bus and the camera (partial occlusion).
*   **MIL Results:** The pedestrian walking does cause the tracked object rectangle to shift slightly but the tracker recovers quickly
- **TLD Results:** The tracker doesn't follow the object correctly (with some shifts) and the occlusion caused by a pedestrian make it fail for some frames
*   **Comparison:** MIL did better at resisting the occlusion

## 4. Experiment #2: Drifting (data/drifting.mp4)
*   **Tracking Object:** Single car wheel.
*   **Scenario:** Change in perspective (side view -> drift/rear view), the wheel disappearing behind an obstacle for several frames.
*   **MIL Results:**
*   **TLD Results:**
*   **Analysis:** How do trackers react to geometric deformation of the object and temporary disappearance?

## 5. Experiment #3: Waiter's Tip / Glass (data/glass.gif / data/fish.gif / data/pizza.gif)
*   **Tracking Object:** Glass (or other target object).
*   **Scenario:** Static object position but dynamic background and significant lighting changes.
*   **MIL Results:**
*   **TLD Results:**
*   **Analysis:** Impact of background noise and illumination on tracking stability.

## 6. Comparative Analysis (Task 10.1)
*   Observed differences in algorithm performance.
*   Strengths and weaknesses of MIL vs. TLD based on the tests.

## 7. Tracking Duration Study (Task 10.2)
*   Analysis of the number of frames each tracker maintains the object without significant drift.
*   Comparison between stable scenarios (constant scale) and dynamic scenarios.

## 8. Multi-Object Tracking (Task 10.3)
*   Demonstration of Multi-ROI tracking (e.g., simultaneous tracking of multiple wheels or both the pedestrian and the bus).
*   Impact of the number of objects on processing performance.

## 9. Conclusions
General summary of algorithm robustness and recommendations for use-cases.

---

## Ideas for Additional Video (#4)
To complement the experiments, consider one of the following scenarios:
1.  **Fast Motion / Motion Blur:** A fast-moving object (sports ball, bird) causing blur. This tests the tracker's ability to handle non-sharp features.
2.  **Similar Distractors:** Tracking an object moving among similar ones (e.g., one white car among several other white cars).
3.  **Low Contrast:** An object with colors very similar to the background (e.g., a grey car on grey asphalt in low light).
