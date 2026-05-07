# CV Lab 10 - Object Tracking

## Trackers Overview
*   **MIL (Multiple Instance Learning):** Brief principle (use of positive and negative examples in "bags" to handle tracker drift).
*   **TLD (Tracking-Learning-Detection):** Brief principle (combination of a tracker, detector, and learning via P-N experts for long-term tracking).

## Experiment #1: Bus and Pedestrian (data/bus.gif)
*   **Tracking Object:** Bus.
*   **Scenario:** The bus slowly moves away from the camera (downscaling), while a pedestrian crosses between the bus and the camera (partial occlusion).
*   **MIL Results:** The pedestrian walking does cause the tracked object rectangle to shift slightly but the tracker recovers quickly
- **TLD Results:** The tracker doesn't follow the object correctly (with some shifts) and the occlusion caused by a pedestrian make it fail for some frames
*   **Comparison:** MIL did better at resisting the occlusion

## Experiment #2: Drifting (data/drifting.mp4)
*   **Tracking Object:** Single car wheel.
*   **Scenario:** Change in perspective (side view -> drift/rear view), the wheel disappearing behind an obstacle for several frames.
*   **MIL Results:** The tracked rectangle follows the object with good precision. When the object is behind the obstacle, the algorithm tries to "guess" the next position (kinda inertia) and doesn't report the absence. After a couple of frames after appearing again, the tracker makes a successful recovery. The perspective change is handled well.
*   **TLD Results:** There's a slight shift of the tracked rectangle as the car (and its wheels) moves faster in the viewport. When the target hides behind the obstacle the tracker correctly states the object did so, but after some frames it starts tracking the false positives. When the object appears again it gets tracked immediately. The change in perspective leads to the tracker highlighting a false positive.
*   **Analysis:** How do trackers react to geometric deformation of the object and temporary disappearance?

## Experiment #3: Drifting (data/drifting.mp4)
*   **Tracking Objects:** Two car wheels.
*   **MIL Results:** After disappearing and appearing again the trackers don't follow the target object but the regions that might be guessed (feels like inertia)
*   **TLD Results:** A false positive is tracked for the front wheel when the real one disappears behind an obstacle. Then both trackers point to the rear wheel 
*   **Analysis:** How do trackers react to geometric deformation of the object and temporary disappearance?

## 5. Experiment #4: Waiter's Tip & Glass (data/glass.gif)
*   **Tracking Object:** Glass.
*   **Scenario:** Static object position but dynamic background and significant lighting changes.
*   **MIL Results:** Does track the object correctly no matter the lightning changes
*   **TLD Results:** For a few frame the tracker tries his best at keeping the object followed but as the background changes significantly as well as the lightning, the tracker starts spamming false positive
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
