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

## Experiment #4: Waiter's Tip & Glass (data/glass.gif)
*   **Tracking Object:** Glass.
*   **Scenario:** Static object position but dynamic background and significant lighting changes.
*   **MIL Results:** Does track the object correctly no matter the lightning changes
*   **TLD Results:** For a few frame the tracker tries his best at keeping the object followed but as the background changes significantly as well as the lightning, the tracker starts spamming false positive
*   **Analysis:** Impact of background noise and illumination on tracking stability.

## Tracker comparison
*   **Observed differences in algorithm performance:**
    *   **MIL** demonstrated higher stability during partial occlusions and significant lighting changes. It tends to stay on the object even when the visual features are slightly distorted.
    *   **TLD** is more sensitive to background noise and illumination. However, it is better at identifying when the object has truly disappeared from the frame, whereas MIL often continues to "track" the background or an empty space (inertia).
*   **Strengths and weaknesses:**
    *   **MIL Strengths:** Robust to occlusion, handles lighting variations well, simple to initialize.
    *   **MIL Weaknesses:** Prone to gradual drift, lacks a dedicated re-detection mechanism (it "forgets" the object if it moves too far from the last known position), doesn't explicitly report tracking failure.
    *   **TLD Strengths:** Includes a detector that can re-localize the object after full occlusion, can report when the object is lost.
    *   **TLD Weaknesses:** Highly susceptible to false positives in complex backgrounds, struggles with rapid scale changes and significant illumination shifts.

## Tracking Duration Study
*   **Stable Scenarios (e.g., glass.gif with MIL):** In scenarios with static camera and consistent object scale, MIL maintained a perfect track for the entire duration (18+ frames) despite lighting changes.
*   **Dynamic Scenarios (e.g., drifting.mp4):** 
    *   In the "drifting" video, MIL tracked the wheel for approximately 15-20 frames before the car drifted behind an obstacle. It "guessed" the position for 3-5 frames and successfully recovered once the wheel reappeared.
    *   TLD maintained the track for a shorter period (~10-12 frames) before being distracted by background features or false positives as the car's perspective changed.
*   **Drift Analysis:** Significant drift was observed in TLD when the object's appearance changed due to rotation or scale, leading to tracking failures or jumps to similar-looking background patches. MIL showed less drift but tended to expand or contract the bounding box inaccurately over time.

## Multi-Object Tracking
*   **Demonstration:** Using the "drifting.mp4" video, we initialized trackers on both the front and rear wheels of the car.
*   **Observed Behavior:**
    *   **MIL:** Both trackers maintained their respective wheels well until the drift maneuver began. When the front wheel was obscured, its tracker drifted into the car body (inertia).
    *   **TLD:** TLD struggled with the similarity between the two wheels. When one wheel became obscured, its tracker occasionally jumped to the other wheel, identifying it as a match (false positive localization).
*   **Performance Impact:** Tracking two objects roughly doubled the processing time per frame. Since the trackers run sequentially in the demo implementation, adding more ROIs directly impacts the real-time performance of the application.
