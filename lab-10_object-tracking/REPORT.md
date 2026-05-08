# CV Lab 10 - Object Tracking

## Trackers Overview
*   **MIL (Multiple Instance Learning):** Brief principle (use of positive and negative examples in "bags" to handle tracker drift).
*   **TLD (Tracking-Learning-Detection):** Brief principle (combination of a tracker, detector, and learning via P-N experts for long-term tracking).

## Experiment #1: Bus and Pedestrian (data/bus.gif)
* **Tracking Object:** Bus.
* **Scenario:** The bus slowly moves away from the camera (downscaling), while a pedestrian crosses between the bus and the camera (partial occlusion).
<img width="1190" height="767" alt="image" src="https://github.com/user-attachments/assets/3fd2f9ea-542f-495a-bbac-970ec885dca8" />

**MIL Results:** The pedestrian walking does cause the tracked object rectangle to shift slightly but the tracker recovers quickly
<img width="1183" height="766" alt="image" src="https://github.com/user-attachments/assets/eba1caaf-b5ad-4fc1-838f-64fd1b6adac8" />

**TLD Results:** The tracker doesn't follow the object correctly (with some shifts) and the occlusion caused by a pedestrian make it fail for some frames
<img width="1182" height="770" alt="image" src="https://github.com/user-attachments/assets/e80a4b7f-b4cf-4d4a-b04c-5e2657d3ea0e" />

**Comparison:** MIL did better at resisting the occlusion

## Experiment #2: Drifting (data/drifting.mp4)
*   **Tracking Object:** Single car wheel.
*   **Scenario:** Change in perspective (side view -> drift/rear view), the wheel disappearing behind an obstacle for several frames.
<img width="1184" height="592" alt="image" src="https://github.com/user-attachments/assets/0964865d-c063-4151-b72e-8cd4574af5b7" />

**MIL Results:** The tracked rectangle follows the object with good precision. When the object is behind the obstacle, the algorithm tries to "guess" the next position (kinda inertia) and doesn't report the absence.
<img width="1092" height="542" alt="image" src="https://github.com/user-attachments/assets/7d084131-92e7-422d-9198-7f29271c1dab" />

After a couple of frames after appearing again, the tracker makes a successful attempt of recovery
<img width="695" height="326" alt="image" src="https://github.com/user-attachments/assets/b1c9f9b7-9b9f-46a6-a743-7ccc5909d53a" />

The change in perspective is handled well.
<img width="696" height="322" alt="image" src="https://github.com/user-attachments/assets/8aae51db-4cb9-41f7-9304-17b4a7256648" />

**TLD Results:** There's a slight shift of the tracked rectangle as the car (and its wheels) moves faster in the viewport. When the target hides behind the obstacle the tracker correctly states the object did so. 

<img width="672" height="325" alt="image" src="https://github.com/user-attachments/assets/d7325dd3-d2a9-4618-9827-25da6fbfad54" />

When the object appears again it gets tracked immediately.
<img width="668" height="321" alt="image" src="https://github.com/user-attachments/assets/84fd9853-07a1-4fd9-a215-d0ee55b2761e" />

The change in perspective is handled well, without any false positives.
<img width="690" height="320" alt="image" src="https://github.com/user-attachments/assets/d3f4394c-64aa-4a77-82d9-bc98ff76bc9d" />

## Experiment #3: Drifting (data/drifting.mp4)
*   **Tracking Objects:** Two car wheels.
<img width="697" height="322" alt="image" src="https://github.com/user-attachments/assets/5aabb4d3-b808-4393-917f-d4ef992882ab" />

**MIL Results:**

When the first wheel disappears the second tracker doesn't jump to the remaining wheel but keep following the guessed location of the object
<img width="702" height="328" alt="image" src="https://github.com/user-attachments/assets/da9207d3-c300-4e78-967a-5934be9527e3" />

The same thing with the rear wheel disappearing. The recovery for the front wheel is rather poor.
<img width="689" height="326" alt="image" src="https://github.com/user-attachments/assets/4d8faeb8-a590-49b6-a1c0-281cc89b0ca9" />

But the rear wheel tracker manages to recover well after the obtacle is passed
<img width="697" height="330" alt="image" src="https://github.com/user-attachments/assets/ced87da9-fe64-4b3d-8ef0-6490ebc18dad" />

The perspective change is handled OK
<img width="690" height="323" alt="image" src="https://github.com/user-attachments/assets/c0be77ea-9f30-42b2-af16-352b59050ef5" />

*   **TLD Results:** A false positive is tracked for the front wheel when the real one disappears behind an obstacle. Then both trackers point to the rear wheel 

<img width="700" height="325" alt="image" src="https://github.com/user-attachments/assets/35c1bb68-abe5-46af-8226-6f5a0b14f0da" />
<img width="672" height="314" alt="image" src="https://github.com/user-attachments/assets/b69c9a38-f63c-4656-8a12-cb22f9742549" />
<img width="693" height="331" alt="image" src="https://github.com/user-attachments/assets/9b129e42-d8d8-4e4f-824a-4aec3617a668" />
<img width="697" height="324" alt="image" src="https://github.com/user-attachments/assets/1725c84e-3870-4b59-81a7-b08903ce55d8" />
<img width="690" height="334" alt="image" src="https://github.com/user-attachments/assets/edf47401-5ab9-4462-b3f7-7aaee9c2d5fb" />
<img width="698" height="329" alt="image" src="https://github.com/user-attachments/assets/54df9cc7-4204-413c-9202-9dcc2260c392" />

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
