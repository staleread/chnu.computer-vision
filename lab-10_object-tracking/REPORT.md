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

**TLD Results:** A false positive is tracked for the front wheel when the real one disappears behind an obstacle. Then both trackers point to the rear wheel 

The tracking is good for some starting frames
<img width="700" height="325" alt="image" src="https://github.com/user-attachments/assets/35c1bb68-abe5-46af-8226-6f5a0b14f0da" />

But the front wheel tracker get crazy at some point and confuses the front wheel from the rear one
<img width="672" height="314" alt="image" src="https://github.com/user-attachments/assets/b69c9a38-f63c-4656-8a12-cb22f9742549" />

When the rear wheel disappears, both trackers state the target objects are not present (false for the front wheel)
<img width="693" height="331" alt="image" src="https://github.com/user-attachments/assets/9b129e42-d8d8-4e4f-824a-4aec3617a668" />

But then the front wheel tracker manages to recover
<img width="697" height="324" alt="image" src="https://github.com/user-attachments/assets/1725c84e-3870-4b59-81a7-b08903ce55d8" />

The rear wheel tracker recovers with ease
<img width="690" height="334" alt="image" src="https://github.com/user-attachments/assets/edf47401-5ab9-4462-b3f7-7aaee9c2d5fb" />

The wheels in perspective are noticed but the rectangles are slightly too small
<img width="698" height="329" alt="image" src="https://github.com/user-attachments/assets/54df9cc7-4204-413c-9202-9dcc2260c392" />

## Experiment #4: Waiter's Tip & Glass (data/glass.gif)
*   **Tracking Object:** Glass.
*   **Scenario:** Static object position but dynamic background and significant lighting changes.
<img width="694" height="323" alt="image" src="https://github.com/user-attachments/assets/5f851426-6ec1-4467-9f40-1706d8e3a615" />


**MIL Results:** Does track the object correctly no matter the lightning changes
<img width="680" height="318" alt="image" src="https://github.com/user-attachments/assets/055a9030-a5f4-41f4-8038-5a2cf0905038" />
<img width="697" height="325" alt="image" src="https://github.com/user-attachments/assets/9da01b66-a618-4a48-aabb-0334757f7268" />


*   **TLD Results:** For a few frame the tracker tries his best at keeping the object followed (with slight rectagle shifts though).
<img width="705" height="334" alt="image" src="https://github.com/user-attachments/assets/ebce7d76-3b38-4f6c-a87b-61681b46d6d9" />

But as the background changes significantly as well as the lightning, the tracker starts spamming false positive
<img width="681" height="332" alt="image" src="https://github.com/user-attachments/assets/1df2ab8a-2ca1-414a-bf7e-72d4f71b2f02" />

## Tracker comparison
*   **Observed differences in algorithm performance:**
    *   **MIL** consistently handles occlusions by "guessing" the next position (inertia). While this prevents jumping to other objects, it can lead to poor recovery if the object's path changes significantly during the occlusion. It is remarkably stable under illumination shifts (Experiment 4).
    *   **TLD** is more reactive; it correctly identifies object loss behind obstacles and can re-acquire the target immediately upon reappearance. However, it is prone to confusing similar-looking objects (e.g., front vs. rear wheel in Experiment 3) and can be distracted by background complexity under dynamic lighting.
*   **Strengths and weaknesses:**
    *   **MIL Strengths:** Robust to partial occlusion and lighting changes; maintains identity of multiple objects by sticking to local searches.
    *   **MIL Weaknesses:** "Inertia" can lead to tracking the background during full occlusion; recovery after long occlusions is inconsistent.
    *   **TLD Strengths:** Excellent at re-detecting objects after full disappearance; handles perspective changes well if the background is distinct.
    *   **TLD Weaknesses:** Susceptible to "identity swaps" when multiple similar objects are present; bounding boxes can be slightly inaccurate in size during perspective shifts.

## Tracking Duration Study
*   **Stable Scenarios (e.g., glass.gif):** MIL maintained a perfect track through significant lighting changes. TLD was stable initially but eventually failed due to background noise and illumination shifts.
*   **Dynamic Scenarios (e.g., drifting.mp4):** 
    *   **TLD** demonstrated high agility, re-acquiring the wheel immediately after it cleared the obstacle. It also successfully handled the side-to-rear perspective change without false positives.
    *   **MIL** tracked successfully through the perspective change but showed poorer recovery for the front wheel compared to the rear wheel after occlusion.
*   **Drift Analysis:** MIL's drift is characterized by "floating" during occlusions. TLD's drift is more "jumpy," manifesting as sudden shifts to similar objects or slightly undersized bounding boxes during rotation.

## Multi-Object Tracking
*   **Demonstration:** Simultaneous tracking of front and rear wheels in "drifting.mp4".
*   **Observed Behavior:**
    *   **MIL:** Showed good independence; even when the front wheel was obscured, the tracker did not "jump" to the rear wheel, instead continuing its predicted path. Recovery was successful for the rear wheel but struggled for the front.
    *   **TLD:** Suffered from identity confusion. The front wheel tracker incorrectly locked onto the rear wheel when the front one was obscured. Interestingly, TLD correctly reported "not present" for both targets when only the rear wheel was actually obscured (a false negative for the visible front wheel).
*   **Performance Impact:** Processing time scales linearly with the number of ROIs. The sequential update of trackers in the current implementation makes it less suitable for high-speed tracking of many objects simultaneously.
