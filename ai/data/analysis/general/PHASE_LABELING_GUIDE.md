# CrickSense — Cricket Bowling Phase Labeling Guide

**Target Scope:** Ground-Truth Action Segmentation for ST-GCN (Spatial-Temporal Graph Convolutional Networks)  
**Dataset:** General Category Landmark Dataset (`ai/data/landmarks/General/`)  
**Output Manifests:** 
- Frame-Level: `ai/data/analysis/general/general_labeling_manifest.csv`
- Sequence-Level: `ai/data/analysis/general/general_sequence_manifest.csv`

---

## 1. Overview and Purpose

In Spatial-Temporal Graph Convolutional Networks (ST-GCN), the body skeleton is represented as a spatial-temporal graph $G = (V, E)$, where $V$ represents the 33 MediaPipe body joints and $E$ represents physical anatomical bone connections across temporal frames.

To train or evaluate ST-GCN for bowling action phase recognition and biomechanical legality assessment, each frame or sequence must be annotated with objective, mutually exclusive ground-truth phase labels. 

This guide defines the 5 canonical phases of cricket bowling:
1. **`RUN_UP`**
2. **`GATHER`**
3. **`DELIVERY_STRIDE`**
4. **`RELEASE`**
5. **`FOLLOW_THROUGH`**

> [!IMPORTANT]
> **Zero Guessing Rule:** Annotators must label strictly based on verifiable visual landmarks. If a frame occurs during a tracking loss (`target_detected == 0`) or the athlete is not performing a bowling action, leave the label empty or flag for exclusion.

---

## 2. Anatomical Keypoint Reference (MediaPipe 33 Landmarks)

Key landmarks utilized to define phase boundaries:
* **Pelvis / Hips (Center of Mass):** Joint 23 (Left Hip), Joint 24 (Right Hip)
* **Knees & Ankles (Lower Extremities):** Joint 25/26 (Left/Right Knee), Joint 27/28 (Left/Right Ankle), Joint 29/30 (Heels), Joint 31/32 (Toes)
* **Shoulders & Spine (Trunk):** Joint 11 (Left Shoulder), Joint 12 (Right Shoulder)
* **Bowling Arm (Upper Extremity):** Joint 12/14/16 (Right Arm: Shoulder, Elbow, Wrist) or Joint 11/13/15 (Left Arm: Shoulder, Elbow, Wrist)

---

## 3. Objective Phase Definitions

```
Timeline:
[ RUN_UP ] ──► [ GATHER ] ──► [ DELIVERY_STRIDE ] ──► [ RELEASE ] ──► [ FOLLOW_THROUGH ]
          Takeoff         BFC                    FFC         Ball Leaves Hand
```

---

### Phase 1: `RUN_UP`

* **Anatomical Summary:** The approach phase where the bowler accelerates cyclically towards the bowling crease to build forward linear kinetic momentum.
* **Objective Start Event:**
  * The athlete initiates forward approach locomotion towards the stumps.
  * *Landmark Trigger:* Continuous forward horizontal displacement of pelvis center (`landmark_23`, `landmark_24`) with alternating knee flexion/extension stride cycles.
* **Objective End Event:**
  * The instant the bowler pushes off the turf to initiate the pre-delivery bound / jump.
  * *Landmark Trigger:* Final ground push-off toe separation of the takeoff leg prior to aerial ascent.
* **Key Visual Characteristics:**
  * Rhythmic reciprocal arm swing.
  * Progressive acceleration; trunk forward lean between 5° and 20°.
  * No overhead arm wind-up initiated yet.

---

### Phase 2: `GATHER` (Pre-Delivery Bound / Jump)

* **Anatomical Summary:** The transition phase between linear running and the rotational delivery sequence. The athlete bounds upward and realigns the body axis into the desired delivery orientation (side-on, front-on, or semi-open).
* **Objective Start Event:**
  * Takeoff into the pre-delivery bound (the moment both feet become airborne).
  * *Landmark Trigger:* Inversion of vertical velocity (center of mass height $y$-coordinate moves upward towards maximum elevation).
* **Objective End Event:**
  * **Back Foot Contact (BFC):** The exact frame where the back foot makes initial physical contact with the pitch surface.
  * *Landmark Trigger:* The back-foot ankle/heel (`landmark_27` or `landmark_28`) reaches ground level and its downward vertical velocity drops to zero.
* **Key Visual Characteristics:**
  * Both feet airborne during peak height.
  * Hands gather together near chest/chin height (`landmark_15` and `landmark_16` converge near `landmark_11`/`landmark_12`).
  * Trunk rotates into the bowler's specific alignment posture.

---

### Phase 3: `DELIVERY_STRIDE`

* **Anatomical Summary:** The loading and braking phase between initial back-foot impact and front-foot bracing. Ground reaction forces are absorbed and kinetic energy is channeled up the kinetic chain.
* **Objective Start Event:**
  * **Back Foot Contact (BFC):** The exact initial touchdown frame of the rear foot.
* **Objective End Event:**
  * **Front Foot Contact (FFC):** The exact frame where the front foot touches the ground surface and plants firmly to initiate the final delivery brace.
  * *Landmark Trigger:* Downward velocity of front heel/toe (`landmark_29`/`landmark_31` or `landmark_30`/`landmark_32`) reaches zero upon turf impact.
* **Key Visual Characteristics:**
  * Rear leg flexes under impact loading then extends.
  * Bowling arm breaks from the gather and executes its backward/downward circular arc into cocking position.
  * Non-bowling arm (lead arm) drives upward and forward toward the target.

---

### Phase 4: `RELEASE` (Delivery Arc & Ball Release Window)

* **Anatomical Summary:** The critical propulsion phase where the trunk flexes forward over the braced front leg and the bowling arm rotates overhead through the vertical release point.
* **Objective Start Event:**
  * **Front Foot Contact (FFC):** The frame of front foot touchdown/plant.
* **Objective End Event:**
  * **Ball Release Point + Terminal Arc Completion:** The exact frame where the ball departs the bowler's hand, extending until the bowling arm completes its upward vertical apex and descends past 45° forward.
  * *Landmark Trigger:* Bowling wrist (`landmark_16` or `landmark_15`) achieves maximum vertical height ($y$-minimum in normalized image coordinates) and begins downward trajectory.
* **Key Visual Characteristics:**
  * Front leg braces (knee joint angle stabilizes or extends).
  * Rapid shoulder axis counter-rotation and trunk flexion.
  * Bowling elbow angle undergoes dynamic loading (primary window evaluated for ICC 15° elbow extension rules).

---

### Phase 5: `FOLLOW_THROUGH`

* **Anatomical Summary:** The deceleration and recovery phase where residual momentum is safely dissipated across multiple strides after the ball has left the hand.
* **Objective Start Event:**
  * Immediately following the release window (bowling arm crosses downward past horizontal/chest level).
* **Objective End Event:**
  * The athlete completes deceleration strides, clears the pitch danger area, and returns to a controlled athletic fielding stance or stops moving.
  * *Landmark Trigger:* Horizontal kinetic velocity decays to baseline; trunk returns to upright balance.
* **Key Visual Characteristics:**
  * Bowling arm sweeps diagonally across the body toward the opposite hip.
  * Back leg swings through past the front leg to absorb forward momentum.
  * Eyes track the trajectory of the ball toward the batter.

---

## 4. Phase Boundary Summary Matrix

| Phase | Starts At (Frame Trigger) | Ends At (Frame Trigger) | Key Landmark Checks |
| :--- | :--- | :--- | :--- |
| **`RUN_UP`** | Athlete starts approach strides | Takeoff push into pre-delivery bound | Cyclic hip forward motion (`23`, `24`), alternating knees (`25`, `26`) |
| **`GATHER`** | Bound takeoff (feet leave turf) | Back Foot Contact (BFC) | Hands together at chest (`15`, `16` to `11`, `12`), peak jump height |
| **`DELIVERY_STRIDE`**| Back Foot Contact (BFC) | Front Foot Contact (FFC) | Rear foot grounded (`27`/`28`), bowling arm circular windup |
| **`RELEASE`** | Front Foot Contact (FFC) | Ball leaves hand / Arm apex pass | Bowling wrist (`16`/`15`) at vertical apex, front knee (`25`/`26`) brace |
| **`FOLLOW_THROUGH`** | Arm descends past chest level | Completion of deceleration strides | Arm cross-body sweep to opposite hip, velocity stabilization |

---

## 5. Instructions for Ground-Truth Annotators

1. **Frame Manifest (`general_labeling_manifest.csv`):**
   * Inspect each row sequentially using video playback synchronized with `frame_id` and `timestamp`.
   * Assign one of: `RUN_UP`, `GATHER`, `DELIVERY_STRIDE`, `RELEASE`, `FOLLOW_THROUGH`.
   * For non-action frames (e.g., player standing before run-up or walking back), leave empty or enter `OUT_OF_ACTION`.
2. **Sequence Manifest (`general_sequence_manifest.csv`):**
   * Update `phase_label` for identified continuous blocks.
   * Change `label_status` from `UNLABELED` to `LABELED` once verified by a qualified biomechanics / cricket analyst.
   * Do NOT use automated heuristics to fabricate labels for training.
