# CrickSense — General Category Manual Annotation Tool Guide

This document describes how to launch and operate the manual ground-truth bowling phase annotation tool for the General category dataset.

---

## 1. Quick Start

### Prerequisites
* Python 3.10+
* OpenCV (`cv2`)
* Pillow (`PIL`)
* Tkinter (standard with Python on Windows)

### Launching the Annotation Tool

From PowerShell at the project root (`cricksense-main`), run:

```powershell
python ai/data/analysis/general/annotate_general.py
```

To run the automated self-test verification suite (headless / non-GUI):

```powershell
python ai/data/analysis/general/annotate_general.py --test
```

---

## 2. Tool Architecture and Features

The annotation tool (`annotate_general.py`) connects the landmark tracking dataset to human biomechanics review:

```
[ai/data/landmarks/General/*.csv] ──┐
                                   ├──► [annotate_general.py] ──► [ai/data/analysis/general/general_annotations.csv]
[ai/data/raw/General/*.mp4]        ──┤        (GUI)
                                   │
[general_sequence_manifest.csv]   ──┘
```

### Key Capabilities
1. **Video Selection:** Dropdown selector populated with all 10 General videos (`General1`, `General2`, `General3`, `S_v5`, `s_v1`, `s_v2`, `s_v3`, `s_v4`, `s_v6`, `s_v7`).
2. **Dual-Mode Rendering:**
   * **Raw Video Mode:** Plays the actual video frames from `ai/data/raw/General/{video_id}.mp4` with MediaPipe skeleton tracking overlay.
   * **Skeleton Replay Mode:** If raw video files have not yet been copied to `ai/data/raw/General/`, the canvas automatically renders an interactive 33-joint skeleton replay on a virtual dark pitch canvas using the verified landmark CSVs.
3. **Frame-Accurate Navigation:**
   * Scrub bar with frame counter and millisecond timestamp.
   * Single frame forward/backward step (`< Prev`, `Next >`).
   * 10-frame jump (`<< -10`, `+10 >>`).
   * First/last frame jump (`|<<`, `>>|`).
   * Play/pause at normal speed (Spacebar).
4. **Sequence & Boundary Setting:**
   * Candidate sequences loaded directly from `general_sequence_manifest.csv`.
   * Set start frame (`[Set Curr (S)]` or key `S`).
   * Set end frame (`[Set Curr (E)]` or key `E`).
   * Active annotation window highlighted directly on the video canvas.
5. **Phase Selection:** Radio buttons to select exactly one of the five canonical phases.
6. **Annotation CRUD Lifecycle:**
   * **Save:** Validates boundary integrity and appends/updates `general_annotations.csv`.
   * **Edit:** Select an existing annotation from the list to reload, modify, and re-save.
   * **Delete:** Remove erroneous annotations with confirmation dialog.

---

## 3. The 5 Bowling Phase Definitions

All annotations must adhere strictly to the objective criteria defined in [`PHASE_LABELING_GUIDE.md`](file:///d:/BE%20project/CricketSense/cricksense-main/ai/data/analysis/general/PHASE_LABELING_GUIDE.md):

### 1. `RUN_UP`
* **Starts:** Athlete begins forward cyclic approach strides towards the bowling crease.
* **Ends:** Final push-off of the takeoff foot prior to the aerial jump.
* **Key Landmarks:** Progressive forward horizontal displacement of pelvis center (`landmark_23`, `landmark_24`), rhythmic alternating knee flexion/extension (`landmark_25`, `landmark_26`).

### 2. `GATHER` (Pre-Delivery Bound / Jump)
* **Starts:** Athlete's feet leave the turf into the aerial bound.
* **Ends:** **Back Foot Contact (BFC)** — the instant the rear foot impacts the pitch surface.
* **Key Landmarks:** Peak vertical elevation of center of mass; hands gather together near chest/chin height (`landmark_15` and `landmark_16` converge near `landmark_11`/`landmark_12`).

### 3. `DELIVERY_STRIDE`
* **Starts:** Back Foot Contact (BFC) touchdown.
* **Ends:** **Front Foot Contact (FFC)** — the instant the lead foot plants firmly on the turf.
* **Key Landmarks:** Weight shifts forward from rear leg to front leg; bowling arm breaks from gather into its backward circular wind-up arc.

### 4. `RELEASE` (Delivery Arc & Ball Release Window)
* **Starts:** Front Foot Contact (FFC) plant and trunk forward flexion.
* **Ends:** Ball Release Point (ball departs bowler's hand) through terminal arm apex pass.
* **Key Landmarks:** Front knee locks/braces; bowling wrist (`landmark_16` or `landmark_15`) reaches maximum vertical height above the head and begins downward trajectory.

### 5. `FOLLOW_THROUGH`
* **Starts:** Immediately following the release window (bowling arm crosses downward past torso level).
* **Ends:** Cessation of forward momentum and recovery of balanced upright athletic posture.
* **Key Landmarks:** Diagonal cross-body sweep of the bowling arm towards the opposite hip; trailing leg swings forward over the crease line to decelerate the athlete.

---

## 4. Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| **`Space`** | Play / Pause playback |
| **`Left Arrow`** | Move backward 1 frame |
| **`Right Arrow`** | Move forward 1 frame |
| **`Down Arrow`** | Move backward 10 frames |
| **`Up Arrow`** | Move forward 10 frames |
| **`S`** | Set current frame as **Start Frame** |
| **`E`** | Set current frame as **End Frame** |
| **`Ctrl + S`** | Save current annotation |

---

## 5. Validation Rules Enforced

Before any annotation is written to `general_annotations.csv`, the following rules are validated:
1. `video_id` must match one of the 10 General videos.
2. `sequence_id` must be present in `general_sequence_manifest.csv` for that video.
3. `start_frame` must be $\le$ `end_frame`.
4. Frame indices must fall within $1 \le \text{frame} \le \text{total\_frames}$.
5. `phase_label` must be one of the 5 allowed phases (`RUN_UP`, `GATHER`, `DELIVERY_STRIDE`, `RELEASE`, `FOLLOW_THROUGH`).

---

## 6. Output Format

Saved annotations reside in [`ai/data/analysis/general/general_annotations.csv`](file:///d:/BE%20project/CricketSense/cricksense-main/ai/data/analysis/general/general_annotations.csv):

```csv
video_id,sequence_id,start_frame,end_frame,phase_label,annotator,notes
General1,General1_seq_01,1,28,RUN_UP,Annotator_1,Clean approach
General1,General1_seq_01,29,42,GATHER,Annotator_1,Takeoff to BFC
```
