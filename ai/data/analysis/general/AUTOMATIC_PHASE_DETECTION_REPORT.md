# CrickSense â€” Automatic Cricket Bowling Phase Detection Technical Report

**Project:** CrickSense Biomechanics Analytics
**Category:** General Category Video Landmark Dataset (`ai/data/landmarks/General/`)
**Generated Date:** October 2026
**Pipeline Source:** `ai/data/analysis/general/auto_phase_detector.py`
**Output Files:**
- Annotations: [`general_auto_annotations.csv`](file:///ai/data/analysis/general/general_auto_annotations.csv)
- Summary CSV: [`automatic_phase_detection_report.csv`](file:///ai/data/analysis/general/automatic_phase_detection_report.csv)

---

## 1. Executive Summary

This report documents the automatic bowling phase segmentation engine developed for CrickSense. The system extracts kinematic and anatomical motion characteristics from 33 MediaPipe pose keypoints to automatically segment sequences into the five canonical bowling phases without requiring manual frame-by-frame labeling:

1. **`RUN_UP`**
2. **`GATHER`**
3. **`DELIVERY_STRIDE`**
4. **`RELEASE`**
5. **`FOLLOW_THROUGH`**

### Pipeline Metrics Overview:
* **Candidate Sequences Evaluated:** 73
* **Successfully Segmented Sequences:** **33** (165 individual phase intervals)
* **Skipped Tracking Fragments (length < 5 frames):** 40
* **Overall Average Boundary Confidence:** **0.73**
* **Total Low-Confidence Boundaries Flagged:** **64**

---

## 2. How the Automatic Phase Segmentation Algorithm Works

The algorithm implements a four-stage biomechanical event detection pipeline derived from physical action characteristics:

```
[Raw MediaPipe CSV]
         â”‚
         â–¼
[Stage 1: Preprocessing & Data Cleaning]
 â”œâ”€â”€ Visibility Gating (cutoff = 0.25)
 â”œâ”€â”€ Temporal Linear Gap Interpolation (max gap = 5)
 â””â”€â”€ Moving-Average Temporal Smoothing (window = 3)
         â”‚
         â–¼
[Stage 2: Kinematic Feature Extraction]
 â”œâ”€â”€ Torso Length Scale Normalization (L_torso)
 â”œâ”€â”€ Dynamic Bowling Arm Detection (360Â° rotational sweep & velocity burst)
 â”œâ”€â”€ Bilateral Leg Extensions (Ankle-to-Hip vertical displacement)
 â”œâ”€â”€ Hand Convergence Distance (D_hands)
 â””â”€â”€ Joint Angles (Lead Knee brace angle, Bowling Elbow extension angle)
         â”‚
         â–¼
[Stage 3: Biomechanical Event Anchoring]
 â”œâ”€â”€ Event 4: Release Point & Overhead Delivery Arc (B4)
 â”œâ”€â”€ Event 3: Front Foot Contact / FFC Plant (B3)
 â”œâ”€â”€ Event 2: Back Foot Contact / BFC Touchdown (B2)
 â””â”€â”€ Event 1: Pre-Delivery Bound Takeoff / Gather (B1)
         â”‚
         â–¼
[Stage 4: Strictly Monotonic Phase Partitioning & Confidence Scoring]
 â”œâ”€â”€ RUN_UP:          [start_frame, B1]
 â”œâ”€â”€ GATHER:          [B1 + 1, B2]
 â”œâ”€â”€ DELIVERY_STRIDE: [B2 + 1, B3]
 â”œâ”€â”€ RELEASE:         [B3 + 1, B4]
 â””â”€â”€ FOLLOW_THROUGH:  [B4 + 1, end_frame]
```

### Strict Partitioning Guarantees:
- **Full Sequence Coverage:** From first sequence frame to last sequence frame.
- **Zero Gaps:** Each subsequent phase starts exactly at the next consecutive frame ($end + 1$).
- **Zero Overlaps:** Every frame belongs to exactly one phase.
- **Chronological Monotonicity:** Phases strictly follow canonical order.
- **All 5 Labels Present:** Valid non-empty start and end frames for every phase.

---

## 3. Landmark Features & Biomechanical Triggers Used

| Phase | Objective Event Anchors | Landmark Features Used | Kinematic Trigger |
| :--- | :--- | :--- | :--- |
| **`RUN_UP`** | Approach forward cyclic strides | Pelvis center (`lm_23`, `lm_24`), Ankles (`lm_27`, `lm_28`) | Linear forward horizontal displacement with alternating cyclic leg kinematics |
| **`GATHER`** | Pre-delivery bound takeoff $	o$ Back Foot Contact (BFC) | Hands (`lm_15`, `lm_16`), Rear Ankle (`lm_28`/`lm_27`), Pelvis vertical velocity | Upward vertical velocity inversion ($v_{pelvis, y} < 0$), hands converge near torso ($D_{hands} < 0.8$), terminates upon rear foot turf impact |
| **`DELIVERY_STRIDE`** | BFC $	o$ Front Foot Contact (FFC) plant | Lead Ankle (`lm_27`/`lm_28`), Lead Knee (`lm_25`/`lm_26`), Bowling Arm wind-up | Lead leg reaches maximum extension relative to hip; terminates upon lead foot turf plant |
| **`RELEASE`** | FFC plant $	o$ Ball release window | Bowling Wrist (`lm_16`/`lm_15`), Bowling Shoulder (`lm_12`/`lm_11`), Lead Knee brace | Bowling wrist passes overhead vertical apex ($Y_{wrist}^{rel}$ local minimum) and whips downward; lead knee locks/braces |
| **`FOLLOW_THROUGH`** | Ball release $	o$ Deceleration & athletic recovery | Bowling Wrist, Trailing Leg, Trunk forward tilt | Bowling arm sweeps across torso towards opposite hip; forward kinetic velocity decays |

---

## 4. Confidence Score Calculation & Quality Assurance (TASK 6 & 10)

For each detected phase and sequence, a confidence score $C \in [0.0, 1.0]$ is computed based on three weighted factors:

1. **Tracking Quality & Visibility ($W = 0.35$):** Average MediaPipe detection visibility across the sequence window.
2. **Biomechanical Salience ($W = 0.4$):** The sharpness and distinctiveness of the physical signals (overhead apex peak prominence, foot plant deceleration, hand convergence).
3. **Sequence Duration Factor ($W = 0.25$):** Temporal duration sufficiency for complete bowling biomechanics ($T \ge 25$ frames optimal).

### Confidence Tiers:
* ðŸŸ¢ **High Confidence ($C \ge 0.85$):** High tracking continuity with sharp, unambiguous biomechanical markers.
* ðŸŸ¡ **Medium Confidence ($0.7 \le C < 0.85$):** Solid action signature with minor noise or moderate landmark visibility.
* ðŸ”´ **Low Confidence ($C < 0.7$):** Compressed sequence, partial tracking loss, or ambiguous event peak; flagged for human quick verification.

> [!IMPORTANT]
> **Confidence Score $
eq$ Model Accuracy:**
> Confidence measures the internal signal-to-noise ratio and biomechanical salience of the detected landmark events. It is **NOT** a measure of model ground-truth accuracy.

---

## 5. Critical Distinction: `AUTO` vs `VERIFIED` Labels (TASK 9)

To ensure scientific integrity in sports analytics and prevent algorithmic bias from polluting future model benchmarks, CrickSense maintains a strict separation:

| Attribute | `AUTO_SUGGESTED` (`AUTO`) | `VERIFIED` (`VERIFIED`) |
| :--- | :--- | :--- |
| **Generation Method** | Heuristic kinematic event detection | Human expert visual inspection & confirmation |
| **Storage Location** | `general_auto_annotations.csv` | `general_annotations.csv` |
| **Ground-Truth Status** | **Candidate suggestions only** â€” NOT ground truth | **True Ground Truth** for model evaluation |
| **Suitability for ST-GCN Benchmarking** | Preliminary training supervision / pre-training | Standard test set benchmarking, accuracy, F1-score |

---

## 6. Detailed Sequence-by-Sequence Segmentation Results

The table below lists the segmentation results for all processed sequences in the General dataset:

| Video ID | Sequence ID | Total Frames | Arm | RUN_UP | GATHER | STRIDE | RELEASE | FOLLOW_THRU | Avg Conf | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **General1** | `General1_seq_01` | 76 | RIGHT | 1-19 | 20-33 | 34-44 | 45-47 | 48-76 | **0.90** | GOOD |
| **General2** | `General2_seq_01` | 94 | RIGHT | 1-27 | 28-38 | 39-47 | 48-50 | 51-94 | **0.94** | GOOD |
| **General2** | `General2_seq_03` | 40 | RIGHT | 101-104 | 105-112 | 113-123 | 124-135 | 136-140 | **0.89** | GOOD |
| **General2** | `General2_seq_04` | 17 | RIGHT | 144-145 | 146-150 | 151-152 | 153-157 | 158-160 | **0.85** | GOOD |
| **General2** | `General2_seq_05` | 15 | LEFT | 163-163 | 164-164 | 165-165 | 166-167 | 168-177 | **0.82** | GOOD |
| **General3** | `General3_seq_01` | 82 | LEFT | 1-13 | 14-27 | 28-28 | 29-30 | 31-82 | **0.92** | GOOD |
| **General3** | `General3_seq_02` | 11 | LEFT | 84-85 | 86-88 | 89-90 | 91-92 | 93-94 | **0.54** | REVIEW |
| **General3** | `General3_seq_04` | 23 | LEFT | 101-115 | 116-116 | 117-119 | 120-122 | 123-123 | **0.82** | GOOD |
| **S_v5** | `S_v5_seq_01` | 50 | RIGHT | 1-11 | 12-17 | 18-29 | 30-39 | 40-50 | **0.94** | GOOD |
| **s_v1** | `s_v1_seq_01` | 170 | RIGHT | 1-130 | 131-134 | 135-144 | 145-146 | 147-170 | **0.83** | GOOD |
| **s_v2** | `s_v2_seq_01` | 126 | RIGHT | 1-74 | 75-88 | 89-99 | 100-101 | 102-126 | **0.86** | GOOD |
| **s_v2** | `s_v2_seq_02` | 76 | RIGHT | 130-144 | 145-147 | 148-159 | 160-168 | 169-205 | **0.89** | GOOD |
| **s_v2** | `s_v2_seq_03` | 108 | RIGHT | 214-242 | 243-244 | 245-254 | 255-256 | 257-321 | **0.87** | GOOD |
| **s_v2** | `s_v2_seq_09` | 12 | RIGHT | 354-354 | 355-355 | 356-356 | 357-358 | 359-365 | **0.60** | REVIEW |
| **s_v2** | `s_v2_seq_10` | 10 | LEFT | 368-370 | 371-372 | 373-373 | 374-376 | 377-377 | **0.49** | REVIEW |
| **s_v2** | `s_v2_seq_12` | 5 | LEFT | 394-394 | 395-395 | 396-396 | 397-397 | 398-398 | **0.30** | REVIEW |
| **s_v2** | `s_v2_seq_17` | 8 | LEFT | 434-434 | 435-435 | 436-436 | 437-437 | 438-441 | **0.39** | REVIEW |
| **s_v2** | `s_v2_seq_19` | 7 | RIGHT | 455-455 | 456-457 | 458-458 | 459-460 | 461-461 | **0.36** | REVIEW |
| **s_v3** | `s_v3_seq_01` | 12 | LEFT | 1-1 | 2-2 | 3-5 | 6-9 | 10-12 | **0.61** | REVIEW |
| **s_v3** | `s_v3_seq_02` | 77 | RIGHT | 14-50 | 51-60 | 61-70 | 71-72 | 73-90 | **0.90** | GOOD |
| **s_v4** | `s_v4_seq_01` | 5 | RIGHT | 5-5 | 6-6 | 7-7 | 8-8 | 9-9 | **0.30** | REVIEW |
| **s_v4** | `s_v4_seq_02` | 49 | RIGHT | 11-14 | 15-20 | 21-32 | 33-41 | 42-59 | **0.92** | GOOD |
| **s_v6** | `s_v6_seq_06` | 19 | RIGHT | 158-158 | 159-162 | 163-165 | 166-170 | 171-176 | **0.85** | GOOD |
| **s_v6** | `s_v6_seq_08` | 9 | LEFT | 180-180 | 181-181 | 182-182 | 183-184 | 185-188 | **0.44** | REVIEW |
| **s_v6** | `s_v6_seq_13` | 10 | RIGHT | 276-278 | 279-281 | 282-283 | 284-284 | 285-285 | **0.55** | REVIEW |
| **s_v6** | `s_v6_seq_14` | 34 | LEFT | 288-298 | 299-299 | 300-305 | 306-309 | 310-321 | **0.90** | GOOD |
| **s_v6** | `s_v6_seq_15` | 19 | RIGHT | 324-330 | 331-333 | 334-337 | 338-340 | 341-342 | **0.82** | GOOD |
| **s_v6** | `s_v6_seq_17` | 16 | LEFT | 368-370 | 371-374 | 375-378 | 379-381 | 382-383 | **0.84** | GOOD |
| **s_v6** | `s_v6_seq_18` | 11 | LEFT | 390-394 | 395-397 | 398-398 | 399-399 | 400-400 | **0.59** | REVIEW |
| **s_v6** | `s_v6_seq_21` | 43 | LEFT | 416-416 | 417-417 | 418-418 | 419-427 | 428-458 | **0.83** | GOOD |
| **s_v6** | `s_v6_seq_22` | 7 | LEFT | 460-460 | 461-462 | 463-463 | 464-465 | 466-466 | **0.37** | REVIEW |
| **s_v7** | `s_v7_seq_01` | 43 | RIGHT | 1-24 | 25-32 | 33-39 | 40-40 | 41-43 | **0.90** | GOOD |
| **s_v7** | `s_v7_seq_02` | 33 | RIGHT | 46-52 | 53-61 | 62-70 | 71-74 | 75-78 | **0.92** | GOOD |

---

## 7. Analysis of Skipped Sequences (Tracking Artifacts < 5 Frames)

A total of **40 sequences** in `general_sequence_manifest.csv` had a duration of less than 5 frames (some only 1 or 2 frames). By mathematical definition, 5 mutually exclusive, non-empty phase intervals require a sequence length of at least 5 frames ($T \ge 5$).

These segments correspond to transient tracking flicker where an umpire or background player briefly intersected the bounding box. They are properly documented below rather than fabricated:

| Video ID | Sequence ID | Frames | Status | Reason |
| :--- | :--- | :---: | :---: | :--- |
| General2 | `General2_seq_02` | 4 | SKIPPED_TOO_SHORT | Sequence length (4 frames) < minimum 5 required for 5 distinct phases. |
| General3 | `General3_seq_03` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| General3 | `General3_seq_05` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| General3 | `General3_seq_06` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| General3 | `General3_seq_07` | 4 | SKIPPED_TOO_SHORT | Sequence length (4 frames) < minimum 5 required for 5 distinct phases. |
| General3 | `General3_seq_08` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v1 | `s_v1_seq_02` | 3 | SKIPPED_TOO_SHORT | Sequence length (3 frames) < minimum 5 required for 5 distinct phases. |
| s_v1 | `s_v1_seq_03` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v1 | `s_v1_seq_04` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v1 | `s_v1_seq_05` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_04` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_05` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_06` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_07` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_08` | 4 | SKIPPED_TOO_SHORT | Sequence length (4 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_11` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_13` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_14` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_15` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_16` | 4 | SKIPPED_TOO_SHORT | Sequence length (4 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_18` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_20` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_21` | 3 | SKIPPED_TOO_SHORT | Sequence length (3 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_22` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v2 | `s_v2_seq_23` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_01` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_02` | 3 | SKIPPED_TOO_SHORT | Sequence length (3 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_03` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_04` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_05` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_07` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_09` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_10` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_11` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_12` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_16` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_19` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_20` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_23` | 1 | SKIPPED_TOO_SHORT | Sequence length (1 frames) < minimum 5 required for 5 distinct phases. |
| s_v6 | `s_v6_seq_24` | 2 | SKIPPED_TOO_SHORT | Sequence length (2 frames) < minimum 5 required for 5 distinct phases. |

---

## 8. Downstream Integration & Future ST-GCN Evaluation

With automatically generated phase annotations in [`general_auto_annotations.csv`](file:///ai/data/analysis/general/general_auto_annotations.csv), researchers can:
1. **Accelerate Annotation:** Review sequences in `annotate_general.py` using boundary shifting shortcuts without manual frame tagging.
2. **Train Spatial-Temporal Models:** Extract windowed graphs $G = (V, E)$ corresponding to specific phases (e.g., evaluating elbow extension strictly within the `RELEASE` window).
3. **Rigorous ML Evaluation:** Once human verification is complete, evaluate ST-GCN classification accuracy, precision, recall, confusion matrix, and temporal boundary intersection-over-union (IoU).
