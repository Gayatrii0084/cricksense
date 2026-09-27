# CrickSense — General Category Dataset Summary & Analysis

**Analysis Scope:** `General` Category Only  
**Source Directory:** `ai/data/landmarks/General/`  
**Generated Date:** September 2026  
**Status:** Audit & Analysis Complete — No source CSVs, videos, or CrickSense code modified

---

## 1. Executive Summary

This report delivers a rigorous, isolated quality analysis of the **General** category landmark dataset in CrickSense. The analysis uses exclusively the 10 existing General-category landmark CSV files generated during pose extraction and target-bowler tracking. No Under-15 (U15) or Under-19 (U19) data was included.

### Dataset Overview
* **Total General Videos:** 10
* **Total Video Frames:** 2,135
* **Total Detected Frames (Bowler Tracked):** 1,392
* **Total Lost Frames:** 743
* **Overall General Detection Percentage:** **65.20%**

---

## 2. Video-Wise Dataset Summary (Task 1)

The table below summarizes the frame counts, tracking status, and detection percentage for every video in the General category:

| Video ID | Category | Total Frames | Detected Frames | Lost Frames | Detection % | Quality Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **General1** | General | 76 | 76 | 0 | 100.00% | **GOOD** |
| **General2** | General | 177 | 170 | 7 | 96.05% | **GOOD** |
| **General3** | General | 146 | 127 | 19 | 86.99% | **ACCEPTABLE** |
| **S_v5** | General | 50 | 50 | 0 | 100.00% | **GOOD** |
| **s_v1** | General | 313 | 176 | 137 | 56.23% | **REVIEW** |
| **s_v2** | General | 558 | 381 | 177 | 68.28% | **REVIEW** |
| **s_v3** | General | 90 | 89 | 1 | 98.89% | **GOOD** |
| **s_v4** | General | 59 | 54 | 5 | 91.53% | **GOOD** |
| **s_v6** | General | 558 | 193 | 365 | 34.59% | **POOR** |
| **s_v7** | General | 108 | 76 | 32 | 70.37% | **REVIEW** |
| **TOTAL / OVERALL** | **General** | **2,135** | **1,392** | **743** | **65.20%** | — |

---

## 3. Video Quality Classification (Task 2)

Videos are classified according to predefined tracking quality thresholds:

### 🟢 GOOD Quality Tier (`Detection >= 90%`) — 5 Videos
High tracking continuity with minimal to zero frame loss. Highly suitable for initial model training and feature extraction.
* **General1**: 76 / 76 frames (100.00% detection, 0 lost frames)
* **S_v5**: 50 / 50 frames (100.00% detection, 0 lost frames)
* **s_v3**: 89 / 90 frames (98.89% detection, 1 lost frame)
* **General2**: 170 / 177 frames (96.05% detection, 7 lost frames, longest lost run = 3)
* **s_v4**: 54 / 59 frames (91.53% detection, 5 lost frames, longest lost run = 4)
* **Subtotal (GOOD):** 5 videos | 452 total frames | 439 detected frames (**97.12% aggregate detection**)

### 🟡 ACCEPTABLE Quality Tier (`75% <= Detection < 90%`) — 1 Video
Reliable tracking with minor, isolated dropouts that can be handled with linear interpolation or sequence windowing.
* **General3**: 127 / 146 frames (86.99% detection, 19 lost frames, longest lost run = 5)
* **Subtotal (ACCEPTABLE):** 1 video | 146 total frames | 127 detected frames (**86.99% aggregate detection**)

### 🟠 REVIEW Quality Tier (`50% <= Detection < 75%`) — 3 Videos
Noticeable frame losses and intermittent tracking drops due to occlusions, fast motion blur, or bowler moving out of frame bounds.
* **s_v7**: 76 / 108 frames (70.37% detection, 32 lost frames, longest lost run = 30)
* **s_v2**: 381 / 558 frames (68.28% detection, 177 lost frames, longest lost run = 44)
* **s_v1**: 176 / 313 frames (56.23% detection, 137 lost frames, longest lost run = 46)
* **Subtotal (REVIEW):** 3 videos | 979 total frames | 633 detected frames (**64.66% aggregate detection**)

### 🔴 POOR Quality Tier (`Detection < 50%`) — 1 Video
Severe tracking failure (>65% frame loss). The target bowler was lost for extended continuous sequences.
* **s_v6**: 193 / 558 frames (34.59% detection, 365 lost frames, longest lost run = 138)
* **Subtotal (POOR):** 1 video | 558 total frames | 193 detected frames (**34.59% aggregate detection**)

---

## 4. Recommended Usable General Videos for the First ML Experiment

To guarantee statistical integrity and avoid feeding fragmented landmark sequences into machine learning models, video selection is prioritized as follows:

### Primary Recommended Cohort (GOOD Videos Only):
1. **General1** (76 detected frames, 100.0%)
2. **S_v5** (50 detected frames, 100.0%)
3. **s_v3** (89 detected frames, 98.89%)
4. **General2** (170 detected frames, 96.05%)
5. **s_v4** (54 detected frames, 91.53%)

* **Primary Usable Total:** **5 videos | 439 high-fidelity detected frames**
* **Advantage:** Minimum contiguous sequence noise, zero multi-second dropouts, near-perfect anatomical landmark consistency.

### Extended Secondary Cohort (GOOD + ACCEPTABLE):
* Adding **General3** (127 detected frames, 86.99%) expands the training/testing corpus to **6 videos | 566 detected frames (94.65% combined detection rate)**.

### Excluded from Initial ML Baseline:
* `s_v1`, `s_v2`, `s_v6`, and `s_v7` should remain excluded from initial algorithm benchmarking until sequence interpolation or keyframe-only slicing (e.g., release-point window extraction) is applied.

---

## 5. Critical Architecture Distinctions (Task 4)

In sports analytics and computer vision pipelines, it is crucial not to conflate tracking metrics with model performance:

```
[Video Frame Stream]
         │
         ▼
1. Raw MediaPipe Pose Detection (Vision Layer)
   Detects whether ANY human keypoints exist in the camera field of view.
         │
         ▼
2. Target Bowler Tracking & Lock-on (Tracker Layer)
   Filters background players/umpires and tracks ONLY the bowling athlete.
   --> This is what yields the "Detection Percentage" reported above.
         │
         ▼
3. Machine Learning Classification (Analytics Layer)
   Evaluates bowling action legality, biomechanics, elbow extension, or style.
   --> NOT YET TRAINED. Does not correlate to raw tracking detection percentages.
```

1. **MediaPipe Pose Detection Percentage:**
   * Measures raw computer vision detection: whether MediaPipe can identify 33 body landmarks for any person present in the frame.
2. **Target Bowler Tracking Percentage:**
   * Measures CrickSense's tracker consistency: whether the primary bowler was locked onto, passed bounding box/velocity gating, and was retained without switching to other athletes (reported as `target_detected == 1`).
   * A frame with umpire detection but bowler occlusion is marked as lost (`target_detected == 0`).
3. **Machine Learning Classification Accuracy:**
   * Measures downstream model classification capability (e.g., Legal vs. Illegal bowling action, Spin vs. Fast bowling style).
   * **No ML algorithm has been trained yet.**
   * Tracking percentage (e.g., 96.05% on General2) is strictly a data availability metric, **NOT** an ML accuracy score.

---

## 6. Output Artifacts Generated

The following structured analysis files are available:
* **CSV:** `ai/data/analysis/general/general_dataset_summary.csv`
* **Markdown:** `ai/data/analysis/general/general_dataset_summary.md`
