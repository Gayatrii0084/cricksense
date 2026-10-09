"""
=============================================================================
CrickSense -- Manual Ground-Truth Annotation Tool (General Category)
=============================================================================
File:
    ai/data/analysis/general/annotate_general.py

Purpose:
    Interactive GUI tool for manual frame-accurate ground-truth phase annotation
    of cricket bowling sequences in the General category dataset.

Phases Supported:
    1. RUN_UP
    2. GATHER
    3. DELIVERY_STRIDE
    4. RELEASE
    5. FOLLOW_THROUGH

Data Sources:
    - Raw Videos (if present): ai/data/raw/General/{video_id}.mp4
    - Landmark Coordinates:   ai/data/landmarks/General/{video_id}_landmarks.csv
    - Sequence Manifest:      ai/data/analysis/general/general_sequence_manifest.csv
    - Phase Guidelines:       ai/data/analysis/general/PHASE_LABELING_GUIDE.md

Output:
    - Saved Annotations:      ai/data/analysis/general/general_annotations.csv

Usage:
    GUI Mode:
        python ai/data/analysis/general/annotate_general.py

    Self-Test / Automated Verification Mode:
        python ai/data/analysis/general/annotate_general.py --test

Author: CrickSense Team
Date:   September 2026
=============================================================================
"""

import os
import sys
import csv
import cv2
import math
import argparse
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox as msgbox
from PIL import Image, ImageTk
from typing import Dict, List, Optional, Tuple, Any

# MediaPipe 33-landmark skeleton connection pairs
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

ALLOWED_PHASES = [
    "RUN_UP",
    "GATHER",
    "DELIVERY_STRIDE",
    "RELEASE",
    "FOLLOW_THROUGH"
]

ANALYSIS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(ANALYSIS_DIR, "..", ".."))
RAW_DIR = os.path.join(DATA_DIR, "raw", "General")
LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks", "General")
SEQUENCE_MANIFEST_PATH = os.path.join(ANALYSIS_DIR, "general_sequence_manifest.csv")
LABELING_MANIFEST_PATH = os.path.join(ANALYSIS_DIR, "general_labeling_manifest.csv")
ANNOTATIONS_CSV_PATH = os.path.join(ANALYSIS_DIR, "general_annotations.csv")
AUTO_ANNOTATIONS_CSV_PATH = os.path.join(ANALYSIS_DIR, "general_auto_annotations.csv")

ANNOTATION_COLUMNS = [
    "video_id",
    "sequence_id",
    "start_frame",
    "end_frame",
    "phase_label",
    "annotator",
    "notes"
]


# =============================================================================
# DATA MANAGER & VALIDATION LOGIC
# =============================================================================

class AnnotationDataManager:
    """Manages reading, validation, and storage of bowling phase annotations."""

    def __init__(self,
                 landmarks_dir: str = LANDMARKS_DIR,
                 raw_dir: str = RAW_DIR,
                 seq_manifest_path: str = SEQUENCE_MANIFEST_PATH,
                 annotations_path: str = ANNOTATIONS_CSV_PATH,
                 auto_annotations_path: str = AUTO_ANNOTATIONS_CSV_PATH):
        self.landmarks_dir = landmarks_dir
        self.raw_dir = raw_dir
        self.seq_manifest_path = seq_manifest_path
        self.annotations_path = annotations_path
        self.auto_annotations_path = auto_annotations_path

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.annotations_path), exist_ok=True)

        self.videos: List[str] = []
        self.video_metadata: Dict[str, Dict[str, Any]] = {}
        self.sequences: Dict[str, List[Dict[str, Any]]] = {} # video_id -> list of sequences
        self.annotations: List[Dict[str, Any]] = []
        self.auto_annotations: Dict[str, List[Dict[str, Any]]] = {}

        self.discover_data()
        self.load_sequences()
        self.load_annotations()
        self.load_auto_annotations()

    def discover_data(self) -> None:
        """Scan General landmark directory for all video CSV files."""
        if not os.path.exists(self.landmarks_dir):
            print(f"[WARN] Landmarks directory not found: {self.landmarks_dir}")
            return

        csv_files = sorted([f for f in os.listdir(self.landmarks_dir) if f.endswith(".csv")])
        self.videos = []
        self.video_metadata = {}

        for fname in csv_files:
            video_id = fname.replace("_landmarks.csv", "")
            fpath = os.path.join(self.landmarks_dir, fname)

            # Fast count total and detected frames
            total_frames = 0
            detected_frames = 0
            with open(fpath, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    if not row:
                        continue
                    total_frames += 1
                    if len(row) > 2 and row[2].strip() == "1":
                        detected_frames += 1

            raw_video_path = self.find_raw_video(video_id)

            self.videos.append(video_id)
            self.video_metadata[video_id] = {
                "csv_path": fpath,
                "raw_path": raw_video_path,
                "total_frames": total_frames,
                "detected_frames": detected_frames,
                "lost_frames": total_frames - detected_frames,
                "detection_pct": round((detected_frames / total_frames * 100) if total_frames > 0 else 0.0, 2)
            }

    def find_raw_video(self, video_id: str) -> Optional[str]:
        """Find video file with common extensions in raw directory."""
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            cand = os.path.join(self.raw_dir, f"{video_id}{ext}")
            if os.path.exists(cand):
                return cand
        return None

    def load_sequences(self) -> None:
        """Load candidate sequences from general_sequence_manifest.csv."""
        self.sequences = {vid: [] for vid in self.videos}
        if not os.path.exists(self.seq_manifest_path):
            print(f"[WARN] Sequence manifest not found: {self.seq_manifest_path}")
            return

        with open(self.seq_manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["video_id"].strip()
                if vid not in self.sequences:
                    self.sequences[vid] = []
                self.sequences[vid].append({
                    "sequence_id": row["sequence_id"].strip(),
                    "start_frame": int(row["start_frame"].strip()),
                    "end_frame": int(row["end_frame"].strip()),
                    "phase_label": row.get("phase_label", "").strip(),
                    "label_status": row.get("label_status", "UNLABELED").strip()
                })

    def load_annotations(self) -> None:
        """Load annotations from general_annotations.csv."""
        self.annotations = []
        if not os.path.exists(self.annotations_path):
            self._init_annotations_csv()
            return

        with open(self.annotations_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get("video_id"):
                    continue
                try:
                    conf_str = row.get("confidence", "").strip() if row.get("confidence") else ""
                    conf = float(conf_str) if conf_str else 1.0
                    source = row.get("label_source", "VERIFIED").strip() if row.get("label_source") else "VERIFIED"
                    self.annotations.append({
                        "video_id": row["video_id"].strip(),
                        "sequence_id": row["sequence_id"].strip(),
                        "start_frame": int(row["start_frame"].strip()),
                        "end_frame": int(row["end_frame"].strip()),
                        "phase_label": row["phase_label"].strip(),
                        "confidence": conf,
                        "label_source": source,
                        "annotator": row.get("annotator", "").strip(),
                        "notes": row.get("notes", "").strip()
                    })
                except (ValueError, KeyError) as e:
                    print(f"[WARN] Skipping corrupted annotation row: {row} ({e})")

    def load_auto_annotations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load automatically detected annotations from general_auto_annotations.csv indexed by sequence_id."""
        self.auto_annotations = {}
        if not os.path.exists(self.auto_annotations_path):
            return self.auto_annotations

        with open(self.auto_annotations_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get("sequence_id"):
                    continue
                sid = row["sequence_id"].strip()
                if sid not in self.auto_annotations:
                    self.auto_annotations[sid] = []
                try:
                    conf = float(row.get("confidence", "0.85").strip())
                except ValueError:
                    conf = 0.85

                self.auto_annotations[sid].append({
                    "video_id": row["video_id"].strip(),
                    "sequence_id": sid,
                    "start_frame": int(row["start_frame"].strip()),
                    "end_frame": int(row["end_frame"].strip()),
                    "phase_label": row["phase_label"].strip(),
                    "confidence": conf,
                    "label_source": row.get("label_source", "AUTO").strip(),
                    "notes": row.get("notes", "").strip()
                })
        return self.auto_annotations

    def _init_annotations_csv(self) -> None:
        """Create empty annotations CSV with standard headers."""
        with open(self.annotations_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ANNOTATION_COLUMNS)

    def validate_annotation(self,
                            video_id: str,
                            sequence_id: str,
                            start_frame: int,
                            end_frame: int,
                            phase_label: str) -> Tuple[bool, str]:
        """
        Validate all strict annotation constraints:
          1. video_id exists in General dataset
          2. sequence_id exists for this video
          3. start_frame <= end_frame
          4. frames are valid (1 <= frame <= total_frames)
          5. phase_label is one of the 5 allowed labels
        """
        if video_id not in self.videos:
            return False, f"Invalid video_id '{video_id}'. Must be one of: {', '.join(self.videos)}"

        vid_seqs = [s["sequence_id"] for s in self.sequences.get(video_id, [])]
        if sequence_id not in vid_seqs:
            return False, f"Invalid sequence_id '{sequence_id}' for video '{video_id}'. Must be in sequence manifest."

        if not isinstance(start_frame, int) or not isinstance(end_frame, int):
            return False, "start_frame and end_frame must be integers."

        if start_frame > end_frame:
            return False, f"start_frame ({start_frame}) cannot be greater than end_frame ({end_frame})."

        max_frames = self.video_metadata[video_id]["total_frames"]
        if start_frame < 1 or end_frame < 1:
            return False, f"Frame indices must be >= 1 (got start={start_frame}, end={end_frame})."

        if start_frame > max_frames or end_frame > max_frames:
            return False, f"Frame indices exceed total video frames ({max_frames})."

        if phase_label not in ALLOWED_PHASES:
            return False, f"Invalid phase_label '{phase_label}'. Must be one of: {', '.join(ALLOWED_PHASES)}"

        return True, "Valid"

    def get_sequence_info(self, video_id: str, sequence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve sequence manifest entry for a given video and sequence ID."""
        seqs = self.sequences.get(video_id, [])
        return next((s for s in seqs if s["sequence_id"] == sequence_id), None)

    def is_short_sequence(self, video_id: str, sequence_id: str) -> bool:
        """Return True if sequence length is less than 5 frames (short continuation segment)."""
        seq_meta = self.get_sequence_info(video_id, sequence_id)
        if not seq_meta:
            return False
        return (int(seq_meta["end_frame"]) - int(seq_meta["start_frame"]) + 1) < 5

    def validate_sequence_phases(self,
                                 video_id: str,
                                 sequence_id: str,
                                 phases_list: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate verified phase annotations for a sequence:
          For normal sequences (>= 5 frames):
            1. Exactly 5 canonical phases are present.
            2. Phase order strictly follows ALLOWED_PHASES.
            3. Frame ranges are valid integers with start <= end and within video limits.
            4. Zero gaps and zero overlaps between consecutive phases.
            5. Complete coverage: phases cover sequence from start_frame to end_frame.
          For short continuation sequences (< 5 frames):
            1. Exactly 1 applicable phase is assigned.
            2. Phase is one of ALLOWED_PHASES.
            3. Start and end frames are valid integers within video limits.
            4. Frame range matches sequence boundaries without inventing artificial phases.
        """
        seq_meta = self.get_sequence_info(video_id, sequence_id)
        if not seq_meta:
            return False, f"Sequence '{sequence_id}' not found for video '{video_id}' in sequence manifest."

        seq_start = int(seq_meta["start_frame"])
        seq_end = int(seq_meta["end_frame"])
        seq_len = seq_end - seq_start + 1
        max_frames = self.video_metadata.get(video_id, {}).get("total_frames", 999999)

        # Branch for short continuation sequences (< 5 frames)
        if seq_len < 5:
            if not phases_list or len(phases_list) != 1:
                count = len(phases_list) if phases_list else 0
                return False, f"Short continuation sequence ({seq_len} frames) requires exactly 1 assigned phase, got {count}."

            p = phases_list[0]
            lbl = p.get("phase_label", "").strip()
            if lbl not in ALLOWED_PHASES:
                return False, f"Invalid phase_label '{lbl}'. Must be one of: {', '.join(ALLOWED_PHASES)}"

            try:
                sf = int(p["start_frame"])
                ef = int(p["end_frame"])
            except (ValueError, KeyError, TypeError):
                return False, f"Phase {lbl} start_frame and end_frame must be integers."

            if sf < 1 or ef < 1:
                return False, f"Phase {lbl} frame numbers must be >= 1 (got {sf}-{ef})."
            if sf > ef:
                return False, f"Phase {lbl} start frame ({sf}) is greater than end frame ({ef})."
            if sf > max_frames or ef > max_frames:
                return False, f"Phase {lbl} frame range ({sf}-{ef}) exceeds total video frames ({max_frames})."
            if sf < seq_start or ef > seq_end:
                return False, f"Phase {lbl} range ({sf}-{ef}) outside sequence boundaries ({seq_start}-{seq_end})."
            if sf != seq_start or ef != seq_end:
                return False, f"Phase {lbl} range ({sf}-{ef}) must cover short sequence boundaries ({seq_start}-{seq_end})."

            return True, "Valid"

        # Normal sequence (>= 5 frames)
        if not phases_list or len(phases_list) != 5:
            count = len(phases_list) if phases_list else 0
            return False, f"Expected exactly 5 canonical phases, but got {count}."

        labels = [p.get("phase_label", "").strip() for p in phases_list]
        missing = [p for p in ALLOWED_PHASES if p not in labels]
        if missing:
            return False, f"Missing required phases: {', '.join(missing)}."

        if labels != ALLOWED_PHASES:
            return False, f"Phases must follow canonical order: {' -> '.join(ALLOWED_PHASES)}. Found: {' -> '.join(labels)}."

        parsed_ranges = []
        for p in phases_list:
            lbl = p.get("phase_label", "")
            try:
                sf = int(p["start_frame"])
                ef = int(p["end_frame"])
            except (ValueError, KeyError, TypeError):
                return False, f"Phase {lbl} start_frame and end_frame must be integers."

            if sf < 1 or ef < 1:
                return False, f"Phase {lbl} frame numbers must be >= 1 (got {sf}-{ef})."

            if sf > ef:
                return False, f"Phase {lbl} start frame ({sf}) is greater than end frame ({ef})."

            if sf > max_frames or ef > max_frames:
                return False, f"Phase {lbl} frame range ({sf}-{ef}) exceeds total video frames ({max_frames})."

            parsed_ranges.append((sf, ef, lbl))

        for i in range(4):
            cur_sf, cur_ef, cur_lbl = parsed_ranges[i]
            next_sf, next_ef, next_lbl = parsed_ranges[i + 1]

            if next_sf <= cur_ef:
                return False, f"Overlap detected: {cur_lbl} (ends at frame {cur_ef}) overlaps with {next_lbl} (starts at frame {next_sf})."

            if next_sf > cur_ef + 1:
                return False, f"Gap detected between {cur_lbl} (ends at frame {cur_ef}) and {next_lbl} (starts at frame {next_sf}). Frames {cur_ef + 1} to {next_sf - 1} are unassigned."

        first_sf = parsed_ranges[0][0]
        last_ef = parsed_ranges[4][1]

        if first_sf != seq_start:
            return False, f"Annotations do not cover sequence start. Sequence '{sequence_id}' starts at frame {seq_start}, but {parsed_ranges[0][2]} starts at frame {first_sf}."

        if last_ef != seq_end:
            return False, f"Annotations do not cover sequence end. Sequence '{sequence_id}' ends at frame {seq_end}, but {parsed_ranges[4][2]} ends at frame {last_ef}."

        return True, "Valid"

    def is_sequence_verified(self, video_id: str, sequence_id: str) -> bool:
        """Check if sequence has valid verified annotations covering the sequence."""
        seq_meta = self.get_sequence_info(video_id, sequence_id)
        if not seq_meta:
            return False

        seq_len = int(seq_meta["end_frame"]) - int(seq_meta["start_frame"]) + 1
        seq_anns = [
            a for a in self.annotations
            if a["video_id"] == video_id and a["sequence_id"] == sequence_id
        ]

        if seq_len < 5:
            if len(seq_anns) != 1:
                return False
            ok, _ = self.validate_sequence_phases(video_id, sequence_id, seq_anns)
            return ok
        else:
            if len(seq_anns) != 5:
                return False
            phase_order = {name: i for i, name in enumerate(ALLOWED_PHASES)}
            sorted_anns = sorted(seq_anns, key=lambda x: phase_order.get(x["phase_label"], 99))
            ok, _ = self.validate_sequence_phases(video_id, sequence_id, sorted_anns)
            return ok

    def save_annotation(self,
                        video_id: str,
                        sequence_id: str,
                        start_frame: int,
                        end_frame: int,
                        phase_label: str,
                        annotator: str = "",
                        notes: str = "",
                        confidence: float = 1.0,
                        label_source: str = "VERIFIED",
                        prev_phase_label: Optional[str] = None) -> Tuple[bool, str]:
        """Validate and append or update an annotation in general_annotations.csv."""
        is_valid, msg = self.validate_annotation(video_id, sequence_id, start_frame, end_frame, phase_label)
        if not is_valid:
            return False, msg

        # If short sequence, ensure bounds stay within sequence
        if self.is_short_sequence(video_id, sequence_id):
            seq_meta = self.get_sequence_info(video_id, sequence_id)
            if seq_meta:
                seq_start = int(seq_meta["start_frame"])
                seq_end = int(seq_meta["end_frame"])
                if start_frame < seq_start or end_frame > seq_end:
                    return False, f"Phase range ({start_frame}-{end_frame}) outside sequence boundaries ({seq_start}-{seq_end})."

        target_phase = prev_phase_label if prev_phase_label else phase_label

        # Check if updating an existing annotation for this sequence_id and phase_label or creating new
        updated = False
        for ann in self.annotations:
            if ann["video_id"] == video_id and ann["sequence_id"] == sequence_id and ann["phase_label"] == target_phase:
                ann["start_frame"] = start_frame
                ann["end_frame"] = end_frame
                ann["phase_label"] = phase_label
                ann["confidence"] = confidence
                ann["label_source"] = label_source
                ann["annotator"] = annotator
                ann["notes"] = notes
                updated = True
                break

        # If not found and prev_phase_label was None, check if there is only 1 annotation for this sequence
        if not updated and prev_phase_label is None:
            seq_anns = [a for a in self.annotations if a["video_id"] == video_id and a["sequence_id"] == sequence_id]
            if len(seq_anns) == 1:
                seq_anns[0]["start_frame"] = start_frame
                seq_anns[0]["end_frame"] = end_frame
                seq_anns[0]["phase_label"] = phase_label
                seq_anns[0]["confidence"] = confidence
                seq_anns[0]["label_source"] = label_source
                seq_anns[0]["annotator"] = annotator
                seq_anns[0]["notes"] = notes
                updated = True

        if not updated:
            self.annotations.append({
                "video_id": video_id,
                "sequence_id": sequence_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "phase_label": phase_label,
                "confidence": confidence,
                "label_source": label_source,
                "annotator": annotator,
                "notes": notes
            })

        self._flush_annotations_to_disk()
        action_verb = "Updated" if updated else "Saved"
        return True, f"{action_verb} annotation for sequence '{sequence_id}' ({phase_label}: {start_frame}-{end_frame})."

    def save_verified_phases(self,
                             video_id: str,
                             sequence_id: str,
                             phases_list: List[Dict[str, Any]],
                             annotator: str = "Annotator_1",
                             notes: str = "Verified by human review") -> Tuple[bool, str]:
        """Validate and save verified phase annotations for a sequence."""
        ok, val_msg = self.validate_sequence_phases(video_id, sequence_id, phases_list)
        if not ok:
            return False, val_msg

        # Remove existing annotations for this sequence to replace with verified set
        self.annotations = [
            a for a in self.annotations
            if not (a["video_id"] == video_id and a["sequence_id"] == sequence_id)
        ]

        phase_order = {name: i for i, name in enumerate(ALLOWED_PHASES)}
        sorted_phases = sorted(phases_list, key=lambda x: phase_order.get(x["phase_label"], 99))

        for p in sorted_phases:
            self.annotations.append({
                "video_id": video_id,
                "sequence_id": sequence_id,
                "start_frame": int(p["start_frame"]),
                "end_frame": int(p["end_frame"]),
                "phase_label": p["phase_label"],
                "annotator": annotator,
                "notes": notes
            })

        self._flush_annotations_to_disk()
        phase_count = len(sorted_phases)
        return True, f"Saved {phase_count} verified phase{'s' if phase_count > 1 else ''} for sequence '{sequence_id}'."

    def delete_annotation(self, video_id: str, sequence_id: str, phase_label: Optional[str] = None) -> Tuple[bool, str]:
        """Remove annotations matching video_id and sequence_id (and optionally phase_label)."""
        initial_count = len(self.annotations)
        if phase_label:
            self.annotations = [
                ann for ann in self.annotations
                if not (ann["video_id"] == video_id and ann["sequence_id"] == sequence_id and ann["phase_label"] == phase_label)
            ]
        else:
            self.annotations = [
                ann for ann in self.annotations
                if not (ann["video_id"] == video_id and ann["sequence_id"] == sequence_id)
            ]
        if len(self.annotations) < initial_count:
            self._flush_annotations_to_disk()
            return True, f"Deleted annotation for sequence '{sequence_id}'."
        return False, f"Annotation not found for '{sequence_id}'."

    def _flush_annotations_to_disk(self) -> None:
        """Write current annotations list to CSV file using standard schema."""
        with open(self.annotations_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for ann in self.annotations:
                writer.writerow(ann)

    def load_landmarks(self, video_id: str) -> Dict[int, Dict[str, Any]]:
        """Load landmark data for video_id indexed by frame_id."""
        if video_id not in self.video_metadata:
            return {}
        fpath = self.video_metadata[video_id]["csv_path"]
        frames_dict = {}

        with open(fpath, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                fid = int(row[1])
                target_detected = int(row[2])
                landmarks = []
                if target_detected == 1:
                    raw_vals = row[3:]
                    for i in range(33):
                        base = i * 4
                        if base + 3 < len(raw_vals) and raw_vals[base] != "":
                            try:
                                x = float(raw_vals[base])
                                y = float(raw_vals[base + 1])
                                z = float(raw_vals[base + 2])
                                vis = float(raw_vals[base + 3])
                                landmarks.append((x, y, z, vis))
                            except ValueError:
                                pass
                frames_dict[fid] = {
                    "target_detected": target_detected,
                    "landmarks": landmarks
                }
        return frames_dict


# =============================================================================
# TKINTER GRAPHICAL INTERFACE
# =============================================================================

class AnnotationApp:
    """Tkinter-based interactive GUI video and skeleton annotation tool."""

    def __init__(self, root: Any, manager: AnnotationDataManager):
        self.root = root
        self.manager = manager

        self.root.title("CrickSense — General Category Ground-Truth Annotation Tool")
        self.root.geometry("1300x850")
        self.root.minsize(1050, 700)

        # State variables
        self.current_video_id: str = self.manager.videos[0] if self.manager.videos else ""
        self.current_frame: int = 1
        self.total_frames: int = 1
        self.is_playing: bool = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.landmarks_data: Dict[int, Dict[str, Any]] = {}
        self.selected_seq_id: str = ""

        # Quick Verification State (TASK 5)
        self.auto_phases: Dict[str, List[Dict[str, Any]]] = self.manager.load_auto_annotations()
        self.current_quick_phases: List[Dict[str, Any]] = []
        self.active_boundary_idx: int = 0
        self.is_previewing_phases: bool = False

        # UI Styling Colors
        self.BG_DARK = "#181825"
        self.BG_CARD = "#212133"
        self.BG_INPUT = "#2e2e42"
        self.TEXT_MAIN = "#cdd6f4"
        self.TEXT_MUTED = "#8990b3"
        self.ACCENT_GREEN = "#a6e3a1"
        self.ACCENT_BLUE = "#89b4fa"
        self.ACCENT_RED = "#f38ba8"
        self.ACCENT_ORANGE = "#fab387"

        self.root.configure(bg=self.BG_DARK)

        self._build_ui()
        self._bind_keyboard_shortcuts()

        if self.current_video_id:
            self._load_video(self.current_video_id)

    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        # Setup modern dark style for ttk widgets
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=self.BG_CARD, foreground=self.TEXT_MAIN, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.BG_CARD, foreground=self.ACCENT_BLUE, font=("Segoe UI", 11, "bold"))
        style.configure("TButton", background=self.BG_INPUT, foreground=self.TEXT_MAIN, font=("Segoe UI", 9))
        style.configure("Treeview", background=self.BG_INPUT, foreground=self.TEXT_MAIN, fieldbackground=self.BG_INPUT, font=("Segoe UI", 9))
        style.map("Treeview", background=[("selected", self.ACCENT_BLUE)], foreground=[("selected", "#000000")])

        # Top Header Bar
        top_bar = tk.Frame(self.root, bg=self.BG_CARD, height=50)
        top_bar.pack(fill=tk.X, side=tk.TOP, padx=10, pady=(10, 5))

        title_lbl = tk.Label(top_bar, text="🏏 CrickSense — General Ground-Truth Phase Annotator",
                             font=("Segoe UI", 14, "bold"), fg=self.ACCENT_BLUE, bg=self.BG_CARD)
        title_lbl.pack(side=tk.LEFT, padx=15, pady=8)

        # Video Selector in Top Bar
        tk.Label(top_bar, text="Select Video:", font=("Segoe UI", 10, "bold"), fg=self.TEXT_MAIN, bg=self.BG_CARD).pack(side=tk.LEFT, padx=(20, 5))
        self.video_combo = ttk.Combobox(top_bar, values=self.manager.videos, state="readonly", width=16)
        if self.manager.videos:
            self.video_combo.set(self.manager.videos[0])
        self.video_combo.pack(side=tk.LEFT, padx=5)
        self.video_combo.bind("<<ComboboxSelected>>", self._on_video_selected)

        self.video_info_lbl = tk.Label(top_bar, text="", font=("Segoe UI", 9), fg=self.TEXT_MUTED, bg=self.BG_CARD)
        self.video_info_lbl.pack(side=tk.LEFT, padx=15)

        # Main Layout: 3-column split (Left: Sequences & Annotations, Center: Video Canvas, Right: Label Controls)
        content_frame = tk.Frame(self.root, bg=self.BG_DARK)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # LEFT PANEL: Sequences list & Saved Annotations
        left_panel = tk.Frame(content_frame, bg=self.BG_CARD, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        ttk.Label(left_panel, text="Candidate Sequences (Manifest)", style="Header.TLabel").pack(anchor=tk.W, padx=10, pady=(10, 2))

        # Sequence treeview
        self.seq_tree = ttk.Treeview(left_panel, columns=("seq", "frames", "status"), show="headings", height=8)
        self.seq_tree.heading("seq", text="Seq ID")
        self.seq_tree.heading("frames", text="Frames")
        self.seq_tree.heading("status", text="Status")
        self.seq_tree.column("seq", width=120)
        self.seq_tree.column("frames", width=80)
        self.seq_tree.column("status", width=90)
        self.seq_tree.pack(fill=tk.X, padx=10, pady=5)
        self.seq_tree.bind("<<TreeviewSelect>>", self._on_seq_selected)

        ttk.Label(left_panel, text="Saved Ground-Truth Annotations", style="Header.TLabel").pack(anchor=tk.W, padx=10, pady=(15, 2))

        # Annotation treeview
        self.ann_tree = ttk.Treeview(left_panel, columns=("phase", "frames", "notes"), show="headings", height=10)
        self.ann_tree.heading("phase", text="Phase")
        self.ann_tree.heading("frames", text="Range")
        self.ann_tree.heading("notes", text="Annotator")
        self.ann_tree.column("phase", width=110)
        self.ann_tree.column("frames", width=80)
        self.ann_tree.column("notes", width=90)
        self.ann_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.ann_tree.bind("<<TreeviewSelect>>", self._on_ann_selected)

        # CENTER PANEL: Video / Skeleton Canvas + Controls
        center_panel = tk.Frame(content_frame, bg=self.BG_CARD)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas = tk.Canvas(center_panel, bg="#0d0e15", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video Slider and Time Bar
        scrub_frame = tk.Frame(center_panel, bg=self.BG_CARD)
        scrub_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.frame_slider = ttk.Scale(scrub_frame, from_=1, to=100, orient=tk.HORIZONTAL, command=self._on_slider_moved)
        self.frame_slider.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 10))

        self.time_lbl = tk.Label(scrub_frame, text="Frame: 1 / 1 (0.000s)", font=("Consolas", 10, "bold"),
                                 fg=self.ACCENT_GREEN, bg=self.BG_CARD, width=22)
        self.time_lbl.pack(side=tk.RIGHT)

        # Playback Controls
        ctrl_frame = tk.Frame(center_panel, bg=self.BG_CARD)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Button(ctrl_frame, text="|<<", width=5, bg=self.BG_INPUT, fg=self.TEXT_MAIN, command=self._jump_first).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text="<< -10", width=6, bg=self.BG_INPUT, fg=self.TEXT_MAIN, command=lambda: self._step_frame(-10)).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text="< Prev", width=6, bg=self.BG_INPUT, fg=self.TEXT_MAIN, command=lambda: self._step_frame(-1)).pack(side=tk.LEFT, padx=2)

        self.play_btn = tk.Button(ctrl_frame, text="▶ Play (Space)", width=14, bg=self.ACCENT_BLUE, fg="#000000",
                                  font=("Segoe UI", 9, "bold"), command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=6)

        tk.Button(ctrl_frame, text="Next >", width=6, bg=self.BG_INPUT, fg=self.TEXT_MAIN, command=lambda: self._step_frame(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text="+10 >>", width=6, bg=self.BG_INPUT, fg=self.TEXT_MAIN, command=lambda: self._step_frame(10)).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text=">>|", width=5, bg=self.BG_INPUT, fg=self.TEXT_MAIN, command=self._jump_last).pack(side=tk.LEFT, padx=2)

        # RIGHT PANEL: Tabbed Notebook (Quick Verification vs Manual Form)
        right_panel = tk.Frame(content_frame, bg=self.BG_CARD, width=380)
        right_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)

        ttk.Label(right_panel, text="Phase Annotation & Verification", style="Header.TLabel").pack(anchor=tk.W, padx=12, pady=(8, 4))

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ---------------------------------------------------------------------
        # TAB 1: Quick Verification (Task 5)
        # ---------------------------------------------------------------------
        tab_verify = tk.Frame(self.notebook, bg=self.BG_CARD)
        self.notebook.add(tab_verify, text="⚡ Quick Verification")

        self.quick_seq_lbl = tk.Label(tab_verify, text="Select a sequence from left panel to verify.",
                                      font=("Segoe UI", 9, "bold"), fg=self.ACCENT_BLUE, bg=self.BG_CARD, wraplength=340, justify="left")
        self.quick_seq_lbl.pack(anchor=tk.W, padx=10, pady=(8, 4))

        # 5-Phase Boundaries Display Table
        self.quick_tree = ttk.Treeview(tab_verify, columns=("phase", "start", "end", "conf"), show="headings", height=5)
        self.quick_tree.heading("phase", text="Phase")
        self.quick_tree.heading("start", text="Start")
        self.quick_tree.heading("end", text="End")
        self.quick_tree.heading("conf", text="Confidence")
        self.quick_tree.column("phase", width=125)
        self.quick_tree.column("start", width=55)
        self.quick_tree.column("end", width=55)
        self.quick_tree.column("conf", width=80)
        self.quick_tree.pack(fill=tk.X, padx=10, pady=4)
        self.quick_tree.bind("<<TreeviewSelect>>", self._on_quick_tree_selected)

        # Active boundary selector
        b_box = tk.Frame(tab_verify, bg=self.BG_INPUT, padx=8, pady=6)
        b_box.pack(fill=tk.X, padx=10, pady=6)

        self.active_boundary_lbl = tk.Label(b_box, text="Boundary: [ 1: RUN_UP / GATHER ]",
                                            font=("Segoe UI", 9, "bold"), fg=self.ACCENT_ORANGE, bg=self.BG_INPUT)
        self.active_boundary_lbl.pack(fill=tk.X, pady=(0, 4))

        b_nav = tk.Frame(b_box, bg=self.BG_INPUT)
        b_nav.pack(fill=tk.X)
        tk.Button(b_nav, text="◀ Previous Boundary", width=15, bg=self.BG_CARD, fg=self.TEXT_MAIN,
                  command=lambda: self._step_boundary(-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(b_nav, text="Next Boundary ▶", width=15, bg=self.BG_CARD, fg=self.TEXT_MAIN,
                  command=lambda: self._step_boundary(1)).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        # Boundary Shifting Controls (-10, -1, +1, +10)
        shift_frame = tk.Frame(tab_verify, bg=self.BG_CARD)
        shift_frame.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(shift_frame, text="Move Boundary Frames:", font=("Segoe UI", 9, "bold"),
                 fg=self.TEXT_MAIN, bg=self.BG_CARD).pack(anchor=tk.W, pady=(2, 4))

        shift_btns = tk.Frame(shift_frame, bg=self.BG_CARD)
        shift_btns.pack(fill=tk.X)
        tk.Button(shift_btns, text="Move -10", bg=self.BG_INPUT, fg=self.TEXT_MAIN, font=("Segoe UI", 9),
                  command=lambda: self._shift_boundary(-10)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(shift_btns, text="Move -1", bg=self.BG_INPUT, fg=self.ACCENT_BLUE, font=("Segoe UI", 9, "bold"),
                  command=lambda: self._shift_boundary(-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(shift_btns, text="Move +1", bg=self.BG_INPUT, fg=self.ACCENT_GREEN, font=("Segoe UI", 9, "bold"),
                  command=lambda: self._shift_boundary(1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(shift_btns, text="Move +10", bg=self.BG_INPUT, fg=self.TEXT_MAIN, font=("Segoe UI", 9),
                  command=lambda: self._shift_boundary(10)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Quick Verification Action Buttons
        act_box = tk.Frame(tab_verify, bg=self.BG_CARD)
        act_box.pack(fill=tk.X, padx=10, pady=(8, 4))

        tk.Button(act_box, text="⚡ Auto Detect Phases", bg=self.BG_INPUT, fg=self.ACCENT_BLUE, font=("Segoe UI", 9, "bold"),
                  command=self._run_auto_detection).pack(fill=tk.X, pady=2)
        tk.Button(act_box, text="▶ Preview Phases", bg=self.BG_INPUT, fg=self.ACCENT_ORANGE, font=("Segoe UI", 9, "bold"),
                  command=self._preview_phases).pack(fill=tk.X, pady=2)
        tk.Button(act_box, text="💾 Save Verified Labels", bg=self.ACCENT_GREEN, fg="#000000", font=("Segoe UI", 10, "bold"),
                  command=self._save_verified_labels).pack(fill=tk.X, pady=(6, 2))
        tk.Button(act_box, text="🔄 Re-run Automatic Detection", bg=self.BG_INPUT, fg=self.TEXT_MUTED, font=("Segoe UI", 8),
                  command=self._rerun_auto_detection).pack(fill=tk.X, pady=2)

        # ---------------------------------------------------------------------
        # TAB 2: Manual Form (Preserved for Backward Compatibility)
        # ---------------------------------------------------------------------
        tab_manual = tk.Frame(self.notebook, bg=self.BG_CARD)
        self.notebook.add(tab_manual, text="✏️ Manual Tagging")

        form_grid = tk.Frame(tab_manual, bg=self.BG_CARD)
        form_grid.pack(fill=tk.X, padx=12, pady=5)

        # Sequence ID Entry/Display
        tk.Label(form_grid, text="Sequence ID:", bg=self.BG_CARD, fg=self.TEXT_MAIN).grid(row=0, column=0, sticky="w", pady=4)
        self.seq_id_var = tk.StringVar()
        self.seq_id_entry = tk.Entry(form_grid, textvariable=self.seq_id_var, bg=self.BG_INPUT, fg=self.TEXT_MAIN, insertbackground=self.TEXT_MAIN)
        self.seq_id_entry.grid(row=0, column=1, sticky="ew", pady=4)

        # Start Frame Entry + Set button
        tk.Label(form_grid, text="Start Frame:", bg=self.BG_CARD, fg=self.TEXT_MAIN).grid(row=1, column=0, sticky="w", pady=4)
        self.start_frame_var = tk.StringVar(value="1")
        start_row = tk.Frame(form_grid, bg=self.BG_CARD)
        start_row.grid(row=1, column=1, sticky="ew", pady=4)
        self.start_entry = tk.Entry(start_row, textvariable=self.start_frame_var, width=6, bg=self.BG_INPUT, fg=self.TEXT_MAIN, insertbackground=self.TEXT_MAIN)
        self.start_entry.pack(side=tk.LEFT)
        tk.Button(start_row, text="[Set Curr (S)]", bg=self.BG_INPUT, fg=self.ACCENT_GREEN, command=self._set_start_frame).pack(side=tk.LEFT, padx=4)

        # End Frame Entry + Set button
        tk.Label(form_grid, text="End Frame:", bg=self.BG_CARD, fg=self.TEXT_MAIN).grid(row=2, column=0, sticky="w", pady=4)
        self.end_frame_var = tk.StringVar(value="1")
        end_row = tk.Frame(form_grid, bg=self.BG_CARD)
        end_row.grid(row=2, column=1, sticky="ew", pady=4)
        self.end_entry = tk.Entry(end_row, textvariable=self.end_frame_var, width=6, bg=self.BG_INPUT, fg=self.TEXT_MAIN, insertbackground=self.TEXT_MAIN)
        self.end_entry.pack(side=tk.LEFT)
        tk.Button(end_row, text="[Set Curr (E)]", bg=self.BG_INPUT, fg=self.ACCENT_GREEN, command=self._set_end_frame).pack(side=tk.LEFT, padx=4)

        form_grid.columnconfigure(1, weight=1)

        # Phase Selection (Radio Buttons)
        tk.Label(tab_manual, text="Bowling Phase (Mandatory):", font=("Segoe UI", 10, "bold"), bg=self.BG_CARD, fg=self.ACCENT_ORANGE).pack(anchor=tk.W, padx=12, pady=(10, 5))
        self.phase_var = tk.StringVar(value="RUN_UP")
        self.phase_var.trace_add("write", self._on_phase_var_changed)
        phase_box = tk.Frame(tab_manual, bg=self.BG_INPUT, padx=8, pady=6)
        phase_box.pack(fill=tk.X, padx=12)

        for phase in ALLOWED_PHASES:
            rb = tk.Radiobutton(phase_box, text=phase, value=phase, variable=self.phase_var,
                                bg=self.BG_INPUT, fg=self.TEXT_MAIN, selectcolor=self.BG_DARK,
                                activebackground=self.BG_INPUT, activeforeground=self.ACCENT_BLUE,
                                font=("Segoe UI", 9))
            rb.pack(anchor=tk.W, pady=2)

        # Annotator Name & Notes
        ann_meta_frame = tk.Frame(tab_manual, bg=self.BG_CARD)
        ann_meta_frame.pack(fill=tk.X, padx=12, pady=(10, 0))

        tk.Label(ann_meta_frame, text="Annotator Name:", bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(anchor=tk.W)
        self.annotator_var = tk.StringVar(value="Annotator_1")
        tk.Entry(ann_meta_frame, textvariable=self.annotator_var, bg=self.BG_INPUT, fg=self.TEXT_MAIN, insertbackground=self.TEXT_MAIN).pack(fill=tk.X, pady=(2, 6))

        tk.Label(ann_meta_frame, text="Notes / Observations:", bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(anchor=tk.W)
        self.notes_entry = tk.Entry(ann_meta_frame, bg=self.BG_INPUT, fg=self.TEXT_MAIN, insertbackground=self.TEXT_MAIN)
        self.notes_entry.pack(fill=tk.X, pady=(2, 8))

        # Save, Edit, Delete Buttons
        btn_box = tk.Frame(tab_manual, bg=self.BG_CARD)
        btn_box.pack(fill=tk.X, padx=12, pady=(8, 5))

        tk.Button(btn_box, text="💾 Save Annotation", bg=self.ACCENT_GREEN, fg="#000000", font=("Segoe UI", 10, "bold"),
                  height=2, command=self._save_annotation).pack(fill=tk.X, pady=3)

        action_row = tk.Frame(btn_box, bg=self.BG_CARD)
        action_row.pack(fill=tk.X, pady=3)
        tk.Button(action_row, text="✏️ Edit Selected", bg=self.BG_INPUT, fg=self.ACCENT_BLUE, width=15, command=self._edit_selected).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(action_row, text="🗑️ Delete", bg=self.BG_INPUT, fg=self.ACCENT_RED, width=12, command=self._delete_selected).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        tk.Button(btn_box, text="🔄 Reset Form", bg=self.BG_INPUT, fg=self.TEXT_MUTED, command=self._clear_form).pack(fill=tk.X, pady=3)

        # Bottom Status Bar
        self.status_var = tk.StringVar(value="Ready. Select a sequence or scrub video frames to begin annotation.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9),
                              bg="#11111b", fg=self.TEXT_MUTED, anchor=tk.W, padx=12, pady=5)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_keyboard_shortcuts(self) -> None:
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Left>", lambda e: self._step_frame(-1))
        self.root.bind("<Right>", lambda e: self._step_frame(1))
        self.root.bind("<Up>", lambda e: self._step_frame(10))
        self.root.bind("<Down>", lambda e: self._step_frame(-10))
        self.root.bind("<s>", lambda e: self._set_start_frame())
        self.root.bind("<S>", lambda e: self._set_start_frame())
        self.root.bind("<e>", lambda e: self._set_end_frame())
        self.root.bind("<E>", lambda e: self._set_end_frame())
        self.root.bind("<Control-s>", lambda e: self._save_annotation())

    def _load_video(self, video_id: str) -> None:
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.current_video_id = video_id
        meta = self.manager.video_metadata.get(video_id, {})
        self.total_frames = meta.get("total_frames", 1)
        self.current_frame = 1

        # Check raw video
        raw_path = meta.get("raw_path")
        if raw_path and os.path.exists(raw_path):
            self.cap = cv2.VideoCapture(raw_path)
            mode_desc = f"Raw Video: {os.path.basename(raw_path)}"
        else:
            mode_desc = "Skeleton Replay Mode (Raw video not in ai/data/raw/General/)"

        self.landmarks_data = self.manager.load_landmarks(video_id)

        self.video_info_lbl.config(
            text=f"Frames: {self.total_frames} | Det: {meta.get('detected_frames', 0)} ({meta.get('detection_pct', 0)}%) | {mode_desc}"
        )

        self.frame_slider.config(to=self.total_frames)
        self.frame_slider.set(1)

        self._refresh_sequence_list()
        self._refresh_annotations_list()
        self._render_current_frame()

    def _on_video_selected(self, event: Any) -> None:
        new_vid = self.video_combo.get()
        if new_vid and new_vid != self.current_video_id:
            self._load_video(new_vid)

    def _refresh_sequence_list(self) -> None:
        selected_seq = self.selected_seq_id

        for item in self.seq_tree.get_children():
            self.seq_tree.delete(item)

        seqs = self.manager.sequences.get(self.current_video_id, [])

        target_item = None
        for s in seqs:
            sid = s["sequence_id"]
            rng = f"{s['start_frame']}-{s['end_frame']}"
            if self.manager.is_sequence_verified(self.current_video_id, sid):
                status = "Verified ✓"
            else:
                status = "Pending"
            item_id = self.seq_tree.insert("", tk.END, values=(sid, rng, status))
            if sid == selected_seq:
                target_item = item_id

        if target_item:
            self.seq_tree.selection_set(target_item)
            self.seq_tree.focus(target_item)

    def _refresh_annotations_list(self) -> None:
        for item in self.ann_tree.get_children():
            self.ann_tree.delete(item)

        for ann in self.manager.annotations:
            if ann["video_id"] == self.current_video_id:
                rng = f"{ann['start_frame']}-{ann['end_frame']}"
                self.ann_tree.insert("", tk.END, values=(ann["phase_label"], rng, ann["annotator"]))

    def _on_seq_selected(self, event: Any) -> None:
        selected = self.seq_tree.selection()
        if not selected:
            return
        item_vals = self.seq_tree.item(selected[0], "values")
        seq_id = item_vals[0]
        self.selected_seq_id = seq_id
        self.seq_id_var.set(seq_id)

        seqs = self.manager.sequences.get(self.current_video_id, [])
        for s in seqs:
            if s["sequence_id"] == seq_id:
                self.start_frame_var.set(str(s["start_frame"]))
                self.end_frame_var.set(str(s["end_frame"]))
                self.current_frame = s["start_frame"]
                self.frame_slider.set(self.current_frame)
                self._load_quick_phases_for_seq(seq_id)
                self._render_current_frame()
                break

    def _on_ann_selected(self, event: Any) -> None:
        selected = self.ann_tree.selection()
        if not selected:
            return
        item_vals = self.ann_tree.item(selected[0], "values")
        phase = item_vals[0]
        frames_part = item_vals[1].split("-")
        if len(frames_part) == 2:
            st = frames_part[0]
            ed = frames_part[1]
            self.start_frame_var.set(st)
            self.end_frame_var.set(ed)
            self.phase_var.set(phase)
            self.current_frame = int(st)
            self.frame_slider.set(self.current_frame)
            for ann in self.manager.annotations:
                if (ann["video_id"] == self.current_video_id and
                    ann["phase_label"] == phase and
                    str(ann["start_frame"]) == st and
                    str(ann["end_frame"]) == ed):
                    self.seq_id_var.set(ann["sequence_id"])
                    self.annotator_var.set(ann.get("annotator", "Annotator_1"))
                    self.notes_entry.delete(0, tk.END)
                    self.notes_entry.insert(0, ann.get("notes", ""))
                    break
            self._render_current_frame()

    def _render_current_frame(self) -> None:
        from PIL import Image, ImageTk

        time_sec = (self.current_frame - 1) / 30.0
        self.time_lbl.config(text=f"Frame: {self.current_frame} / {self.total_frames} ({time_sec:.3f}s)")

        w = max(self.canvas.winfo_width(), 640)
        h = max(self.canvas.winfo_height(), 480)

        frame_rgb: Optional[np.ndarray] = None

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
            ret, frame_bgr = self.cap.read()
            if ret and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame_rgb is None:
            # Draw synthetic high-contrast dark pitch canvas
            frame_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            frame_rgb[:, :] = (20, 24, 35)

            # Draw subtle cricket pitch ground perspective guide
            cv2.line(frame_rgb, (0, int(h * 0.82)), (w, int(h * 0.82)), (40, 50, 70), 2)
            cv2.line(frame_rgb, (int(w * 0.2), int(h * 0.82)), (int(w * 0.35), int(h * 0.95)), (40, 50, 70), 1)
            cv2.line(frame_rgb, (int(w * 0.8), int(h * 0.82)), (int(w * 0.65), int(h * 0.95)), (40, 50, 70), 1)

        # Overlay MediaPipe skeleton from landmarks data
        lm_info = self.landmarks_data.get(self.current_frame, {})
        td = lm_info.get("target_detected", 0)
        lms = lm_info.get("landmarks", [])

        fh, fw, _ = frame_rgb.shape

        if td == 1 and len(lms) >= 33:
            # Draw bones
            for p1_idx, p2_idx in POSE_CONNECTIONS:
                if p1_idx < len(lms) and p2_idx < len(lms):
                    x1, y1, _, v1 = lms[p1_idx]
                    x2, y2, _, v2 = lms[p2_idx]
                    if v1 > 0.3 and v2 > 0.3:
                        pt1 = (int(x1 * fw), int(y1 * fh))
                        pt2 = (int(x2 * fw), int(y2 * fh))
                        cv2.line(frame_rgb, pt1, pt2, (80, 160, 255), 2, cv2.LINE_AA)

            # Draw joints
            for idx, (x, y, z, vis) in enumerate(lms):
                if vis > 0.3:
                    pt = (int(x * fw), int(y * fh))
                    cv2.circle(frame_rgb, pt, 4, (120, 255, 120), -1, cv2.LINE_AA)

            status_text = "● TARGET BOWLER TRACKED"
            status_color = (120, 255, 120)
        else:
            status_text = "○ TARGET LOST / NO SKELETON"
            status_color = (255, 100, 100)

        # Overlay frame header
        cv2.putText(frame_rgb, f"{self.current_video_id} | Frame {self.current_frame}/{self.total_frames}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_rgb, status_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # Overlay Quick Verification Phase Information
        if self.current_quick_phases:
            active_p = next((p for p in self.current_quick_phases if p["start_frame"] <= self.current_frame <= p["end_frame"]), None)
            if active_p:
                p_label = active_p["phase_label"]
                p_conf = active_p.get("confidence", 1.0)
                p_src = active_p.get("label_source", "AUTO")
                p_st = active_p["start_frame"]
                p_ed = active_p["end_frame"]

                banner_txt = f"PHASE: {p_label} ({p_st} -> {p_ed}) | Conf: {p_conf:.2f} [{p_src}]"
                cv2.rectangle(frame_rgb, (15, fh - 65), (min(fw - 15, 620), fh - 20), (35, 30, 20), -1)
                cv2.rectangle(frame_rgb, (15, fh - 65), (min(fw - 15, 620), fh - 20), (100, 220, 255), 2)
                cv2.putText(frame_rgb, banner_txt, (25, fh - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2, cv2.LINE_AA)

        # Highlight if current frame is in current annotation window
        try:
            sf = int(self.start_frame_var.get())
            ef = int(self.end_frame_var.get())
            if sf <= self.current_frame <= ef:
                cv2.rectangle(frame_rgb, (10, 10), (fw - 10, fh - 10), (255, 180, 50), 3)
                phase_name = self.phase_var.get()
                cv2.putText(frame_rgb, f"MANUAL WINDOW: {phase_name} ({sf} -> {ef})",
                            (20, fh - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 50), 2, cv2.LINE_AA)
        except ValueError:
            pass

        # Resize to fit canvas
        canvas_img = Image.fromarray(frame_rgb)
        canvas_img = canvas_img.resize((w, h), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(canvas_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def _step_frame(self, step: int) -> None:
        self.current_frame = max(1, min(self.total_frames, self.current_frame + step))
        self.frame_slider.set(self.current_frame)
        self._render_current_frame()

    def _on_slider_moved(self, val: str) -> None:
        self.current_frame = int(float(val))
        self._render_current_frame()

    def _jump_first(self) -> None:
        self.current_frame = 1
        self.frame_slider.set(1)
        self._render_current_frame()

    def _jump_last(self) -> None:
        self.current_frame = self.total_frames
        self.frame_slider.set(self.total_frames)
        self._render_current_frame()

    def _toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="⏸ Pause (Space)", bg=self.ACCENT_ORANGE)
            self._play_loop()
        else:
            self.play_btn.config(text="▶ Play (Space)", bg=self.ACCENT_BLUE)

    def _play_loop(self) -> None:
        if not self.is_playing:
            return
        if self.current_frame >= self.total_frames:
            self.is_playing = False
            self.play_btn.config(text="▶ Play (Space)", bg=self.ACCENT_BLUE)
            return

        self._step_frame(1)
        # 30 fps playback = ~33ms delay
        self.root.after(33, self._play_loop)

    def _set_start_frame(self) -> None:
        self.start_frame_var.set(str(self.current_frame))
        self.status_var.set(f"Start frame set to {self.current_frame}.")
        self._render_current_frame()

    def _set_end_frame(self) -> None:
        self.end_frame_var.set(str(self.current_frame))
        self.status_var.set(f"End frame set to {self.current_frame}.")
        self._render_current_frame()

    def _save_annotation(self) -> None:
        import tkinter.messagebox as msgbox

        vid = self.current_video_id
        seq_id = self.seq_id_var.get().strip()
        if not seq_id:
            msgbox.showerror("Validation Error", "Sequence ID cannot be empty. Select a sequence or enter a valid ID.")
            return

        try:
            start_f = int(self.start_frame_var.get().strip())
            end_f = int(self.end_frame_var.get().strip())
        except ValueError:
            msgbox.showerror("Validation Error", "Start Frame and End Frame must be valid integers.")
            return

        phase = self.phase_var.get().strip()
        annotator = self.annotator_var.get().strip()
        notes = self.notes_entry.get().strip()

        ok, err_msg = self.manager.save_annotation(
            video_id=vid,
            sequence_id=seq_id,
            start_frame=start_f,
            end_frame=end_f,
            phase_label=phase,
            annotator=annotator,
            notes=notes
        )

        if ok:
            if self.manager.is_short_sequence(vid, seq_id):
                self.current_quick_phases = [{
                    "video_id": vid,
                    "sequence_id": seq_id,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "phase_label": phase,
                    "confidence": 1.0,
                    "label_source": "VERIFIED",
                    "annotator": annotator,
                    "notes": notes
                }]
                self.quick_seq_lbl.config(text=f"Sequence: {seq_id} (Short Continuation — Verified ✓)")
                self._update_quick_phase_table()

            self.status_var.set(f"[SUCCESS] {err_msg}")
            self._refresh_sequence_list()
            self._refresh_annotations_list()
            self._render_current_frame()
            msgbox.showinfo("Saved", err_msg)
        else:
            self.status_var.set(f"[ERROR] {err_msg}")
            msgbox.showerror("Validation Error", err_msg)

    # =========================================================================
    # QUICK VERIFICATION HANDLERS (TASK 5)
    # =========================================================================

    def _guess_continuation_phase(self, seq_id: str) -> str:
        """Infer continuation phase from the preceding sequence of the same video."""
        seqs = self.manager.sequences.get(self.current_video_id, [])
        idx = next((i for i, s in enumerate(seqs) if s["sequence_id"] == seq_id), -1)
        if idx > 0:
            prev_seq_id = seqs[idx - 1]["sequence_id"]
            prev_anns = [a for a in self.manager.annotations if a["video_id"] == self.current_video_id and a["sequence_id"] == prev_seq_id]
            if prev_anns:
                last_ann = max(prev_anns, key=lambda a: int(a["end_frame"]))
                return last_ann["phase_label"]
            if prev_seq_id in self.manager.auto_annotations and self.manager.auto_annotations[prev_seq_id]:
                last_auto = max(self.manager.auto_annotations[prev_seq_id], key=lambda a: int(a["end_frame"]))
                return last_auto["phase_label"]
        return "FOLLOW_THROUGH"

    def _on_phase_var_changed(self, *args: Any) -> None:
        if not self.selected_seq_id:
            return
        if self.manager.is_short_sequence(self.current_video_id, self.selected_seq_id):
            new_phase = self.phase_var.get().strip()
            if new_phase in ALLOWED_PHASES and self.current_quick_phases:
                self.current_quick_phases[0]["phase_label"] = new_phase
                self._update_quick_phase_table()
                self._render_current_frame()

    def _on_quick_tree_selected(self, event: Any) -> None:
        selected = self.quick_tree.selection()
        if not selected:
            return
        idx = self.quick_tree.index(selected[0])
        if self.current_quick_phases and len(self.current_quick_phases) >= 5:
            b_idx = min(3, idx)
            self._select_boundary(b_idx)
        elif self.current_quick_phases and len(self.current_quick_phases) == 1:
            p = self.current_quick_phases[0]
            self.current_frame = p["start_frame"]
            self.frame_slider.set(self.current_frame)
            self._render_current_frame()

    def _select_boundary(self, b_idx: int) -> None:
        if not self.current_quick_phases:
            return
        if len(self.current_quick_phases) < 5:
            if len(self.current_quick_phases) == 1:
                p = self.current_quick_phases[0]
                self.active_boundary_lbl.config(text=f"Single Phase: [ {p['phase_label']} ({p['start_frame']}–{p['end_frame']}) ]")
            return
        self.active_boundary_idx = max(0, min(3, b_idx))
        b_names = [
            f"1: RUN_UP / GATHER (Frame {self.current_quick_phases[0]['end_frame']})",
            f"2: GATHER / DELIVERY_STRIDE (Frame {self.current_quick_phases[1]['end_frame']})",
            f"3: DELIVERY_STRIDE / RELEASE (Frame {self.current_quick_phases[2]['end_frame']})",
            f"4: RELEASE / FOLLOW_THROUGH (Frame {self.current_quick_phases[3]['end_frame']})"
        ]
        self.active_boundary_lbl.config(text=f"Boundary: [ {b_names[self.active_boundary_idx]} ]")
        target_f = self.current_quick_phases[self.active_boundary_idx]["end_frame"]
        self.current_frame = target_f
        self.frame_slider.set(target_f)
        self._render_current_frame()

    def _step_boundary(self, step: int) -> None:
        if not self.current_quick_phases or len(self.current_quick_phases) < 5:
            return
        new_idx = (self.active_boundary_idx + step) % 4
        self._select_boundary(new_idx)

    def _shift_boundary(self, delta: int) -> None:
        if not self.current_quick_phases or len(self.current_quick_phases) < 5:
            self.status_var.set("[NOTICE] Boundary shifting applies to 5-phase sequences only.")
            return

        k = self.active_boundary_idx
        cur_b = self.current_quick_phases[k]["end_frame"]
        min_allowed = self.current_quick_phases[k]["start_frame"] if k == 0 else self.current_quick_phases[k-1]["end_frame"] + 1
        max_allowed = self.current_quick_phases[k+1]["end_frame"] - 1

        if min_allowed > max_allowed:
            self.status_var.set("[WARN] Cannot move boundary further without collapsing adjacent phase.")
            return

        new_b = max(min_allowed, min(max_allowed, cur_b + delta))
        if new_b == cur_b:
            self.status_var.set(f"[NOTICE] Boundary {k+1} is at its limit ({cur_b}).")
            return

        # Apply update
        self.current_quick_phases[k]["end_frame"] = new_b
        self.current_quick_phases[k+1]["start_frame"] = new_b + 1
        self.current_quick_phases[k]["confidence"] = 1.0 # human confirmed
        self.current_quick_phases[k+1]["confidence"] = 1.0

        self._update_quick_phase_table()
        self._select_boundary(k)
        self.status_var.set(f"Boundary {k+1} moved by {delta:+d} -> Frame {new_b} (Preserved zero gaps/overlaps).")

    def _update_quick_phase_table(self) -> None:
        for item in self.quick_tree.get_children():
            self.quick_tree.delete(item)

        for p in self.current_quick_phases:
            conf_val = p.get("confidence", 1.0)
            src = p.get("label_source", "AUTO")
            conf_str = f"{conf_val:.2f} ({src})"
            self.quick_tree.insert("", tk.END, values=(p["phase_label"], p["start_frame"], p["end_frame"], conf_str))

        if self.current_quick_phases and len(self.current_quick_phases) >= 5:
            k = self.active_boundary_idx
            b_names = [
                f"1: RUN_UP / GATHER (Frame {self.current_quick_phases[0]['end_frame']})",
                f"2: GATHER / DELIVERY_STRIDE (Frame {self.current_quick_phases[1]['end_frame']})",
                f"3: DELIVERY_STRIDE / RELEASE (Frame {self.current_quick_phases[2]['end_frame']})",
                f"4: RELEASE / FOLLOW_THROUGH (Frame {self.current_quick_phases[3]['end_frame']})"
            ]
            self.active_boundary_lbl.config(text=f"Boundary: [ {b_names[k]} ]")
        elif self.current_quick_phases and len(self.current_quick_phases) == 1:
            p = self.current_quick_phases[0]
            self.active_boundary_lbl.config(text=f"Single Phase: [ {p['phase_label']} ({p['start_frame']}–{p['end_frame']}) ]")
        else:
            self.active_boundary_lbl.config(text="Boundary: [ None ]")

    def _load_quick_phases_for_seq(self, seq_id: str) -> None:
        self.selected_seq_id = seq_id

        # Check if verified annotations already exist for this sequence
        ver_anns = [a for a in self.manager.annotations if a.get("video_id") == self.current_video_id and a.get("sequence_id") == seq_id]

        is_short = self.manager.is_short_sequence(self.current_video_id, seq_id)

        if is_short:
            if len(ver_anns) == 1:
                ok, _ = self.manager.validate_sequence_phases(self.current_video_id, seq_id, ver_anns)
                if ok:
                    self.current_quick_phases = [dict(a) for a in ver_anns]
                    for p in self.current_quick_phases:
                        p["label_source"] = "VERIFIED"
                    self.phase_var.set(ver_anns[0]["phase_label"])
                    self.quick_seq_lbl.config(text=f"Sequence: {seq_id} (Short Continuation — Verified ✓)")
                    self._update_quick_phase_table()
                    return

            # Short continuation sequence not yet verified:
            # Infer applicable phase (for General2_seq_02 -> FOLLOW_THROUGH: 96-99)
            applicable_phase = self._guess_continuation_phase(seq_id)
            cur_seq = self.manager.get_sequence_info(self.current_video_id, seq_id)
            sf = int(cur_seq["start_frame"]) if cur_seq else int(self.start_frame_var.get())
            ef = int(cur_seq["end_frame"]) if cur_seq else int(self.end_frame_var.get())

            self.phase_var.set(applicable_phase)
            self.current_quick_phases = [{
                "video_id": self.current_video_id,
                "sequence_id": seq_id,
                "start_frame": sf,
                "end_frame": ef,
                "phase_label": applicable_phase,
                "confidence": 1.0,
                "label_source": "MANUAL",
                "annotator": self.annotator_var.get().strip() or "Annotator_1",
                "notes": f"Continuation of {applicable_phase}"
            }]
            self.quick_seq_lbl.config(text=f"Sequence: {seq_id} (Short Continuation Segment: {sf}–{ef})")
            self._update_quick_phase_table()
            return

        # Normal sequence (>= 5 frames)
        if len(ver_anns) == 5:
            phase_order = {name: i for i, name in enumerate(ALLOWED_PHASES)}
            sorted_ver = sorted(ver_anns, key=lambda x: phase_order.get(x["phase_label"], 99))
            is_valid, _ = self.manager.validate_sequence_phases(self.current_video_id, seq_id, sorted_ver)
            if is_valid:
                self.current_quick_phases = [dict(a) for a in sorted_ver]
                for p in self.current_quick_phases:
                    p["label_source"] = "VERIFIED"
                self.quick_seq_lbl.config(text=f"Sequence: {seq_id} (Verified ✓)")
                self._update_quick_phase_table()
                self._select_boundary(0)
                return

        # Check auto annotations
        if seq_id in self.manager.auto_annotations:
            self.current_quick_phases = [dict(a) for a in self.manager.auto_annotations[seq_id]]
            self.quick_seq_lbl.config(text=f"Sequence: {seq_id} (⚡ AUTO-DETECTED)")
            self._update_quick_phase_table()
            self._select_boundary(0)
        else:
            self.current_quick_phases = []
            self.quick_seq_lbl.config(text=f"Sequence: {seq_id} (No auto annotations — click Auto Detect)")
            self._update_quick_phase_table()

    def _run_auto_detection(self) -> None:
        import tkinter.messagebox as msgbox
        from auto_phase_detector import AutoPhaseDetector, LandmarkPreprocessor

        if not self.selected_seq_id:
            msgbox.showwarning("Select Sequence", "Please select a sequence from the manifest first.")
            return

        seqs = self.manager.sequences.get(self.current_video_id, [])
        cur_seq = next((s for s in seqs if s["sequence_id"] == self.selected_seq_id), None)
        if not cur_seq:
            msgbox.showerror("Error", f"Sequence {self.selected_seq_id} not found.")
            return

        sf = int(cur_seq["start_frame"])
        ef = int(cur_seq["end_frame"])
        seq_len = ef - sf + 1

        if seq_len < 5:
            assigned_phase = self._guess_continuation_phase(self.selected_seq_id)
            self.phase_var.set(assigned_phase)
            self.current_quick_phases = [{
                "video_id": self.current_video_id,
                "sequence_id": self.selected_seq_id,
                "start_frame": sf,
                "end_frame": ef,
                "phase_label": assigned_phase,
                "confidence": 1.0,
                "label_source": "MANUAL",
                "annotator": self.annotator_var.get().strip() or "Annotator_1",
                "notes": f"Continuation of {assigned_phase}"
            }]
            self._update_quick_phase_table()
            self._render_current_frame()
            self.status_var.set(f"[SHORT CONTINUATION] Sequence has {seq_len} frames (<5). Assigned continuation phase: {assigned_phase}.")
            msgbox.showinfo(
                "Short Continuation Sequence",
                f"Sequence '{self.selected_seq_id}' has {seq_len} frames (< 5 required for 5 distinct phases).\n\n"
                f"This is a short continuation segment. No automatic 5-phase segmentation is generated.\n\n"
                f"Applicable phase '{assigned_phase}' has been assigned ({sf}–{ef}). You can verify and click 'Save Verified Labels'."
            )
            return

        detector = AutoPhaseDetector()
        p = os.path.join(self.manager.landmarks_dir, f"{self.current_video_id}_landmarks.csv")
        rows = LandmarkPreprocessor.load_landmarks_csv(p)
        seq_rows = [r for r in rows if sf <= r[0] <= ef]

        res = detector.detect_sequence_phases(seq_rows, self.current_video_id, self.selected_seq_id)
        if res["status"] == "SUCCESS":
            self.current_quick_phases = list(res["phases"])
            self.manager.auto_annotations[self.selected_seq_id] = list(res["phases"])
            self._update_quick_phase_table()
            self._select_boundary(0)
            self.status_var.set(f"[AUTO-DETECT] Successfully generated 5 phases for {self.selected_seq_id} (Avg Conf: {res['report']['average_confidence']:.2f}).")
            msgbox.showinfo("Auto Detection Complete", f"Generated 5 bowling phases for {self.selected_seq_id}!\nAverage Confidence: {res['report']['average_confidence']:.2f}\nStatus: {res['report']['detection_status']}")
        else:
            msgbox.showwarning("Detection Failed", f"Could not auto-detect phases for {self.selected_seq_id}:\n{res.get('reason', 'Unknown reason')}")

    def _preview_phases(self) -> None:
        if not self.current_quick_phases:
            import tkinter.messagebox as msgbox
            msgbox.showwarning("Notice", "Load or assign phases first to preview.")
            return

        self.is_previewing_phases = True
        self.current_frame = self.current_quick_phases[0]["start_frame"]
        self.frame_slider.set(self.current_frame)
        self.is_playing = True
        self.play_btn.config(text="⏸ Pause (Space)", bg=self.ACCENT_ORANGE)
        self._play_loop()

    def _save_verified_labels(self) -> None:
        import tkinter.messagebox as msgbox
        if not self.selected_seq_id:
            msgbox.showwarning("No Sequence Selected", "Please select a sequence from the Candidate Sequences panel first.")
            return

        is_short = self.manager.is_short_sequence(self.current_video_id, self.selected_seq_id)
        if is_short:
            if not self.current_quick_phases or len(self.current_quick_phases) != 1:
                msgbox.showerror("Validation Error", "Please assign the applicable phase for this short continuation sequence.")
                return
        else:
            if not self.current_quick_phases or len(self.current_quick_phases) != 5:
                msgbox.showerror("Validation Error", "No complete 5-phase set available to save. Please auto-detect or define all 5 phases first.")
                return

        # Validate before saving (verifies all phases present, order, ranges, continuity, sequence coverage)
        ok, err_msg = self.manager.validate_sequence_phases(
            video_id=self.current_video_id,
            sequence_id=self.selected_seq_id,
            phases_list=self.current_quick_phases
        )
        if not ok:
            msgbox.showerror("Validation Error", f"Cannot save verified labels:\n\n{err_msg}")
            return

        annotator = self.annotator_var.get().strip() or "Annotator_1"
        notes = self.notes_entry.get().strip() or "Verified by human review"

        ok, msg = self.manager.save_verified_phases(
            video_id=self.current_video_id,
            sequence_id=self.selected_seq_id,
            phases_list=self.current_quick_phases,
            annotator=annotator,
            notes=notes
        )

        if ok:
            for p in self.current_quick_phases:
                p["label_source"] = "VERIFIED"
            status_desc = "Short Continuation — Verified ✓" if is_short else "Verified ✓"
            self.quick_seq_lbl.config(text=f"Sequence: {self.selected_seq_id} ({status_desc})")
            self._update_quick_phase_table()
            self._refresh_annotations_list()
            self._refresh_sequence_list()
            self._render_current_frame()
            self.status_var.set(f"[VERIFIED] {msg}")
            phase_detail = f"{self.current_quick_phases[0]['phase_label']} ({self.current_quick_phases[0]['start_frame']}–{self.current_quick_phases[0]['end_frame']})" if is_short else "all 5 verified bowling phases"
            msgbox.showinfo("Verified Saved", f"Successfully saved verified annotation ({phase_detail}) for '{self.selected_seq_id}'!\n\nSequence status updated to Verified ✓.")
        else:
            msgbox.showerror("Validation Error", msg)

    def _rerun_auto_detection(self) -> None:
        self._run_auto_detection()

    def _edit_selected(self) -> None:
        selected = self.ann_tree.selection()
        if not selected:
            self.status_var.set("Select an annotation from the list below to edit.")
            return
        self._on_ann_selected(None)
        self.status_var.set("Loaded selected annotation into form. Modify values and click Save Annotation.")

    def _delete_selected(self) -> None:
        import tkinter.messagebox as msgbox
        selected = self.ann_tree.selection()
        if not selected:
            self.status_var.set("Select an annotation from the list below to delete.")
            return

        seq_id = self.seq_id_var.get().strip()
        if not seq_id:
            msgbox.showwarning("Notice", "Select an annotation with a valid Sequence ID to delete.")
            return

        confirm = msgbox.askyesno("Confirm Deletion", f"Delete ground-truth annotation for sequence '{seq_id}'?")
        if confirm:
            ok, msg = self.manager.delete_annotation(self.current_video_id, seq_id)
            if ok:
                self.status_var.set(f"[DELETED] {msg}")
                self._refresh_sequence_list()
                self._refresh_annotations_list()
                self._render_current_frame()
            else:
                self.status_var.set(f"[ERROR] {msg}")

    def _clear_form(self) -> None:
        self.seq_id_var.set("")
        self.start_frame_var.set("1")
        self.end_frame_var.set("1")
        self.phase_var.set("RUN_UP")
        self.notes_entry.delete(0, "end")
        self.status_var.set("Form reset.")
        self._render_current_frame()


# =============================================================================
# SELF-TEST & AUTOMATED VERIFICATION SUITE
# =============================================================================

def run_self_test() -> bool:
    """
    Automated non-interactive validation test suite:
      1. Verifies discovery of all 10 General videos.
      2. Verifies loading of general_sequence_manifest.csv.
      3. Tests validation engine for correct rules rejection.
      4. Saves a test annotation and verifies persistence.
      5. Reloads annotations from disk and verifies consistency.
      6. Edits test annotation and verifies update.
      7. Deletes test annotation and verifies clean state.
    """
    print("\n" + "=" * 70)
    print("  CrickSense — Manual Annotation Tool Self-Test Suite")
    print("=" * 70)

    mgr = AnnotationDataManager()

    # 1. Video discovery check
    print(f"\n[Test 1] Video Discovery:")
    print(f"  Discovered videos count: {len(mgr.videos)}")
    assert len(mgr.videos) == 10, f"Expected 10 General videos, found {len(mgr.videos)}"
    expected_vids = ["General1", "General2", "General3", "S_v5", "s_v1", "s_v2", "s_v3", "s_v4", "s_v6", "s_v7"]
    for ev in expected_vids:
        assert ev in mgr.videos, f"Missing expected video: {ev}"
        total_f = mgr.video_metadata[ev]["total_frames"]
        det_f = mgr.video_metadata[ev]["detected_frames"]
        print(f"    - {ev:<10}: {total_f} frames, {det_f} detected ({mgr.video_metadata[ev]['detection_pct']}%)")
    print("  --> PASS: All 10 General videos correctly discovered.")

    # 2. Sequence manifest check
    print(f"\n[Test 2] Sequence Manifest Loading:")
    total_seqs = sum(len(seqs) for seqs in mgr.sequences.values())
    print(f"  Total candidate sequences loaded: {total_seqs}")
    assert total_seqs == 73, f"Expected 73 candidate sequences in manifest, found {total_seqs}"
    assert "General1" in mgr.sequences and len(mgr.sequences["General1"]) == 1
    assert mgr.sequences["General1"][0]["sequence_id"] == "General1_seq_01"
    print("  --> PASS: Sequence manifest successfully loaded.")

    # 3. Validation rules check
    print(f"\n[Test 3] Validation Rules Enforcement:")
    # Wrong video_id
    ok, _ = mgr.validate_annotation("U15_v1", "General1_seq_01", 1, 10, "RUN_UP")
    assert not ok, "Validation should reject non-General video_id"

    # Wrong sequence_id
    ok, _ = mgr.validate_annotation("General1", "non_existent_seq", 1, 10, "RUN_UP")
    assert not ok, "Validation should reject invalid sequence_id"

    # start_frame > end_frame
    ok, _ = mgr.validate_annotation("General1", "General1_seq_01", 50, 20, "RUN_UP")
    assert not ok, "Validation should reject start_frame > end_frame"

    # Frame exceeding total_frames (General1 has 76 frames)
    ok, _ = mgr.validate_annotation("General1", "General1_seq_01", 1, 999, "RUN_UP")
    assert not ok, "Validation should reject frames exceeding total_frames"

    # Invalid phase label
    ok, _ = mgr.validate_annotation("General1", "General1_seq_01", 1, 20, "INVALID_PHASE")
    assert not ok, "Validation should reject invalid phase label"

    # Valid case
    ok, msg = mgr.validate_annotation("General1", "General1_seq_01", 1, 25, "RUN_UP")
    assert ok, f"Validation should accept valid input: {msg}"
    print("  --> PASS: All validation rules strictly enforced.")

    # 4. Save and reload test
    print(f"\n[Test 4] Save, Reload, Edit, and Delete Lifecycle:")
    test_vid = "General1"
    test_seq = "General1_seq_01"
    test_annotator = "SelfTest_Bot"

    # Ensure clean slate for test sequence
    mgr.delete_annotation(test_vid, test_seq)

    # Save
    ok, msg = mgr.save_annotation(test_vid, test_seq, 1, 30, "RUN_UP", annotator=test_annotator, notes="Unit test")
    assert ok, f"Save failed: {msg}"
    print(f"  Saved annotation: {msg}")

    # Reload from fresh manager instance
    fresh_mgr = AnnotationDataManager()
    saved = [a for a in fresh_mgr.annotations if a["video_id"] == test_vid and a["sequence_id"] == test_seq]
    assert len(saved) == 1, "Saved annotation not persisted to CSV"
    assert saved[0]["phase_label"] == "RUN_UP"
    assert saved[0]["start_frame"] == 1
    assert saved[0]["end_frame"] == 30
    print("  Reload verification: PERSISTED SUCCESSFULLY.")

    # Edit
    ok, msg = fresh_mgr.save_annotation(test_vid, test_seq, 1, 35, "GATHER", annotator=test_annotator, notes="Edited test")
    assert ok, f"Edit failed: {msg}"
    fresh_mgr2 = AnnotationDataManager()
    edited = [a for a in fresh_mgr2.annotations if a["video_id"] == test_vid and a["sequence_id"] == test_seq]
    assert len(edited) == 1 and edited[0]["phase_label"] == "GATHER" and edited[0]["end_frame"] == 35
    print("  Edit verification: UPDATED SUCCESSFULLY.")

    # Delete
    ok, msg = fresh_mgr2.delete_annotation(test_vid, test_seq)
    assert ok, f"Delete failed: {msg}"
    fresh_mgr3 = AnnotationDataManager()
    cleaned = [a for a in fresh_mgr3.annotations if a["video_id"] == test_vid and a["sequence_id"] == test_seq]
    assert len(cleaned) == 0, "Deleted annotation was not removed from CSV"
    print("  Delete verification: REMOVED CLEANLY.")
    print("  --> PASS: Complete CRUD lifecycle verified.")

    # 5. Landmark loading check
    print(f"\n[Test 5] Landmark Extraction Check:")
    lms = mgr.load_landmarks("General1")
    assert len(lms) == 76, f"Expected 76 frames in General1 landmark data, got {len(lms)}"
    assert lms[1]["target_detected"] == 1
    assert len(lms[1]["landmarks"]) == 33
    print("  --> PASS: Landmark loading verified (33 joints parsed).")

    # 6. GUI Layout & Initialization check
    print(f"\n[Test 6] Tkinter GUI Layout & Initialization Check:")
    root = tk.Tk()
    root.withdraw()
    app = AnnotationApp(root, mgr)
    root.update()
    root.destroy()
    print("  --> PASS: AnnotationApp window layout and canvas rendered cleanly.")

    # 7. 5-Phase Sequence Validation Enforcement (Task 2)
    print(f"\n[Test 7] 5-Phase Sequence Validation Enforcement:")
    # Invalid: missing phase (only 4 phases)
    incomplete_phases = [
        {"phase_label": "RUN_UP", "start_frame": 1, "end_frame": 19},
        {"phase_label": "GATHER", "start_frame": 20, "end_frame": 33},
        {"phase_label": "DELIVERY_STRIDE", "start_frame": 34, "end_frame": 44},
        {"phase_label": "FOLLOW_THROUGH", "start_frame": 45, "end_frame": 76},
    ]
    ok, err = mgr.validate_sequence_phases("General1", "General1_seq_01", incomplete_phases)
    assert not ok and "5" in err, f"Validation should reject missing phase: {err}"

    # Invalid: wrong order
    scrambled_phases = [
        {"phase_label": "GATHER", "start_frame": 1, "end_frame": 19},
        {"phase_label": "RUN_UP", "start_frame": 20, "end_frame": 33},
        {"phase_label": "DELIVERY_STRIDE", "start_frame": 34, "end_frame": 44},
        {"phase_label": "RELEASE", "start_frame": 45, "end_frame": 47},
        {"phase_label": "FOLLOW_THROUGH", "start_frame": 48, "end_frame": 76},
    ]
    ok, err = mgr.validate_sequence_phases("General1", "General1_seq_01", scrambled_phases)
    assert not ok and "canonical order" in err, f"Validation should reject wrong order: {err}"

    # Invalid: gap between RUN_UP and GATHER
    gap_phases = [
        {"phase_label": "RUN_UP", "start_frame": 1, "end_frame": 18},
        {"phase_label": "GATHER", "start_frame": 20, "end_frame": 33},
        {"phase_label": "DELIVERY_STRIDE", "start_frame": 34, "end_frame": 44},
        {"phase_label": "RELEASE", "start_frame": 45, "end_frame": 47},
        {"phase_label": "FOLLOW_THROUGH", "start_frame": 48, "end_frame": 76},
    ]
    ok, err = mgr.validate_sequence_phases("General1", "General1_seq_01", gap_phases)
    assert not ok and "Gap detected" in err, f"Validation should reject gap: {err}"

    # Invalid: overlap between RUN_UP and GATHER
    overlap_phases = [
        {"phase_label": "RUN_UP", "start_frame": 1, "end_frame": 20},
        {"phase_label": "GATHER", "start_frame": 20, "end_frame": 33},
        {"phase_label": "DELIVERY_STRIDE", "start_frame": 34, "end_frame": 44},
        {"phase_label": "RELEASE", "start_frame": 45, "end_frame": 47},
        {"phase_label": "FOLLOW_THROUGH", "start_frame": 48, "end_frame": 76},
    ]
    ok, err = mgr.validate_sequence_phases("General1", "General1_seq_01", overlap_phases)
    assert not ok and "Overlap detected" in err, f"Validation should reject overlap: {err}"

    # Invalid: does not cover complete sequence (starts at frame 2 instead of 1)
    incomplete_start_phases = [
        {"phase_label": "RUN_UP", "start_frame": 2, "end_frame": 19},
        {"phase_label": "GATHER", "start_frame": 20, "end_frame": 33},
        {"phase_label": "DELIVERY_STRIDE", "start_frame": 34, "end_frame": 44},
        {"phase_label": "RELEASE", "start_frame": 45, "end_frame": 47},
        {"phase_label": "FOLLOW_THROUGH", "start_frame": 48, "end_frame": 76},
    ]
    ok, err = mgr.validate_sequence_phases("General1", "General1_seq_01", incomplete_start_phases)
    assert not ok and "cover sequence start" in err, f"Validation should reject incomplete start: {err}"

    # Invalid: does not cover complete sequence (ends at frame 75 instead of 76)
    incomplete_end_phases = [
        {"phase_label": "RUN_UP", "start_frame": 1, "end_frame": 19},
        {"phase_label": "GATHER", "start_frame": 20, "end_frame": 33},
        {"phase_label": "DELIVERY_STRIDE", "start_frame": 34, "end_frame": 44},
        {"phase_label": "RELEASE", "start_frame": 45, "end_frame": 47},
        {"phase_label": "FOLLOW_THROUGH", "start_frame": 48, "end_frame": 75},
    ]
    ok, err = mgr.validate_sequence_phases("General1", "General1_seq_01", incomplete_end_phases)
    assert not ok and "cover sequence end" in err, f"Validation should reject incomplete end: {err}"
    print("  --> PASS: 5-phase validation engine strictly catches missing phases, order, gaps, overlaps, and sequence coverage.")

    # 8. Save Verified Labels for General1_seq_01 & Status Updates
    print(f"\n[Test 8] Save Verified Labels for General1_seq_01 & Status Updates:")
    g1_phases = [
        {"phase_label": "RUN_UP", "start_frame": 1, "end_frame": 19},
        {"phase_label": "GATHER", "start_frame": 20, "end_frame": 33},
        {"phase_label": "DELIVERY_STRIDE", "start_frame": 34, "end_frame": 44},
        {"phase_label": "RELEASE", "start_frame": 45, "end_frame": 47},
        {"phase_label": "FOLLOW_THROUGH", "start_frame": 48, "end_frame": 76},
    ]

    # Verify initial status is Pending if no verified annotations exist
    mgr.delete_annotation("General1", "General1_seq_01")
    assert not mgr.is_sequence_verified("General1", "General1_seq_01"), "Sequence should be Pending initially"

    # Save verified phases (simulating user selecting FOLLOW_THROUGH row or any phase beforehand)
    ok, msg = mgr.save_verified_phases("General1", "General1_seq_01", g1_phases, annotator="Annotator_1", notes="Verified by human review")
    assert ok, f"Save verified phases failed: {msg}"

    # Confirm exactly 5 rows saved
    fresh_mgr4 = AnnotationDataManager()
    ver_saved = [a for a in fresh_mgr4.annotations if a["video_id"] == "General1" and a["sequence_id"] == "General1_seq_01"]
    assert len(ver_saved) == 5, f"Expected exactly 5 verified saved rows, got {len(ver_saved)}"
    expected_order = ["RUN_UP", "GATHER", "DELIVERY_STRIDE", "RELEASE", "FOLLOW_THROUGH"]
    for idx, exp_label in enumerate(expected_order):
        assert ver_saved[idx]["phase_label"] == exp_label, f"Expected {exp_label} at index {idx}, got {ver_saved[idx]['phase_label']}"

    # Confirm sequence status changed to Verified
    assert fresh_mgr4.is_sequence_verified("General1", "General1_seq_01"), "Sequence status should be Verified ✓"

    # Test saving again does NOT create duplicate rows
    ok, msg = fresh_mgr4.save_verified_phases("General1", "General1_seq_01", g1_phases, annotator="Annotator_1", notes="Re-save check")
    assert ok
    fresh_mgr5 = AnnotationDataManager()
    re_saved = [a for a in fresh_mgr5.annotations if a["video_id"] == "General1" and a["sequence_id"] == "General1_seq_01"]
    assert len(re_saved) == 5, f"Re-saving created duplicates! Expected 5 rows, got {len(re_saved)}"

    # Confirm CSV schema has the 7 required columns
    with open(fresh_mgr5.annotations_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ANNOTATION_COLUMNS, f"CSV header mismatch! Expected {ANNOTATION_COLUMNS}, got {header}"
    print("  --> PASS: Exactly 5 General1 annotations saved, no duplicates on re-save, status changed Pending -> Verified [OK], 7-column CSV schema preserved.")

    # 9. Short Continuation Sequence Validation & General2_seq_02 Verification
    print(f"\n[Test 9] Short Continuation Sequence Validation & General2_seq_02 Verification:")
    # 9.1 Short sequence identification
    assert mgr.is_short_sequence("General2", "General2_seq_02"), "General2_seq_02 (4 frames) must be identified as short sequence"
    assert not mgr.is_short_sequence("General2", "General2_seq_01"), "General2_seq_01 (94 frames) must NOT be short sequence"

    # 9.2 Validation: reject 5 artificial phases on short sequence
    ok, err = mgr.validate_sequence_phases("General2", "General2_seq_02", g1_phases)
    assert not ok and "requires exactly 1 assigned phase" in err, f"Must reject 5 artificial phases on short sequence: {err}"

    # 9.3 Validation: reject invalid phase label
    invalid_short = [{"phase_label": "NOT_A_PHASE", "start_frame": 96, "end_frame": 99}]
    ok, err = mgr.validate_sequence_phases("General2", "General2_seq_02", invalid_short)
    assert not ok and "Invalid phase_label" in err, f"Must reject invalid phase label: {err}"

    # 9.4 Validation: reject frame bounds mismatching sequence
    mismatch_short = [{"phase_label": "FOLLOW_THROUGH", "start_frame": 96, "end_frame": 98}]
    ok, err = mgr.validate_sequence_phases("General2", "General2_seq_02", mismatch_short)
    assert not ok and "must cover short sequence boundaries" in err, f"Must reject mismatched bounds: {err}"

    # 9.5 Validation: accept valid single applicable phase
    valid_g2_s2 = [{"phase_label": "FOLLOW_THROUGH", "start_frame": 96, "end_frame": 99}]
    ok, msg = mgr.validate_sequence_phases("General2", "General2_seq_02", valid_g2_s2)
    assert ok, f"Must accept valid FOLLOW_THROUGH: 96-99 for General2_seq_02: {msg}"

    # 9.6 Save verified single phase for General2_seq_02
    ok, msg = mgr.save_verified_phases("General2", "General2_seq_02", valid_g2_s2, annotator="Annotator_1", notes="Verified by human review")
    assert ok, f"Save verified failed for General2_seq_02: {msg}"

    # 9.7 Confirm sequence status is Verified
    assert mgr.is_sequence_verified("General2", "General2_seq_02"), "General2_seq_02 status must be Verified ✓"

    # 9.8 Reopen / reload fresh manager from disk and verify persistence
    reloaded_mgr = AnnotationDataManager()
    assert reloaded_mgr.is_sequence_verified("General2", "General2_seq_02"), "General2_seq_02 must remain Verified ✓ after disk reload"
    g2_s2_anns = [a for a in reloaded_mgr.annotations if a["video_id"] == "General2" and a["sequence_id"] == "General2_seq_02"]
    assert len(g2_s2_anns) == 1, f"Expected 1 annotation for General2_seq_02, found {len(g2_s2_anns)}"
    assert g2_s2_anns[0]["phase_label"] == "FOLLOW_THROUGH"
    assert g2_s2_anns[0]["start_frame"] == 96
    assert g2_s2_anns[0]["end_frame"] == 99
    assert g2_s2_anns[0]["annotator"] == "Annotator_1"

    # 9.9 Verify other completed annotations are completely preserved
    assert reloaded_mgr.is_sequence_verified("General2", "General2_seq_01"), "General2_seq_01 must remain Verified ✓"
    assert reloaded_mgr.is_sequence_verified("General2", "General2_seq_03"), "General2_seq_03 must remain Verified ✓"
    assert reloaded_mgr.is_sequence_verified("General2", "General2_seq_05"), "General2_seq_05 must remain Verified ✓"
    assert reloaded_mgr.is_sequence_verified("General1", "General1_seq_01"), "General1_seq_01 must remain Verified ✓"

    # 9.10 Tkinter GUI headless verification for General2_seq_02
    root = tk.Tk()
    root.withdraw()
    app = AnnotationApp(root, reloaded_mgr)
    app._load_video("General2")
    app.selected_seq_id = "General2_seq_02"
    app._load_quick_phases_for_seq("General2_seq_02")
    assert len(app.current_quick_phases) == 1, "Quick phases should have 1 item for General2_seq_02"
    assert app.current_quick_phases[0]["phase_label"] == "FOLLOW_THROUGH"
    assert app.phase_var.get() == "FOLLOW_THROUGH"
    assert "Verified ✓" in app.quick_seq_lbl.cget("text")
    root.destroy()

    print("  --> PASS: General2_seq_02 correctly verified as FOLLOW_THROUGH (96-99), persisted to CSV, Verified [OK] displayed, existing annotations preserved.")

    print("\n" + "=" * 70)
    print("  ALL SELF-TESTS PASSED SUCCESSFULLY! (Code 0)")
    print("=" * 70 + "\n")
    return True


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CrickSense General Category Manual Annotation Tool")
    parser.add_argument("--test", action="store_true", help="Run automated self-test verification suite without GUI")
    args = parser.parse_args()

    if args.test:
        success = run_self_test()
        sys.exit(0 if success else 1)

    # Launch GUI
    import tkinter as tk
    root = tk.Tk()
    manager = AnnotationDataManager()
    app = AnnotationApp(root, manager)
    root.mainloop()


if __name__ == "__main__":
    main()
