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
                 annotations_path: str = ANNOTATIONS_CSV_PATH):
        self.landmarks_dir = landmarks_dir
        self.raw_dir = raw_dir
        self.seq_manifest_path = seq_manifest_path
        self.annotations_path = annotations_path

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.annotations_path), exist_ok=True)

        self.videos: List[str] = []
        self.video_metadata: Dict[str, Dict[str, Any]] = {}
        self.sequences: Dict[str, List[Dict[str, Any]]] = {} # video_id -> list of sequences
        self.annotations: List[Dict[str, Any]] = []

        self.discover_data()
        self.load_sequences()
        self.load_annotations()

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
                    self.annotations.append({
                        "video_id": row["video_id"].strip(),
                        "sequence_id": row["sequence_id"].strip(),
                        "start_frame": int(row["start_frame"].strip()),
                        "end_frame": int(row["end_frame"].strip()),
                        "phase_label": row["phase_label"].strip(),
                        "annotator": row.get("annotator", "").strip(),
                        "notes": row.get("notes", "").strip()
                    })
                except (ValueError, KeyError) as e:
                    print(f"[WARN] Skipping corrupted annotation row: {row} ({e})")

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

    def save_annotation(self,
                        video_id: str,
                        sequence_id: str,
                        start_frame: int,
                        end_frame: int,
                        phase_label: str,
                        annotator: str = "",
                        notes: str = "") -> Tuple[bool, str]:
        """Validate and append or update an annotation in general_annotations.csv."""
        is_valid, msg = self.validate_annotation(video_id, sequence_id, start_frame, end_frame, phase_label)
        if not is_valid:
            return False, msg

        # Check if updating an existing annotation for this sequence_id or creating new
        updated = False
        for ann in self.annotations:
            if ann["video_id"] == video_id and ann["sequence_id"] == sequence_id:
                ann["start_frame"] = start_frame
                ann["end_frame"] = end_frame
                ann["phase_label"] = phase_label
                ann["annotator"] = annotator
                ann["notes"] = notes
                updated = True
                break

        if not updated:
            self.annotations.append({
                "video_id": video_id,
                "sequence_id": sequence_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "phase_label": phase_label,
                "annotator": annotator,
                "notes": notes
            })

        self._flush_annotations_to_disk()
        action_verb = "Updated" if updated else "Saved"
        return True, f"{action_verb} annotation for sequence '{sequence_id}' ({phase_label}: {start_frame}-{end_frame})."

    def delete_annotation(self, video_id: str, sequence_id: str) -> Tuple[bool, str]:
        """Remove an annotation matching video_id and sequence_id."""
        initial_count = len(self.annotations)
        self.annotations = [
            ann for ann in self.annotations
            if not (ann["video_id"] == video_id and ann["sequence_id"] == sequence_id)
        ]
        if len(self.annotations) < initial_count:
            self._flush_annotations_to_disk()
            return True, f"Deleted annotation for sequence '{sequence_id}'."
        return False, f"Annotation not found for '{sequence_id}'."

    def _flush_annotations_to_disk(self) -> None:
        """Write current annotations list to CSV file."""
        with open(self.annotations_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
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

        # RIGHT PANEL: Annotation Form
        right_panel = tk.Frame(content_frame, bg=self.BG_CARD, width=320)
        right_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)

        ttk.Label(right_panel, text="Phase Annotation Form", style="Header.TLabel").pack(anchor=tk.W, padx=12, pady=(10, 10))

        form_grid = tk.Frame(right_panel, bg=self.BG_CARD)
        form_grid.pack(fill=tk.X, padx=12)

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
        tk.Label(right_panel, text="Bowling Phase (Mandatory):", font=("Segoe UI", 10, "bold"), bg=self.BG_CARD, fg=self.ACCENT_ORANGE).pack(anchor=tk.W, padx=12, pady=(15, 5))
        self.phase_var = tk.StringVar(value="RUN_UP")
        phase_box = tk.Frame(right_panel, bg=self.BG_INPUT, padx=8, pady=6)
        phase_box.pack(fill=tk.X, padx=12)

        for phase in ALLOWED_PHASES:
            rb = tk.Radiobutton(phase_box, text=phase, value=phase, variable=self.phase_var,
                                bg=self.BG_INPUT, fg=self.TEXT_MAIN, selectcolor=self.BG_DARK,
                                activebackground=self.BG_INPUT, activeforeground=self.ACCENT_BLUE,
                                font=("Segoe UI", 9))
            rb.pack(anchor=tk.W, pady=2)

        # Annotator Name & Notes
        ann_meta_frame = tk.Frame(right_panel, bg=self.BG_CARD)
        ann_meta_frame.pack(fill=tk.X, padx=12, pady=(12, 0))

        tk.Label(ann_meta_frame, text="Annotator Name:", bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(anchor=tk.W)
        self.annotator_var = tk.StringVar(value="Annotator_1")
        tk.Entry(ann_meta_frame, textvariable=self.annotator_var, bg=self.BG_INPUT, fg=self.TEXT_MAIN, insertbackground=self.TEXT_MAIN).pack(fill=tk.X, pady=(2, 8))

        tk.Label(ann_meta_frame, text="Notes / Observations:", bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(anchor=tk.W)
        self.notes_entry = tk.Entry(ann_meta_frame, bg=self.BG_INPUT, fg=self.TEXT_MAIN, insertbackground=self.TEXT_MAIN)
        self.notes_entry.pack(fill=tk.X, pady=(2, 10))

        # Save, Edit, Delete Buttons
        btn_box = tk.Frame(right_panel, bg=self.BG_CARD)
        btn_box.pack(fill=tk.X, padx=12, pady=(10, 5))

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
        for item in self.seq_tree.get_children():
            self.seq_tree.delete(item)

        seqs = self.manager.sequences.get(self.current_video_id, [])
        annotated_seq_ids = {a["sequence_id"]: a["phase_label"] for a in self.manager.annotations if a["video_id"] == self.current_video_id}

        for s in seqs:
            sid = s["sequence_id"]
            rng = f"{s['start_frame']}-{s['end_frame']}"
            if sid in annotated_seq_ids:
                status = f"✓ {annotated_seq_ids[sid]}"
            else:
                status = "Pending"
            self.seq_tree.insert("", tk.END, values=(sid, rng, status))

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

        # Highlight if current frame is in current annotation window
        try:
            sf = int(self.start_frame_var.get())
            ef = int(self.end_frame_var.get())
            if sf <= self.current_frame <= ef:
                cv2.rectangle(frame_rgb, (10, 10), (fw - 10, fh - 10), (255, 180, 50), 3)
                phase_name = self.phase_var.get()
                cv2.putText(frame_rgb, f"ACTIVE WINDOW: {phase_name} ({sf} -> {ef})",
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
            self.status_var.set(f"[SUCCESS] {err_msg}")
            self._refresh_sequence_list()
            self._refresh_annotations_list()
            self._render_current_frame()
            msgbox.showinfo("Saved", err_msg)
        else:
            self.status_var.set(f"[ERROR] {err_msg}")
            msgbox.showerror("Validation Error", err_msg)

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
