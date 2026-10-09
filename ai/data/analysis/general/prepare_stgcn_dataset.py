import os
import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

LANDMARK_DIR = os.path.join(
    BASE, "ai", "data", "landmarks", "General"
)

ANNOTATION_FILE = os.path.join(
    BASE,
    "ai",
    "data",
    "analysis",
    "general",
    "general_annotations.csv"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "ai",
    "data",
    "analysis",
    "general",
    "general_stgcn_dataset.csv"
)

# ---------------------------------------------------------
# Phase labels
# ---------------------------------------------------------

PHASES = {
    "RUN_UP": 0,
    "GATHER": 1,
    "DELIVERY_STRIDE": 2,
    "RELEASE": 3,
    "FOLLOW_THROUGH": 4,
}

# ---------------------------------------------------------
# Load annotations
# ---------------------------------------------------------

print("Loading annotations...")

annotations = pd.read_csv(ANNOTATION_FILE)

print(f"Annotation rows: {len(annotations)}")

# ---------------------------------------------------------
# Prepare dataset
# ---------------------------------------------------------

rows = []

for video_id, group in annotations.groupby("video_id"):

    landmark_file = os.path.join(
        LANDMARK_DIR,
        f"{video_id}_landmarks.csv"
    )

    print(f"\nProcessing: {video_id}")

    if not os.path.exists(landmark_file):
        print(f"WARNING: Landmark file not found:")
        print(landmark_file)
        continue

    landmarks = pd.read_csv(landmark_file)

    print(f"Landmark frames: {len(landmarks)}")

    for _, ann in group.iterrows():

        start_frame = int(ann["start_frame"])
        end_frame = int(ann["end_frame"])

        phase_label = ann["phase_label"]
        sequence_id = ann["sequence_id"]

        if phase_label not in PHASES:
            print(
                f"WARNING: Unknown phase '{phase_label}' "
                f"for {video_id} {sequence_id}"
            )
            continue

        phase_id = PHASES[phase_label]

        # Select the annotated frame range
        selected = landmarks[
            (landmarks["frame_id"] >= start_frame)
            & (landmarks["frame_id"] <= end_frame)
        ].copy()

        if selected.empty:
            print(
                f"WARNING: No landmark frames found for "
                f"{video_id} {sequence_id} "
                f"{start_frame}-{end_frame}"
            )
            continue

        # Add annotation information
        selected["phase_label"] = phase_label
        selected["phase_id"] = phase_id
        selected["sequence_id"] = sequence_id

        rows.append(selected)

# ---------------------------------------------------------
# Combine everything
# ---------------------------------------------------------

if not rows:
    raise RuntimeError(
        "No labeled landmark frames were found. "
        "Check the annotation and landmark paths."
    )

dataset = pd.concat(
    rows,
    ignore_index=True
)

# ---------------------------------------------------------
# Remove duplicate frames
# ---------------------------------------------------------

dataset = dataset.drop_duplicates(
    subset=[
        "video_id",
        "sequence_id",
        "frame_id"
    ]
)

# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ST-GCN DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nTotal labeled frames: {len(dataset)}")

print("\nPhase distribution:")

phase_counts = (
    dataset["phase_label"]
    .value_counts()
    .sort_index()
)

for phase, count in phase_counts.items():
    print(f"{phase:20s}: {count}")

print("\nVideo distribution:")

video_counts = (
    dataset["video_id"]
    .value_counts()
    .sort_index()
)

for video, count in video_counts.items():
    print(f"{video:10s}: {count}")

print("\nDataset columns:", len(dataset.columns))

print("\nDone.")