import os
import numpy as np
import pandas as pd

print("=== BALANCED ST-GCN WINDOW GENERATION ===")

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

DATA_DIR = os.path.join(
    BASE,
    "ai",
    "data",
    "analysis",
    "general"
)

PHASES = {
    "RUN_UP": 0,
    "GATHER": 1,
    "DELIVERY_STRIDE": 2,
    "RELEASE": 3,
    "FOLLOW_THROUGH": 4,
}

WINDOW_SIZE = 15
HALF_WINDOW = WINDOW_SIZE // 2
NUM_JOINTS = 33

# Extra sampling for rare training phases.
# This does NOT affect validation or test data.
TRAIN_AUGMENT = {
    0: 1,   # RUN_UP
    1: 2,   # GATHER
    2: 5,   # DELIVERY_STRIDE
    3: 8,   # RELEASE
    4: 1,   # FOLLOW_THROUGH
}


def get_landmark_columns():

    columns = []

    for joint in range(NUM_JOINTS):

        columns.extend([
            f"landmark_{joint}_x",
            f"landmark_{joint}_y",
            f"landmark_{joint}_z",
        ])

    return columns


LANDMARK_COLUMNS = get_landmark_columns()


def create_window(sequence_df, center):

    sequence_df = sequence_df.reset_index(drop=True)

    start = center - HALF_WINDOW
    end = center + HALF_WINDOW + 1

    # Normal window.
    if start >= 0 and end <= len(sequence_df):

        window = sequence_df.iloc[start:end]

        values = window[
            LANDMARK_COLUMNS
        ].to_numpy(dtype=np.float32)

    else:

        # Short / boundary window.
        values = sequence_df[
            LANDMARK_COLUMNS
        ].to_numpy(dtype=np.float32)

        values = values.reshape(
            len(values),
            NUM_JOINTS,
            3
        )

        pad_before = max(0, -start)
        pad_after = max(0, end - len(sequence_df))

        values = np.pad(
            values,
            (
                (pad_before, pad_after),
                (0, 0),
                (0, 0)
            ),
            mode="edge"
        )

        values = values[
            :WINDOW_SIZE
        ]

        if len(values) < WINDOW_SIZE:

            values = np.pad(
                values,
                (
                    (0, WINDOW_SIZE - len(values)),
                    (0, 0),
                    (0, 0)
                ),
                mode="edge"
            )

        return np.transpose(
            values,
            (2, 0, 1)
        )

    values = values.reshape(
        WINDOW_SIZE,
        NUM_JOINTS,
        3
    )

    return np.transpose(
        values,
        (2, 0, 1)
    )


def make_windows(csv_file, training=False):

    df = pd.read_csv(csv_file)

    X = []
    y = []

    for (video_id, sequence_id), sequence_df in df.groupby(
        ["video_id", "sequence_id"]
    ):

        sequence_df = sequence_df.sort_values(
            "frame_id"
        ).reset_index(drop=True)

        if len(sequence_df) == 0:
            continue

        frame_ids = sequence_df[
            "frame_id"
        ].to_numpy()

        # Never create a temporal window across missing frames.
        if not np.all(np.diff(frame_ids) == 1):
            continue

        labels = sequence_df[
            "phase_id"
        ].to_numpy()

        for center in range(len(sequence_df)):

            label = int(labels[center])

            window = create_window(
                sequence_df,
                center
            )

            X.append(window)
            y.append(label)

            # Training-only oversampling.
            extra = 0

            if training:
                extra = TRAIN_AUGMENT.get(
                    label,
                    1
                ) - 1

            for _ in range(extra):

                # Small temporal shift where possible.
                shifted_center = center

                if len(sequence_df) > WINDOW_SIZE:

                    shift = np.random.choice(
                        [-2, -1, 1, 2]
                    )

                    shifted_center = max(
                        0,
                        min(
                            len(sequence_df) - 1,
                            center + shift
                        )
                    )

                augmented_window = create_window(
                    sequence_df,
                    shifted_center
                )

                X.append(augmented_window)
                y.append(label)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64)
    )


for split in [
    "train",
    "validation",
    "test"
]:

    csv_file = os.path.join(
        DATA_DIR,
        f"stgcn_{split}.csv"
    )

    print("\n" + "=" * 55)
    print(f"PROCESSING {split.upper()}")
    print(csv_file)

    X, y = make_windows(
        csv_file,
        training=(split == "train")
    )

    output_file = os.path.join(
        DATA_DIR,
        f"stgcn_{split}_windows.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    print("\nWindows:", len(X))
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nClass distribution:")

    for class_id, phase_name in enumerate(PHASES):

        count = int(
            np.sum(y == class_id)
        )

        print(
            f"  {phase_name}: {count}"
        )

print("\n" + "=" * 55)
print("BALANCED WINDOW GENERATION COMPLETE")
print("=" * 55)