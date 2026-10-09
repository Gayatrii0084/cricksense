import os
import pandas as pd

print("=== ST-GCN SPLIT SCRIPT STARTED ===")

base = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

input_file = os.path.join(
    base,
    "ai",
    "data",
    "analysis",
    "general",
    "general_stgcn_dataset.csv"
)

output_dir = os.path.join(
    base,
    "ai",
    "data",
    "analysis",
    "general"
)

print("Input:")
print(input_file)

if not os.path.exists(input_file):
    raise FileNotFoundError(input_file)

df = pd.read_csv(input_file)

print("\nTotal frames:", len(df))

videos = sorted(df["video_id"].unique())

print("\nVideos:")
for video in videos:
    print(" ", video)

# ---------------------------------------------------------
# Video-level split
# ---------------------------------------------------------

test_videos = ["s_v2", "s_v6"]
validation_videos = ["General3"]

train_videos = [
    v for v in videos
    if v not in test_videos and v not in validation_videos
]

print("\nTRAIN:")
print(train_videos)

print("\nVALIDATION:")
print(validation_videos)

print("\nTEST:")
print(test_videos)

# ---------------------------------------------------------
# Create datasets
# ---------------------------------------------------------

train = df[df["video_id"].isin(train_videos)].copy()
validation = df[df["video_id"].isin(validation_videos)].copy()
test = df[df["video_id"].isin(test_videos)].copy()

train_file = os.path.join(output_dir, "stgcn_train.csv")
validation_file = os.path.join(output_dir, "stgcn_validation.csv")
test_file = os.path.join(output_dir, "stgcn_test.csv")

train.to_csv(train_file, index=False)
validation.to_csv(validation_file, index=False)
test.to_csv(test_file, index=False)

print("\n=== SPLIT COMPLETE ===")

print("Training frames:", len(train))
print("Validation frames:", len(validation))
print("Test frames:", len(test))

print("\nTraining phases:")
print(train["phase_label"].value_counts().sort_index())

print("\nValidation phases:")
print(validation["phase_label"].value_counts().sort_index())

print("\nTest phases:")
print(test["phase_label"].value_counts().sort_index())

print("\nFiles created:")
print(train_file)
print(validation_file)
print(test_file)

print("\n=== DONE ===")
