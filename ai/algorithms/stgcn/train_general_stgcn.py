"""
CrickSense - General-only ST-GCN training script.

IMPORTANT:
Fill GENERAL_LABELS with the REAL labels for your 10 General videos.
Do not guess the labels.

Run from the CrickSense repository root:
    python ai/algorithms/stgcn/train_general_stgcn.py
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from algorithms.stgcn.graph import MediaPipeGraph
from algorithms.stgcn.model import STGCNModel


# ============================================================
# ENTER YOUR REAL LABELS HERE
# Example ONLY:
#   0 = class 0
#   1 = class 1
# Replace these with your actual General labels.
# ============================================================
GENERAL_LABELS = {
    # "General1": 0,
    # "General2": 1,
    # "General3": 0,
    # "S_v1": 1,
    # "S_v2": 0,
    # "S_v3": 1,
    # "S_v4": 0,
    # "S_v5": 1,
    # "S_v6": 0,
    # "S_v7": 1,
}

NUM_CLASSES = 2
CHANNELS = 3
NUM_FRAMES = 80
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 0.001
TEST_RATIO = 0.20
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_csv(csv_path):
    """Load x,y,z landmarks as (T,33,3)."""
    frames = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if not row:
                continue

            detected = int(row[2])
            raw = row[3:]

            if len(raw) != 132:
                raise ValueError(
                    f"{csv_path}: expected 132 landmark values, got {len(raw)}"
                )

            joints = []
            for j in range(33):
                i = j * 4
                if detected == 1:
                    xyz = [float(raw[i]), float(raw[i+1]), float(raw[i+2])]
                else:
                    xyz = [0.0, 0.0, 0.0]
                joints.append(xyz)

            frames.append(joints)

    return np.asarray(frames, dtype=np.float32)


def resize_sequence(seq, target=80):
    """Resample (T,V,C) to exactly (80,V,C)."""
    T, V, C = seq.shape
    if T == target:
        return seq

    old = np.linspace(0, 1, T)
    new = np.linspace(0, 1, target)
    out = np.zeros((target, V, C), dtype=np.float32)

    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(new, old, seq[:, v, c])

    return out


class GeneralDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        seq, label, name = self.samples[i]
        # (T,V,C) -> (C,T,V)
        x = np.transpose(seq, (2, 0, 1)).copy()
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            name,
        )


def load_general(data_dir):
    files = sorted(Path(data_dir).glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found: {data_dir}")

    samples = []

    for path in files:
        name = path.stem.replace("_landmarks", "")

        if name not in GENERAL_LABELS:
            raise ValueError(
                f"Missing label for {name}. Add it to GENERAL_LABELS."
            )

        seq = load_csv(path)
        original_frames = len(seq)
        seq = resize_sequence(seq, NUM_FRAMES)

        samples.append((seq, GENERAL_LABELS[name], name))
        print(
            f"Loaded {name:10s} | frames={original_frames:4d} | "
            f"label={GENERAL_LABELS[name]}"
        )

    return samples


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += loss.item() * y.size(0) if False else 0
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = 100.0 * correct / total if total else 0.0
    return accuracy, correct, total


def main():
    set_seed(SEED)

    # Repository root = cricksense/
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "ai" / "data" / "landmarks" / "General"

    print("=" * 70)
    print("CrickSense - GENERAL ONLY ST-GCN TRAINING")
    print("=" * 70)

    if len(GENERAL_LABELS) != 10:
        raise ValueError(
            "GENERAL_LABELS must contain all 10 General videos "
            "before training can start."
        )

    samples = load_general(data_dir)

    # Split by VIDEO, not by individual frames.
    random.Random(SEED).shuffle(samples)
    test_count = max(1, round(len(samples) * TEST_RATIO))

    test_samples = samples[:test_count]
    train_samples = samples[test_count:]

    print("\nTraining videos:", [s[2] for s in train_samples])
    print("Testing videos :", [s[2] for s in test_samples])

    train_loader = DataLoader(
        GeneralDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        GeneralDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    graph = MediaPipeGraph(strategy="spatial")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STGCNModel(
        in_channels=CHANNELS,
        num_classes=NUM_CLASSES,
        graph_adjacency=graph.A,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nDevice: {device}")
    print(f"Epochs: {EPOCHS}\n")

    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0
        running_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total
        test_acc, _, _ = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Loss: {running_loss/total:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    model_dir = project_root / "ai" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "general_stgcn.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "in_channels": CHANNELS,
            "num_frames": NUM_FRAMES,
            "labels": GENERAL_LABELS,
        },
        model_path,
    )

    final_acc, correct, total = evaluate(model, test_loader, device)

    print("\n" + "=" * 70)
    print("FINAL GENERAL-ONLY RESULTS")
    print("=" * 70)
    print(f"Test samples : {total}")
    print(f"Correct      : {correct}")
    print(f"Accuracy     : {final_acc:.2f}%")
    print(f"Model saved  : {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
