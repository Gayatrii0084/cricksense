import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE = r"D:\BE project\CricketSense\cricksense-main"

DATA_DIR = os.path.join(
    BASE,
    "ai",
    "data",
    "analysis",
    "general"
)
MODEL_DIR = os.path.join(
    BASE,
    "algorithms",
    "stgcn",
    "trained_models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "stgcn_train_windows.npz"
)

VAL_FILE = os.path.join(
    DATA_DIR,
    "stgcn_validation_windows.npz"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "stgcn_test_windows.npz"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "phase_stgcn_best.pt"
)

# ---------------------------------------------------------
# Import ST-GCN
# ---------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(__file__)
)

from model import STGCNModel
from graph import MediaPipeGraph


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_CLASSES = 5
NUM_JOINTS = 33
IN_CHANNELS = 3

BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.001

PHASES = [
    "RUN_UP",
    "GATHER",
    "DELIVERY_STRIDE",
    "RELEASE",
    "FOLLOW_THROUGH"
]


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("5-CLASS CRICKET PHASE ST-GCN TRAINING")
print("=" * 60)

print("Device:", device)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_npz(filename):

    data = np.load(filename)

    X = data["X"]
    y = data["y"]

    print("\nLoaded:", os.path.basename(filename))
    print("X:", X.shape)
    print("y:", y.shape)

    return X, y


X_train, y_train = load_npz(TRAIN_FILE)
X_val, y_val = load_npz(VAL_FILE)
X_test, y_test = load_npz(TEST_FILE)


# ---------------------------------------------------------
# Convert to tensors
# ---------------------------------------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

test_dataset = TensorDataset(
    X_test,
    y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------
graph = MediaPipeGraph(
    strategy="spatial"
)

A = graph.A
# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = STGCNModel(
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    graph_adjacency=A,
    edge_importance_weighting=True
)

model = model.to(device)

print("\nModel created.")
print("Parameters:",
      sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------
# Class weights
# ---------------------------------------------------------

class_counts = np.bincount(
    y_train.numpy(),
    minlength=NUM_CLASSES
)

print("\nTraining class counts:")

for i, count in enumerate(class_counts):
    print(
        f"  {PHASES[i]}: {count}"
    )

# Inverse-frequency weighting.
weights = (
    class_counts.sum()
    /
    (NUM_CLASSES * np.maximum(class_counts, 1))
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print("\nClass weights:")

for i, weight in enumerate(weights):
    print(
        f"  {PHASES[i]}: {weight.item():.4f}"
    )


# ---------------------------------------------------------
# Loss + optimizer
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

def evaluate(loader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            output = model(X)

            loss = criterion(
                output,
                y
            )

            total_loss += (
                loss.item()
                * X.size(0)
            )

            total_samples += X.size(0)

            predictions = torch.argmax(
                output,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    avg_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        np.asarray(all_labels),
        np.asarray(all_predictions)
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

best_f1 = -1.0

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    samples = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
            * X.size(0)
        )

        samples += X.size(0)

    train_loss = (
        running_loss / samples
    )

    (
        val_loss,
        val_accuracy,
        val_f1,
        _,
        _
    ) = evaluate(val_loader)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Val Macro-F1: {val_f1:.4f}"
    )

    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_f1,
                "num_classes": NUM_CLASSES,
                "in_channels": IN_CHANNELS,
                "num_joints": NUM_JOINTS,
                "phases": PHASES
            },
            MODEL_FILE
        )

        print(
            "  -> Best model saved."
        )


# ---------------------------------------------------------
# Load best model
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "Best validation Macro-F1:",
    checkpoint["best_val_f1"]
)


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

(
    test_loss,
    test_accuracy,
    test_f1,
    test_labels,
    test_predictions
) = evaluate(test_loader)


print("\nTEST RESULTS")
print("-" * 40)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro-F1: {test_f1:.4f}"
)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        test_labels,
        test_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=PHASES,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nCONFUSION MATRIX")
print("-" * 60)

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print(
    "             " +
    " ".join(
        f"{p[:5]:>8}"
        for p in PHASES
    )
)

for i, phase in enumerate(PHASES):

    print(
        f"{phase[:5]:>12} " +
        " ".join(
            f"{cm[i, j]:8d}"
            for j in range(NUM_CLASSES)
        )
    )


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Model saved to:"
)

print(
    MODEL_FILE
)