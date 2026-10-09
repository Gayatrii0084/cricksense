import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from model import STGCNModel
from graph import MediaPipeGraph


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

MODEL_FILE = os.path.join(
    BASE,
    "algorithms",
    "stgcn",
    "trained_models",
    "phase_stgcn_best.pt"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "stgcn_test_windows.npz"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PHASES = [
    "RUN_UP",
    "GATHER",
    "DELIVERY_STRIDE",
    "RELEASE",
    "FOLLOW_THROUGH"
]

NUM_CLASSES = 5
IN_CHANNELS = 3
BATCH_SIZE = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

data = np.load(TEST_FILE)

X_test = data["X"]
y_test = data["y"]

print("\nLoaded test dataset")
print("X:", X_test.shape)
print("y:", y_test.shape)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------------------------------------------------
# Build graph
# ---------------------------------------------------------

graph = MediaPipeGraph(
    strategy="spatial"
)

A = graph.A


# ---------------------------------------------------------
# Build model
# ---------------------------------------------------------

model = STGCNModel(
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    graph_adjacency=A,
    edge_importance_weighting=True
)

model = model.to(device)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

print("\nLoading trained model:")
print(MODEL_FILE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Best validation Macro-F1:",
    checkpoint.get("best_val_f1", "N/A")
)


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

all_labels = []
all_predictions = []
total_loss = 0.0

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        total_loss += (
            loss.item() * X_batch.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            y_batch.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

test_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)

test_loss = (
    total_loss / len(test_dataset)
)


print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

print(
    f"\nTest Loss: {test_loss:.4f}"
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
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=PHASES,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nCONFUSION MATRIX")
print("-" * 60)

print("Rows = Actual")
print("Columns = Predicted")
print()

print(
    "             " +
    " ".join(
        f"{phase[:8]:>10}"
        for phase in PHASES
    )
)

for i, row in enumerate(cm):

    print(
        f"{PHASES[i][:8]:>10} " +
        " ".join(
            f"{value:>10}"
            for value in row
        )
    )