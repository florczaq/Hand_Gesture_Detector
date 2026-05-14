import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def load_samples(folder_path):
    """Load `(coords, label)` samples from all CSV files in a folder.

    Malformed rows, rows without labels, and rows that do not contain exactly
    63 numeric coordinate values are skipped silently.
    """

    samples = []
    for csv_path in sorted(glob.glob(os.path.join(folder_path, "*.csv"))):
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                # Ignore incomplete rows so partially written files do not fail training.
                if len(row) < 2:
                    continue
                label = row[-1].strip()
                if not label:
                    continue
                try:
                    coords = np.asarray(row[:-1], dtype=np.float32)
                except ValueError:
                    continue
                if coords.size == 63:
                    samples.append((coords, label))
    return samples


class GestureCSVDataset(Dataset):
    """PyTorch dataset wrapper for labeled gesture landmark samples."""

    def __init__(self, samples, label_to_idx):
        """Keep only samples whose labels are present in the provided mapping."""

        self.samples = [(x, label_to_idx[y]) for x, y in samples if y in label_to_idx]

    def __len__(self):
        """Return the number of usable training samples."""

        return len(self.samples)

    def __getitem__(self, idx):
        """Return one sample as float features and a long integer class id."""

        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class GestureMLP(nn.Module):
    """Simple fully connected classifier for 63-value hand landmark vectors."""

    def __init__(self, num_classes):
        """Build a small multi-layer perceptron with ReLU activations."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """Run a forward pass and return unnormalized class logits."""

        return self.net(x)


def train(model, loader, epochs=40, lr=0.0008):
    """Train the model in place and print loss progress for each epoch."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}")


def build_distance_rejection(train_samples, label_to_idx, percentile=90.0, scale=1.0):
    """Compute per-class centroids and distance thresholds for rejection.

    Each threshold is derived from the chosen distance percentile and then
    scaled so runtime prediction can reject samples that are too far from the
    training distribution of the predicted class.
    """

    by_label = {label: [] for label in label_to_idx}
    for coords, label in train_samples:
        if label in by_label:
            by_label[label].append(coords)

    centroids = {}
    thresholds = {}
    for label, vectors in by_label.items():
        arr = np.asarray(vectors, dtype=np.float32)
        centroid = arr.mean(axis=0)
        dists = np.linalg.norm(arr - centroid, axis=1)
        threshold = float(np.percentile(dists, percentile) * scale + 1e-6)
        centroids[label] = centroid
        thresholds[label] = threshold

    return centroids, thresholds


def detect_gesture(model, coords_63, idx_to_label, centroids, dist_thresholds,
                   min_confidence=0.9, min_margin=0.3):
    """Predict a gesture label or return `"N/A"` when checks fail.

    A prediction is accepted only if the top class clears the confidence,
    confidence-margin, and centroid-distance thresholds.
    """

    x = torch.tensor(coords_63, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)

    top_values, top_indices = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
    conf = float(top_values[0, 0].item())
    second_conf = float(top_values[0, 1].item()) if probs.shape[1] > 1 else 0.0
    margin = conf - second_conf
    pred_idx = int(top_indices[0, 0].item())
    pred_label = idx_to_label[pred_idx]

    centroid = centroids[pred_label]
    dist = float(np.linalg.norm(np.asarray(coords_63, dtype=np.float32) - centroid))
    # Fall back to the null label when the sample is uncertain or too far away.
    if conf >= min_confidence and margin >= min_margin and dist <= dist_thresholds[pred_label]:
        return pred_label, conf, margin, dist
    return "N/A", conf, margin, dist


def test_and_print(model, test_samples, idx_to_label, centroids, dist_thresholds,
                   min_confidence=0.9, min_margin=0.3):
    """Run inference on test samples and print a compact per-sample report."""

    if not test_samples:
        print("No test samples found.")
        return

    for i, (coords, actual_label) in enumerate(test_samples, start=1):
        predicted_label, conf, margin, dist = detect_gesture(
            model,
            coords,
            idx_to_label,
            centroids,
            dist_thresholds,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
        print(
            f"#{i} pred={predicted_label} actual={actual_label} "
            f"conf={conf * 100:.2f}% margin={margin:.3f} dist={dist:.3f}"
        )


def save_model_config(config_path, classes, min_confidence, min_margin,
                      dist_thresholds, dist_percentile, dist_scale):
    """Persist runtime settings needed by the live detector."""

    config = {
        "input_size": 63,
        "classes": classes,
        "null_label": "N/A",
        "min_confidence": min_confidence,
        "min_margin": min_margin,
        "distance_thresholds": dist_thresholds,
        "distance_percentile": dist_percentile,
        "distance_scale": dist_scale,
        "architecture": [63, 128, 64, 32],
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    """Train the classifier from `data/`, then save artifacts under `models/`."""

    train_folder = "data/train"
    test_folder = "data/test"
    model_path = os.path.join("models", "gesture_mlp.pth")
    config_path = os.path.join("models", "gesture_mlp.config.json")

    MIN_CONFIDENCE = 0.95
    MIN_MARGIN = 0.35
    DIST_PERCENTILE = 90.0
    DIST_SCALE = 0.97

    train_samples = load_samples(train_folder)
    test_samples = load_samples(test_folder)
    if not train_samples:
        raise ValueError(f"No training samples found in {train_folder}")

    # The classifier is intended for multiple gestures plus the implicit N/A case.
    classes = sorted({label for _, label in train_samples})
    if len(classes) < 2:
        raise ValueError("Need at least 2 gesture classes in training data.")

    label_to_idx = {name: idx for idx, name in enumerate(classes)}
    dataset = GestureCSVDataset(train_samples, label_to_idx)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = GestureMLP(num_classes=len(classes))
    train(model, loader)

    # Store class centroids and thresholds so runtime detection can reject outliers.
    centroids, dist_thresholds = build_distance_rejection(
        train_samples,
        label_to_idx,
        percentile=DIST_PERCENTILE,
        scale=DIST_SCALE,
    )
    print("Distance thresholds:", json.dumps(dist_thresholds, indent=2))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    save_model_config(
        config_path,
        classes,
        MIN_CONFIDENCE,
        MIN_MARGIN,
        dist_thresholds,
        DIST_PERCENTILE,
        DIST_SCALE,
    )

    print(f"Model saved as {model_path}")
    print(f"Model config saved as {config_path}")

    print("\n--- TESTING ---")
    test_and_print(
        model,
        test_samples,
        classes,
        centroids,
        dist_thresholds,
        min_confidence=MIN_CONFIDENCE,
        min_margin=MIN_MARGIN,
    )
