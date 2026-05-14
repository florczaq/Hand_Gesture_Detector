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
    samples = []
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    for csv_path in csv_files:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue

                label = row[-1].strip()
                if not label:
                    continue

                try:
                    coords = np.asarray(row[:-1], dtype=np.float32)
                except ValueError:
                    continue

                if coords.size != 63:
                    continue

                samples.append((coords, label))

    return samples


class GestureCSVDataset(Dataset):
    def __init__(self, samples, label_to_idx):
        self.samples = []
        for x, gesture_name in samples:
            if gesture_name in label_to_idx:
                self.samples.append((x, label_to_idx[gesture_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class GestureMLP(nn.Module):
    def __init__(self, num_classes):
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
        return self.net(x)


def train(model, loader, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}")


def predict_scores(model, coords_63):
    x = torch.tensor(coords_63, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top2_vals, top2_idx = torch.topk(probs, k=2, dim=1)

    conf = top2_vals[0, 0].item()
    second_conf = top2_vals[0, 1].item() if probs.shape[1] > 1 else 0.0
    pred_idx = top2_idx[0, 0].item()
    margin = conf - second_conf
    return pred_idx, conf, margin


def build_distance_rejection(train_samples, label_to_idx, percentile=95.0, scale=1.2):
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

        # Use a high percentile from train data so rare-but-valid poses are accepted.
        dist_list = sorted(float(d) for d in dists.tolist())
        p_index = int((percentile / 100.0) * (len(dist_list) - 1))
        percentile_value = dist_list[p_index]
        thresh = percentile_value * scale + 1e-6

        centroids[label] = centroid
        thresholds[label] = thresh

    return centroids, thresholds


def detect_gesture(model, coords_63, idx_to_label, centroids, dist_thresholds,
                   min_confidence=0.75, min_margin=0.20):
    pred_idx, conf, margin = predict_scores(model, coords_63)
    pred_label = idx_to_label[pred_idx]

    centroid = centroids[pred_label]
    dist = float(np.linalg.norm(np.asarray(coords_63, dtype=np.float32) - centroid))
    dist_ok = dist <= dist_thresholds[pred_label]

    if conf >= min_confidence and margin >= min_margin and dist_ok:
        return pred_label, conf, margin, dist
    return "N/A", conf, margin, dist


def test_and_print(model, test_samples, idx_to_label, centroids, dist_thresholds,
                   min_confidence=0.75, min_margin=0.20):
    if not test_samples:
        print("No test samples found.")
        return

    for i, (coords, _) in enumerate(test_samples, start=1):
        predicted_label, conf, margin, dist = detect_gesture(
            model,
            coords,
            idx_to_label,
            centroids,
            dist_thresholds,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
        print(f"#{i} {predicted_label} (conf={conf * 100:.2f}% margin={margin:.3f} dist={dist:.3f})")


def save_model_config(config_path, idx_to_label, min_confidence, min_margin,
                      dist_thresholds, dist_percentile, dist_scale):
    config = {
        "input_size": 63,
        "classes": idx_to_label,
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
    train_folder = "data/train"
    test_folder = "data/test"
    model_path = os.path.join("models", "gesture_mlp.pth")
    config_path = os.path.join("models", "gesture_mlp.config.json")

    MIN_CONFIDENCE = 0.9
    MIN_MARGIN = 0.35
    DIST_PERCENTILE = 95.0
    DIST_SCALE = 1.0

    train_samples = load_samples(train_folder)
    test_samples = load_samples(test_folder)

    if not train_samples:
        raise ValueError(f"No training samples found in {train_folder}")

    label_names = sorted({gesture_name for _, gesture_name in train_samples})
    if len(label_names) < 2:
        raise ValueError("Need at least 2 gesture classes in training data.")

    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    idx_to_label = label_names

    train_dataset = GestureCSVDataset(train_samples, label_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    model = GestureMLP(num_classes=len(label_names))
    train(model, train_loader, epochs=40, lr=0.0008)

    centroids, dist_thresholds = build_distance_rejection(
        train_samples,
        label_to_idx,
        percentile=DIST_PERCENTILE,
        scale=DIST_SCALE,
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    save_model_config(
        config_path,
        idx_to_label,
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
        idx_to_label,
        centroids,
        dist_thresholds,
        min_confidence=MIN_CONFIDENCE,
        min_margin=MIN_MARGIN,
    )
