import json
import os
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import torch

from learn_model.learn import GestureMLP, build_distance_rejection, detect_gesture, load_samples
from register_cords.register_cords import get_flat_landmarks


COOLDOWN_SECONDS = 1.5


@dataclass(frozen=True)
class RuntimeArtifacts:
    model: GestureMLP
    classes: list[str]
    centroids: dict[str, object]
    dist_thresholds: dict[str, float]
    config: dict


@dataclass(frozen=True)
class GesturePrediction:
    """Prediction details for one processed video frame."""

    gesture: str = "N/A"
    conf: float = 0.0
    margin: float = 0.0
    dist: float = 0.0


def resolve_repo_dir():
    """Return the repository root based on the current file location."""

    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_runtime_paths(repo_dir):
    """Return the model, config, and training-data paths used by the detector."""

    return (
        os.path.join(repo_dir, "models", "gesture_mlp.pth"),
        os.path.join(repo_dir, "models", "gesture_mlp.config.json"),
        os.path.join(repo_dir, "data", "train"),
    )


def load_config(config_path):
    """Load the saved detector configuration from disk."""

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_to_idx(classes):
    """Map gesture label names to their integer indices."""

    return {name: idx for idx, name in enumerate(classes)}


def load_model(model_path, classes):
    """Instantiate the trained MLP and load its saved weights."""

    model = GestureMLP(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def load_training_samples(train_folder):
    """Load training samples used to rebuild centroid-based rejection rules."""

    train_samples = load_samples(train_folder)
    if not train_samples:
        raise ValueError(f"No training samples found in {train_folder}")
    return train_samples


def resolve_distance_thresholds(classes, config, fallback_thresholds):
    """Prefer saved thresholds and fall back to rebuilt values when needed."""

    return {
        label: float(config.get("distance_thresholds", {}).get(label, fallback_thresholds[label]))
        for label in classes
    }


def load_runtime(repo_dir):
    """Load the trained model, classes, centroids, thresholds, and config."""

    model_path, config_path, train_folder = build_runtime_paths(repo_dir)
    config = load_config(config_path)

    classes = config["classes"]
    label_to_idx = build_label_to_idx(classes)

    model = load_model(model_path, classes)

    train_samples = load_training_samples(train_folder)

    centroids, fallback_thresholds = build_distance_rejection(
        train_samples,
        label_to_idx,
        percentile=float(config.get("distance_percentile", 90.0)),
        scale=float(config.get("distance_scale", 1.0)),
    )

    dist_thresholds = resolve_distance_thresholds(classes, config, fallback_thresholds)

    return RuntimeArtifacts(model, classes, centroids, dist_thresholds, config)


def read_runtime_thresholds(config):
    """Read gesture acceptance thresholds from the saved config."""

    return float(config.get("min_confidence", 0.95)), float(config.get("min_margin", 0.35))


def open_camera():
    """Open the default camera and fail early if it is unavailable."""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    return cap


def prepare_frame(frame):
    """Mirror the frame and convert it to RGB for MediaPipe processing."""

    flipped = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    return flipped, rgb


def predict_from_landmarks(hand, runtime, min_confidence, min_margin):
    """Convert landmarks to normalized coordinates and classify the gesture."""

    coords = get_flat_landmarks(hand)
    if len(coords) != 63:
        return GesturePrediction()

    gesture, conf, margin, dist = detect_gesture(
        runtime.model,
        coords,
        runtime.classes,
        runtime.centroids,
        runtime.dist_thresholds,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )
    return GesturePrediction(gesture, conf, margin, dist)


def detect_frame_gesture(frame, hands, runtime, mp_draw, mp_hands, min_confidence, min_margin):
    """Process a frame, draw the first detected hand, and return prediction data."""

    frame, rgb = prepare_frame(frame)
    results = hands.process(rgb)
    prediction = GesturePrediction()

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        prediction = predict_from_landmarks(hand, runtime, min_confidence, min_margin)

    return frame, prediction


def update_printed_gesture(last_printed_status, last_printed_at, prediction, now):
    """Print status changes, including ``N/A``, while respecting a minimum cooldown."""

    current_status = prediction.gesture

    if current_status == last_printed_status:
        return last_printed_status, last_printed_at

    if (now - last_printed_at) >= COOLDOWN_SECONDS:
        print(
            f"Gesture: {current_status} "
            f"(conf={prediction.conf * 100:.2f}% margin={prediction.margin:.3f} "
            f"dist={prediction.dist:.3f})"
        )
        return current_status, now

    return last_printed_status, last_printed_at


def draw_prediction_overlay(frame, status):
    """Render the last accepted status on top of the video frame."""

    cv2.putText(
        frame,
        f"Gesture: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if status != "N/A" else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def should_close_window(window_name, key):
    """Return True when the user requests exit via ESC or window close."""

    if key == 27:
        return True
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1


def run_detection_loop(cap, runtime):
    """Read camera frames until the detector window is closed."""

    window_name = "Gesture Detector"
    min_confidence, min_margin = read_runtime_thresholds(runtime.config)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    last_printed_status = None
    last_printed_at = 0.0

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, prediction = detect_frame_gesture(
                frame,
                hands,
                runtime,
                mp_draw,
                mp_hands,
                min_confidence,
                min_margin,
            )
            now = time.monotonic()
            last_printed_status, last_printed_at = update_printed_gesture(
                last_printed_status,
                last_printed_at,
                prediction,
                now,
            )

            draw_prediction_overlay(frame, last_printed_status or "N/A")
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if should_close_window(window_name, key):
                break


def shutdown_camera(cap):
    """Release camera resources and destroy any OpenCV windows."""

    cap.release()
    cv2.destroyAllWindows()


def main():
    repo_dir = resolve_repo_dir()
    runtime = load_runtime(repo_dir)
    cap = open_camera()

    try:
        run_detection_loop(cap, runtime)
    finally:
        shutdown_camera(cap)


if __name__ == "__main__":
    main()
