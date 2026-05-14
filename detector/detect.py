import json
import os

import cv2
import mediapipe as mp
import torch

from learn_model.learn import GestureMLP, build_distance_rejection, detect_gesture, load_samples
from register_cords.register_cords import get_flat_landmarks


def load_runtime(repo_dir):
    model_path = os.path.join(repo_dir, "models", "gesture_mlp.pth")
    config_path = os.path.join(repo_dir, "models", "gesture_mlp.config.json")
    train_folder = os.path.join(repo_dir, "data", "train")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    classes = config["classes"]
    label_to_idx = {name: idx for idx, name in enumerate(classes)}

    model = GestureMLP(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    train_samples = load_samples(train_folder)
    if not train_samples:
        raise ValueError(f"No training samples found in {train_folder}")

    centroids, fallback_thresholds = build_distance_rejection(
        train_samples,
        label_to_idx,
        percentile=float(config.get("distance_percentile", 90.0)),
        scale=float(config.get("distance_scale", 1.0)),
    )

    # Prefer thresholds saved with the model; fallback keeps runtime robust.
    dist_thresholds = {
        label: float(config.get("distance_thresholds", {}).get(label, fallback_thresholds[label]))
        for label in classes
    }

    return model, classes, centroids, dist_thresholds, config


def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model, classes, centroids, dist_thresholds, config = load_runtime(repo_dir)

    min_confidence = float(config.get("min_confidence", 0.95))
    min_margin = float(config.get("min_margin", 0.35))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    last_printed_gesture = None

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = "N/A"
            conf = 0.0
            margin = 0.0
            dist = 0.0

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                coords = get_flat_landmarks(hand)
                if len(coords) == 63:
                    gesture, conf, margin, dist = detect_gesture(
                        model,
                        coords,
                        classes,
                        centroids,
                        dist_thresholds,
                        min_confidence=min_confidence,
                        min_margin=min_margin,
                    )

            if gesture == "N/A":
                last_printed_gesture = None
            elif gesture != last_printed_gesture:
                print(
                    f"Gesture: {gesture} "
                    f"(conf={conf * 100:.2f}% margin={margin:.3f} dist={dist:.3f})"
                )
                last_printed_gesture = gesture

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if gesture != "N/A" else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gesture Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if cv2.getWindowProperty("Gesture Detector", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
