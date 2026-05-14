import math
import os

import cv2
import mediapipe as mp

GESTURE_LABEL = "thumps_down"
FILE_NAME = f"{GESTURE_LABEL}.csv"
FINGER_COLORS = {
    0: (255, 255, 255),  # wrist - white
    1: (255, 0, 0), 2: (255, 0, 0), 3: (255, 0, 0), 4: (255, 0, 0),  # thumb - red
    5: (0, 255, 0), 6: (0, 255, 0), 7: (0, 255, 0), 8: (0, 255, 0),  # index - green
    9: (0, 0, 255), 10: (0, 0, 255), 11: (0, 0, 255), 12: (0, 0, 255),  # middle - blue
    13: (255, 255, 0), 14: (255, 255, 0), 15: (255, 255, 0), 16: (255, 255, 0),  # ring - yellow
    17: (255, 0, 255), 18: (255, 0, 255), 19: (255, 0, 255), 20: (255, 0, 255)  # pinky - purple
}


def setup_camera_window():
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(
        "frame",
        cv2.WND_PROP_TOPMOST,
        cv2.WINDOW_NORMAL
    )
    return cv2.VideoCapture(0)


def get_flat_landmarks(hand):
    landmarks_flat = []

    wrist = hand.landmark[0]
    ref_x, ref_y, ref_z = wrist.x, wrist.y, wrist.z

    index_mcp = hand.landmark[5]
    pinky_mcp = hand.landmark[17]

    scale = math.sqrt(
        (index_mcp.x - pinky_mcp.x) ** 2 +
        (index_mcp.y - pinky_mcp.y) ** 2 +
        (index_mcp.z - pinky_mcp.z) ** 2
    )
    scale = scale if scale > 1e-6 else 1.0

    for landmark in hand.landmark:
        landmarks_flat.extend([
            (landmark.x - ref_x) / scale,
            (landmark.y - ref_y) / scale,
            (landmark.z - ref_z) / scale
        ])

    return landmarks_flat


def draw_hand_landmarks(frame, hand, w, h, mp_draw, mp_hands):
    for idx, lm in enumerate(hand.landmark):
        x = int(lm.x * w)
        y = int(lm.y * h)

        color = FINGER_COLORS[idx]

        cv2.circle(
            frame,
            (x, y),
            7,
            color,
            -1
        )

        cv2.putText(
            frame,
            str(idx),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                               landmark_drawing_spec=mp_draw.DrawingSpec(
                                   color=(255, 255, 255),
                                   thickness=1,
                                   circle_radius=1
                               ),
                               )


def maybe_save_sample(k, landmarks_flat, sample_count):
    if k == 115 and len(landmarks_flat) == 63:  # S and hand visible
        sample_count += 1
        print(f"Save number #{sample_count}: {GESTURE_LABEL}")

        row = ",".join(map(str, landmarks_flat)) + f",{GESTURE_LABEL}\n"
        with open(FILE_NAME, "a") as f:
            f.write(row)
        f.close()

    return sample_count


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = setup_camera_window()

    sample_count = count_rows(FILE_NAME)

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            h, w, _ = frame.shape
            landmarks_flat = []

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks_flat = get_flat_landmarks(hand)
                draw_hand_landmarks(frame, hand, w, h, mp_draw, mp_hands)

            cv2.imshow("frame", frame)
            k = cv2.waitKey(1)

            if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
                break

            if k == 27:  # ESC
                break

            sample_count = maybe_save_sample(k, landmarks_flat, sample_count)
    cap.release()
    cv2.destroyAllWindows()


def count_rows(filename):
    if not os.path.exists(filename):
        return 0
    with open(filename, "r") as f:
        return sum(1 for _ in f)
    return None


if __name__ == "__main__":
    main()
