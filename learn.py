import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)

GESTURE_LABEL = "thumbs_up"
sample_count = 0

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

        landmarks_flat = []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # --- NORMALIZACJA ---
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

            for lm in hand.landmark:
                landmarks_flat.extend([
                    (lm.x - ref_x) / scale,
                    (lm.y - ref_y) / scale,
                    (lm.z - ref_z) / scale
                ])

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)

        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break

        if k == 27:
            break

        if k == 115 and len(landmarks_flat) == 63:
            sample_count += 1
            print(f"Save number #{sample_count}: {GESTURE_LABEL}")

            row = ",".join(map(str, landmarks_flat)) + f",{GESTURE_LABEL}\n"
            with open("data.csv", "a") as f:
                f.write(row)

cap.release()
cv2.destroyAllWindows()
