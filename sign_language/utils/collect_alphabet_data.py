import cv2
import numpy as np
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# CONFIG
# -----------------------------
ALPHABET = "Z"          # CHANGE THIS (A, B, C, ...)
SAMPLES = 10            # samples to collect this run
FRAMES_PER_SAMPLE = 30  # frames per sample
SAVE_DIR = f"dataset/{ALPHABET}"
MODEL_PATH = "hand_landmarker.task"

os.makedirs(SAVE_DIR, exist_ok=True)

# Find existing samples
existing = [
    int(f.split("_")[1].split(".")[0])
    for f in os.listdir(SAVE_DIR)
    if f.startswith("sample_") and f.endswith(".npy")
]

sample_count = max(existing) + 1 if existing else 0
starting_count = sample_count

# -----------------------------
# MEDIAPIPE HAND LANDMARKER
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# DRAWING CONNECTIONS
# -----------------------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape

    for x, y, _ in landmarks:
        cx, cy = int(x * w), int(y * h)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)

    for s, e in HAND_CONNECTIONS:
        x1, y1 = landmarks[s][:2]
        x2, y2 = landmarks[e][:2]

        cv2.line(
            image,
            (int(x1 * w), int(y1 * h)),
            (int(x2 * w), int(y2 * h)),
            (255, 0, 0),
            2
        )

# -----------------------------
# CAMERA LOOP
# -----------------------------
cap = cv2.VideoCapture(0)

# Bigger window
cv2.namedWindow("Alphabet Data Collection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Alphabet Data Collection", 1280, 720)

recording = False
frames_collected = []

print(f"Collecting alphabet: {ALPHABET}")
print("Press 's' to start recording | 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        landmarks = [(lm.x, lm.y, lm.z) for lm in hand]

        draw_landmarks(frame, landmarks)

        if recording:
            flat = []

            for lm in landmarks:
                flat.extend(lm)

            frames_collected.append(flat)

    cv2.putText(
        frame,
        f"Alphabet: {ALPHABET} | New: {sample_count - starting_count}/{SAMPLES} | Total: {sample_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    if recording:
        cv2.putText(
            frame,
            "RECORDING...",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Alphabet Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and not recording:
        recording = True
        frames_collected = []

        print("Recording started...")
        time.sleep(0.5)

    if recording and len(frames_collected) == FRAMES_PER_SAMPLE:
        data = np.array(frames_collected)  # (30, 63)

        path = os.path.join(
            SAVE_DIR,
            f"sample_{sample_count:03d}.npy"
        )

        np.save(path, data)

        print(f"Saved {path} | shape {data.shape}")

        sample_count += 1
        recording = False
        frames_collected = []

        time.sleep(0.5)

        # Collect SAMPLES new recordings per run
        if sample_count - starting_count >= SAMPLES:
            break

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()