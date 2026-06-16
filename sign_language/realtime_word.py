import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# =============================
# CONFIG
# =============================
MODEL_PATH = "models/alphabet_tcn_best.pth"
HAND_MODEL = "hand_landmarker.task"
SEQ_LEN = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIRM_FRAMES = 6        # frames to confirm a letter
LETTER_DELAY = 1.5        # seconds lock between letters
WORD_TIMEOUT = 1.2        # seconds -> space

# =============================
# NORMALIZATION (same as training)
# =============================
def normalize_landmarks(sequence):
    seq = sequence.reshape(sequence.shape[0], 21, 3)
    wrist = seq[:, 0:1, :]
    seq = seq - wrist
    palm = seq[:, 9, :]
    scale = np.linalg.norm(palm, axis=1, keepdims=True) + 1e-6
    seq = seq / scale[:, None, :]
    return seq.reshape(sequence.shape[0], -1)

# =============================
# TCN MODEL
# =============================
class AlphabetTCN(nn.Module):
    def __init__(self, input_dim=63, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)

# =============================
# LOAD MODEL
# =============================
model = AlphabetTCN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

CLASSES = [chr(ord("A") + i) for i in range(26)]

# =============================
# MEDIAPIPE HAND LANDMARKER
# =============================
base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = vision.HandLandmarker.create_from_options(options)

# =============================
# STATE VARIABLES
# =============================
frame_buffer = deque(maxlen=SEQ_LEN)
vote_buffer = deque(maxlen=5)

current_word = ""
sentence = ""

last_letter = ""
last_committed_letter = ""
confirm_count = 0

next_allowed_time = 0.0
countdown_time = 0.0
last_input_time = time.time()

ready_for_same_letter = False   # 🔑 allows double letters

# =============================
# CAMERA LOOP
# =============================
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    prediction = ""
    current_time = time.time()

    # -------------------------
    # NEUTRAL STATE (NO HAND)
    # -------------------------
    if not result.hand_landmarks:
        ready_for_same_letter = True
        confirm_count = 0
        last_letter = ""

    # -------------------------
    # HAND DETECTION
    # -------------------------
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        landmarks = []
        for lm in hand:
            landmarks.extend([lm.x, lm.y, lm.z])

        frame_buffer.append(landmarks)

        if len(frame_buffer) == SEQ_LEN:
            seq = normalize_landmarks(np.array(frame_buffer))
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_idx = model(x).argmax(dim=1).item()

            vote_buffer.append(pred_idx)
            prediction = CLASSES[max(set(vote_buffer), key=vote_buffer.count)]

            # -------------------------
            # TIME-BASED LETTER GATE
            # -------------------------
            if current_time < next_allowed_time:
                countdown_time = next_allowed_time - current_time
            else:
                countdown_time = 0.0

                if prediction == last_letter:
                    confirm_count += 1
                else:
                    confirm_count = 1
                    last_letter = prediction

                if confirm_count >= CONFIRM_FRAMES:
                    if (
                        prediction != last_committed_letter
                        or ready_for_same_letter
                    ):
                        current_word += prediction
                        print("Confirmed:", prediction)

                        last_committed_letter = prediction
                        ready_for_same_letter = False
                        last_input_time = current_time

                        next_allowed_time = current_time + LETTER_DELAY

                    confirm_count = 0
                    last_letter = ""

    # -------------------------
    # WORD TIMEOUT → SPACE
    # -------------------------
    if current_word and (current_time - last_input_time) > WORD_TIMEOUT:
        sentence += current_word + " "
        current_word = ""
        last_committed_letter = ""
        ready_for_same_letter = False

    # =============================
    # DISPLAY
    # =============================
    cv2.putText(frame, f"Letter: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.putText(frame, f"Word: {current_word}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)

    cv2.putText(frame, f"Sentence: {sentence}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 200), 2)

    if countdown_time > 0:
        cv2.putText(frame,
                    f"Next letter in: {countdown_time:.1f}s",
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)

    cv2.imshow("ASL Word Formation (Final)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
