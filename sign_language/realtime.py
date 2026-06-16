import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/alphabet_tcn_best.pth"
HAND_MODEL = "hand_landmarker.task"
SEQ_LEN = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# NORMALIZATION (MUST MATCH TRAINING)
# -----------------------------
def normalize_landmarks(sequence):
    seq = np.array(sequence).reshape(sequence.shape[0], 21, 3)

    wrist = seq[:, 0:1, :]
    seq = seq - wrist

    palm = seq[:, 9, :]
    scale = np.linalg.norm(palm, axis=1, keepdims=True) + 1e-6
    seq = seq / scale[:, None, :]

    return seq.reshape(sequence.shape[0], -1)

# -----------------------------
# TCN MODEL (same as training)
# -----------------------------
class AlphabetTCN(nn.Module):
    def __init__(self, input_dim=63, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = AlphabetTCN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Loaded model:", MODEL_PATH)

# -----------------------------
# LABELS
# -----------------------------
CLASSES = [chr(ord("A") + i) for i in range(26)]

# -----------------------------
# MEDIAPIPE HAND LANDMARKER
# -----------------------------
base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# BUFFERS
# -----------------------------
frame_buffer = deque(maxlen=SEQ_LEN)
vote_buffer = deque(maxlen=5)

# -----------------------------
# CAMERA LOOP
# -----------------------------
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

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
    prediction = ""

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
                logits = model(x)
                pred_idx = logits.argmax(dim=1).item()

            vote_buffer.append(pred_idx)
            final_idx = max(set(vote_buffer), key=vote_buffer.count)
            prediction = CLASSES[final_idx]

    # -----------------------------
    # DISPLAY
    # -----------------------------
    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )

    cv2.imshow("ASL Alphabet Recognition (TCN)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()