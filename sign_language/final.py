import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

from sentence_corrector import correct_sentence

# CONFIG
MODEL_PATH = "models/alphabet_tcn_best.pth"
HAND_MODEL = "hand_landmarker.task"
SEQ_LEN = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIRM_FRAMES = 6
LETTER_DELAY = 1.2
SENTENCE_GAP = 5.0   
POSE_CHANGE_THRESHOLD = 0.15

# NORMALIZATION

def normalize_landmarks(sequence):
    seq = sequence.reshape(sequence.shape[0], 21, 3)
    wrist = seq[:, 0:1, :]
    seq = seq - wrist
    palm = seq[:, 9, :]
    scale = np.linalg.norm(palm, axis=1, keepdims=True) + 1e-6
    seq = seq / scale[:, None, :]
    return seq.reshape(sequence.shape[0], -1)

# POSE CHANGE CHECK
def pose_changed(current, previous):
    if previous is None:
        return True
    diff = np.linalg.norm(np.array(current) - np.array(previous))
    return diff > POSE_CHANGE_THRESHOLD

# MODEL
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

model = AlphabetTCN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

CLASSES = [chr(ord("A") + i) for i in range(26)]

# MEDIAPIPE

base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

# STATE
frame_buffer = deque(maxlen=SEQ_LEN)
vote_buffer = deque(maxlen=5)

current_word = ""
raw_sentence = ""
fixed_sentence = ""

last_letter = ""
last_committed_letter = ""
last_committed_landmarks = None
confirm_count = 0
next_allowed_time = 0.0

hand_present = False
last_hand_seen_time = time.time()

# CAMERA LOOP
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

    current_time = time.time()
    prediction = ""

    # HAND STATE LOGIC
    if result.hand_landmarks:
        if not hand_present:
            hand_present = True
        last_hand_seen_time = current_time

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

            if current_time >= next_allowed_time:
                if prediction == last_letter:
                    confirm_count += 1
                else:
                    confirm_count = 1
                    last_letter = prediction

                if confirm_count >= CONFIRM_FRAMES:
                    if prediction != last_committed_letter and pose_changed(
                        landmarks, last_committed_landmarks
                    ):
                        current_word += prediction
                        last_committed_letter = prediction
                        last_committed_landmarks = landmarks.copy()
                        next_allowed_time = current_time + LETTER_DELAY

                    confirm_count = 0
                    last_letter = ""

    else:
        if hand_present:
            if current_word:
                raw_sentence += current_word + " "
            current_word = ""

            last_committed_letter = ""
            last_committed_landmarks = None
            last_letter = ""
            confirm_count = 0
            vote_buffer.clear()
            frame_buffer.clear()

            hand_present = False

        if raw_sentence and (current_time - last_hand_seen_time) >= SENTENCE_GAP:
            fixed_sentence = correct_sentence(raw_sentence.strip())
            raw_sentence = ""
            last_hand_seen_time = current_time

    # DISPLAY
    cv2.putText(frame, f"Letter: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Word: {current_word}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"Raw: {raw_sentence}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,255), 2)

    cv2.putText(frame, f"Fixed: {fixed_sentence}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Final ASL System (Hand-State Driven)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
