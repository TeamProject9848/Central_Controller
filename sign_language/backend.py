import time
import logging
import threading
import os
from queue import Queue, Full
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sign_language.sentence_corrector import correct_sentence

logger = logging.getLogger(__name__)

# Constants
SEQ_LEN = 30
CONFIRM_FRAMES = 6
LETTER_DELAY = 1.2
SENTENCE_GAP = 5.0
POSE_CHANGE_THRESHOLD = 0.15
CLASSES = [chr(ord("A") + i) for i in range(26)]

# Model and normalisation helpers
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

def normalize_landmarks(sequence):
    seq = sequence.reshape(sequence.shape[0], 21, 3)
    wrist = seq[:, 0:1, :]
    seq = seq - wrist
    palm = seq[:, 9, :]
    scale = np.linalg.norm(palm, axis=1, keepdims=True) + 1e-6
    seq = seq / scale[:, None, :]
    return seq.reshape(sequence.shape[0], -1)

def normalize_single_frame(landmarks_list):
    landmarks = np.array(landmarks_list).reshape(21, 3)
    wrist = landmarks[0, :]
    translated = landmarks - wrist
    palm = translated[9, :]
    scale = np.linalg.norm(palm) + 1e-6
    normalized = translated / scale
    return normalized.flatten()

def pose_changed(current, previous):
    if previous is None:
        return True
    norm_current = normalize_single_frame(current)
    norm_previous = normalize_single_frame(previous)
    diff = np.linalg.norm(norm_current - norm_previous)
    return diff > POSE_CHANGE_THRESHOLD

class SignDetectionBackend:
    def __init__(self):
        self.module_name = "SignDetectionBackend"
        self.is_running = False
        self.is_healthy = True
        self.on_sentence_callback = None
        self.is_active_callback = None

        self._frame_queue = Queue(maxsize=2)
        self._worker_thread = None
        
        self.model = None
        self.landmarker = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset(self):
        """Reset the internal state, buffers, and clear frame queue."""
        logger.info("Resetting SignDetectionBackend state...")
        
        # Clear deques
        if hasattr(self, 'frame_buffer') and self.frame_buffer is not None:
            self.frame_buffer.clear()
        if hasattr(self, 'vote_buffer') and self.vote_buffer is not None:
            self.vote_buffer.clear()

        # Drain frame queue
        if hasattr(self, '_frame_queue') and self._frame_queue is not None:
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    break

        self.current_word = ""
        self.raw_sentence = ""
        self.fixed_sentence = ""

        self.last_letter = ""
        self.last_committed_letter = ""
        self.last_committed_landmarks = None
        self.confirm_count = 0
        self.next_allowed_time = 0.0

        self.hand_present = False
        self.last_hand_seen_time = time.time()
        logger.info("SignDetectionBackend state reset completed")

    def start(self) -> bool:
        if self.is_running:
            logger.warning("SignDetectionBackend already running")
            return True

        logger.info("SignDetectionBackend starting...")
        
        # Resolve paths relative to the backend file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "alphabet_tcn_best.pth")
        hand_model_path = os.path.join(current_dir, "hand_landmarker.task")

        try:
            self.model = AlphabetTCN().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded TCN model from {model_path} on {self.device}")

            base_options = python.BaseOptions(model_asset_path=hand_model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info(f"Loaded MediaPipe landmarker from {hand_model_path}")
        except Exception as e:
            logger.error(f"Failed to load sign detection models/landmarker: {e}", exc_info=True)
            self.is_healthy = False
            return False

        # Initialize/reset buffers and state variables
        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self.vote_buffer = deque(maxlen=5)

        self.current_word = ""
        self.raw_sentence = ""
        self.fixed_sentence = ""

        self.last_letter = ""
        self.last_committed_letter = ""
        self.last_committed_landmarks = None
        self.confirm_count = 0
        self.next_allowed_time = 0.0

        self.hand_present = False
        self.last_hand_seen_time = time.time()

        # Clear queue and start background worker thread
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Exception:
                break

        self.is_running = True
        self.is_healthy = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="SignWorker",
            daemon=True
        )
        self._worker_thread.start()
        logger.info("SignDetectionBackend started successfully")
        return True

    def stop(self):
        if not self.is_running:
            return

        logger.info("SignDetectionBackend stopping...")
        self.is_running = False
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception as e:
                logger.error(f"Error closing landmarker: {e}")
            self.landmarker = None

        self.model = None
        logger.info("SignDetectionBackend stopped cleanly")

    def on_frame(self, frame):
        if not self.is_running:
            return

        if self.is_active_callback and not self.is_active_callback():
            return

        logger.debug("SIGN FRAME RECEIVED")

        if frame is None:
            return

        raw = frame.frame if hasattr(frame, 'frame') else frame

        if raw is None:
            return

        try:
            self._frame_queue.put_nowait(raw)
        except Full:
            pass

    def _worker_loop(self):
        logger.info("SignDetectionBackend worker loop started")
        while self.is_running:
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except Exception:
                # Timeout occurred, check timeouts for sentence completion
                self._check_timeouts()
                continue

            if frame is None:
                continue

            try:
                self._process_frame(frame)
            except Exception as e:
                logger.error(f"Error processing sign frame in worker: {e}", exc_info=True)

        logger.info("SignDetectionBackend worker loop exited")

    def _process_frame(self, frame):
        flipped = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        current_time = time.time()

        if result.hand_landmarks:
            if not self.hand_present:
                self.hand_present = True
            self.last_hand_seen_time = current_time
            logger.debug("HAND DETECTED")

            hand = result.hand_landmarks[0]
            landmarks = []
            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])

            self.frame_buffer.append(landmarks)
            logger.debug(f"BUFFER={len(self.frame_buffer)}/{SEQ_LEN}")

            if len(self.frame_buffer) == SEQ_LEN:
                logger.debug("RUNNING MODEL")
                seq = normalize_landmarks(np.array(self.frame_buffer))
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred_idx = self.model(x).argmax(dim=1).item()

                self.vote_buffer.append(pred_idx)
                prediction = CLASSES[max(set(self.vote_buffer), key=self.vote_buffer.count)]

                if current_time >= self.next_allowed_time:
                    if prediction == self.last_letter:
                        self.confirm_count += 1
                    else:
                        self.confirm_count = 1
                        self.last_letter = prediction

                    if self.confirm_count >= CONFIRM_FRAMES:
                        if prediction != self.last_committed_letter and pose_changed(
                            landmarks, self.last_committed_landmarks
                        ):
                            self.current_word += prediction
                            logger.info(f"Sign confirmed letter: {prediction}, current_word: {self.current_word}")
                            self.last_committed_letter = prediction
                            self.last_committed_landmarks = landmarks.copy()
                            self.next_allowed_time = current_time + LETTER_DELAY
                            # Clear buffer so user has time to transition to next letter
                            self.frame_buffer.clear()
                            self.vote_buffer.clear()

                            # Immediate broadcast of current letter/word update
                            if self.on_sentence_callback:
                                try:
                                    current_text = (self.raw_sentence + self.current_word).strip()
                                    self.on_sentence_callback(current_text)
                                except Exception as cb_err:
                                    logger.error(f"Error calling on_sentence_callback for intermediate letter: {cb_err}", exc_info=True)

                        self.confirm_count = 0
                        self.last_letter = ""
        else:
            if self.hand_present:
                if self.current_word:
                    self.raw_sentence += self.current_word + " "
                    logger.info(f"Sign word completed: {self.current_word}, raw_sentence: {self.raw_sentence}")
                    
                    # Immediate broadcast of word completion
                    if self.on_sentence_callback:
                        try:
                            self.on_sentence_callback(self.raw_sentence.strip())
                        except Exception as cb_err:
                            logger.error(f"Error calling on_sentence_callback for word completion: {cb_err}", exc_info=True)
                            
                self.current_word = ""

                self.last_committed_letter = ""
                self.last_committed_landmarks = None
                self.last_letter = ""
                self.confirm_count = 0
                self.vote_buffer.clear()
                self.frame_buffer.clear()

                self.hand_present = False

            self._check_timeouts(current_time)

    def _check_timeouts(self, current_time=None):
        if current_time is None:
            current_time = time.time()

        if self.raw_sentence and (current_time - self.last_hand_seen_time) >= SENTENCE_GAP:
            raw = self.raw_sentence.strip()
            logger.info(f"Sign sentence completed. Correcting: '{raw}'")
            try:
                fixed = correct_sentence(raw)
            except Exception as e:
                logger.error(f"Error correcting sentence: {e}", exc_info=True)
                fixed = raw

            self.raw_sentence = ""
            self.last_hand_seen_time = current_time

            logger.info(f"Corrected sign sentence: '{fixed}'")
            if self.on_sentence_callback:
                try:
                    self.on_sentence_callback(fixed)
                except Exception as cb_err:
                    logger.error(f"Error calling on_sentence_callback: {cb_err}", exc_info=True)
