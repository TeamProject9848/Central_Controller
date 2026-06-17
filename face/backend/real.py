import logging
from typing import Any, Dict, Optional, Tuple

from config import FaceConfig
from face.backend.base import FaceBackend

logger = logging.getLogger(__name__)


class RealFaceBackend(FaceBackend):
    """Face alignment + FaceNet embedding backend adapted from the standalone PoC."""

    def __init__(self, device: Optional[str] = None):
        self._device = device or FaceConfig.FACE_BACKEND_DEVICE
        try:
            import face_alignment
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise RuntimeError(
                'Real face backend requires face_alignment and facenet_pytorch. '
                'Install optional face dependencies before enabling FaceConfig.ENABLE_FACE_MODULE.'
            ) from exc

        self._torch = torch
        self._face_alignment = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self._device,
            flip_input=False,
        )
        self._embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
        if self._device != 'cpu':
            self._embedding_model = self._embedding_model.to(self._device)
        logger.info('Real face backend initialized')

    def detect_and_align(self, frame) -> Optional[Any]:
        try:
            import cv2
            import numpy as np

            img = np.asarray(frame)
            if img.size == 0:
                return None
                
            # Convert BGR (from WebRTC) to RGB for face_alignment
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            landmarks_list = self._face_alignment.get_landmarks(img_rgb)
            if landmarks_list is None or len(landmarks_list) == 0:
                return None
            return self._align_and_crop_face(img_rgb, landmarks_list[0], output_size=FaceConfig.FACE_CROP_SIZE)
        except Exception as exc:
            logger.error(f'Face detect/align failed: {exc}', exc_info=True)
            return None

    def extract_embedding(self, aligned_face) -> Optional[Any]:
        try:
            import cv2
            import numpy as np

            face_rgb = cv2.resize(aligned_face, FaceConfig.FACE_CROP_SIZE)
            face_tensor = np.transpose(face_rgb, (2, 0, 1)) / 255.0
            face_tensor = np.expand_dims(face_tensor, 0)
            face_tensor = self._torch.tensor(face_tensor, dtype=self._torch.float32)
            if self._device != 'cpu':
                face_tensor = face_tensor.to(self._device)
            with self._torch.no_grad():
                emb = self._embedding_model(face_tensor).detach().cpu().numpy()[0]
            return emb
        except Exception as exc:
            logger.error(f'Face embedding failed: {exc}', exc_info=True)
            return None

    def evaluate_liveness(self, frame, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        if not FaceConfig.LIVENESS_REQUIRED:
            return True, {'method': 'disabled'}
        return False, {'method': 'deferred', 'reason': 'liveness_not_implemented'}

    def close(self) -> None:
        return

    @staticmethod
    def _align_and_crop_face(img_rgb, landmarks, output_size=(160, 160)):
        import cv2
        import numpy as np

        if len(landmarks) <= 45:
            return None
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        nose = landmarks[30]

        eyes_center = (left_eye + right_eye) / 2.0
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        desired_dist = output_size[0] * 0.35
        actual_dist = np.sqrt(dx * dx + dy * dy)
        if actual_dist == 0:
            return None
        scale = desired_dist / actual_dist

        transform = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
        transform[0, 2] += output_size[0] * 0.5 - nose[0] * scale
        transform[1, 2] += output_size[1] * 0.4 - nose[1] * scale
        return cv2.warpAffine(img_rgb, transform, output_size, flags=cv2.INTER_CUBIC)
