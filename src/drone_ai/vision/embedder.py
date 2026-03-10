"""Face embedding model wrapper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drone_ai.vision.types import BoundingBox


class SFaceEmbedder:
    """OpenCV SFace embedding wrapper using a local ONNX model file."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise RuntimeError(
                f"SFace model file is missing: {self._model_path}. Download the ONNX model and place it there."
            )

        self._model = cv2.FaceRecognizerSF_create(str(self._model_path), "")
        self._input_size = (112, 112)

    def embed(self, frame_bgr: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
        face_crop = self._crop_face(frame_bgr, bounding_box)
        if face_crop.size == 0:
            raise RuntimeError("Cannot build embedding from an empty face crop.")

        aligned = cv2.resize(face_crop, self._input_size, interpolation=cv2.INTER_LINEAR)
        features = self._model.feature(aligned).reshape(-1).astype(np.float32)

        norm = np.linalg.norm(features)
        if norm == 0.0:
            raise RuntimeError("Face embedder returned a zero vector.")

        return features / norm

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    @staticmethod
    def _crop_face(frame_bgr: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
        margin_x = int(bounding_box.width * 0.15)
        margin_y = int(bounding_box.height * 0.15)
        x0 = max(bounding_box.x - margin_x, 0)
        y0 = max(bounding_box.y - margin_y, 0)
        x1 = min(bounding_box.x + bounding_box.width + margin_x, frame_bgr.shape[1])
        y1 = min(bounding_box.y + bounding_box.height + margin_y, frame_bgr.shape[0])
        return frame_bgr[y0:y1, x0:x1]
