"""Face embedding model wrapper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drone_ai.constants.vision import (
    FACE_CHIP_MARGIN_RATIO,
    ONE_SHOT_FACE_CHIP_VARIANTS,
    SFACE_INPUT_SIZE,
)
from drone_ai.vision.schemas import BoundingBox


class SFaceEmbedder:
    """OpenCV SFace embedding wrapper using a local ONNX model file."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise RuntimeError(
                f"SFace model file is missing: {self._model_path}. Download the ONNX model and place it there."
            )

        self._model = cv2.FaceRecognizerSF_create(str(self._model_path), "")
        self._input_size = SFACE_INPUT_SIZE

    def embed(self, frame_bgr: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
        face_chip = self._extract_face_chip(frame_bgr, bounding_box)
        return self._embed_face_chip(face_chip)

    def embed_variants(self, frame_bgr: np.ndarray, bounding_box: BoundingBox) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for margin_ratio, offset_x_ratio, offset_y_ratio in ONE_SHOT_FACE_CHIP_VARIANTS:
            face_chip = self._extract_face_chip(
                frame_bgr,
                bounding_box,
                margin_ratio=margin_ratio,
                offset_x_ratio=offset_x_ratio,
                offset_y_ratio=offset_y_ratio,
            )
            if face_chip.size == 0:
                continue
            embeddings.append(self._embed_face_chip(face_chip))

        if not embeddings:
            raise RuntimeError("Cannot build embeddings from an empty face crop.")

        return embeddings

    def _embed_face_chip(self, face_chip: np.ndarray) -> np.ndarray:
        if face_chip.size == 0:
            raise RuntimeError("Cannot build embedding from an empty face crop.")

        features = self._model.feature(face_chip).reshape(-1).astype(np.float32)

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

    def _extract_face_chip(
        self,
        frame_bgr: np.ndarray,
        bounding_box: BoundingBox,
        *,
        margin_ratio: float = FACE_CHIP_MARGIN_RATIO,
        offset_x_ratio: float = 0.0,
        offset_y_ratio: float = 0.0,
    ) -> np.ndarray:
        face_crop = self._crop_face(
            frame_bgr,
            bounding_box,
            margin_ratio=margin_ratio,
            offset_x_ratio=offset_x_ratio,
            offset_y_ratio=offset_y_ratio,
        )
        if face_crop.size == 0:
            return face_crop

        square_crop = self._square_crop(face_crop)
        return cv2.resize(square_crop, self._input_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _crop_face(
        frame_bgr: np.ndarray,
        bounding_box: BoundingBox,
        *,
        margin_ratio: float = FACE_CHIP_MARGIN_RATIO,
        offset_x_ratio: float = 0.0,
        offset_y_ratio: float = 0.0,
    ) -> np.ndarray:
        if bounding_box.width <= 0 or bounding_box.height <= 0:
            return frame_bgr[0:0, 0:0]

        offset_x = int(round(bounding_box.width * offset_x_ratio))
        offset_y = int(round(bounding_box.height * offset_y_ratio))
        margin_x = int(round(bounding_box.width * margin_ratio))
        margin_y = int(round(bounding_box.height * margin_ratio))
        x0 = max(bounding_box.x + offset_x - margin_x, 0)
        y0 = max(bounding_box.y + offset_y - margin_y, 0)
        x1 = min(
            bounding_box.x + offset_x + bounding_box.width + margin_x,
            frame_bgr.shape[1],
        )
        y1 = min(
            bounding_box.y + offset_y + bounding_box.height + margin_y,
            frame_bgr.shape[0],
        )
        if x1 <= x0 or y1 <= y0:
            return frame_bgr[0:0, 0:0]
        return frame_bgr[y0:y1, x0:x1]

    @staticmethod
    def _square_crop(face_crop: np.ndarray) -> np.ndarray:
        height, width = face_crop.shape[:2]
        if height == width:
            return face_crop

        size = max(height, width)
        top = (size - height) // 2
        bottom = size - height - top
        left = (size - width) // 2
        right = size - width - left
        return cv2.copyMakeBorder(
            face_crop,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )
