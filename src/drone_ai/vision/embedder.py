"""Face embedding model wrapper."""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np

from drone_ai.vision.schemas import BoundingBox

try:
    import mediapipe as mp
except ImportError:
    mp = None


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
        self._face_mesh = self._create_face_mesh()
        self._template_landmarks = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

    def embed(self, frame_bgr: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
        face_chip = self._extract_face_chip(frame_bgr, bounding_box)
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

    def _extract_face_chip(self, frame_bgr: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
        face_crop = self._crop_face(frame_bgr, bounding_box)
        if face_crop.size == 0:
            return face_crop

        aligned = self._align_face(face_crop)
        if aligned is not None:
            return aligned

        square_crop = self._square_crop(face_crop)
        return cv2.resize(square_crop, self._input_size, interpolation=cv2.INTER_LINEAR)

    def _align_face(self, face_crop: np.ndarray) -> np.ndarray | None:
        if self._face_mesh is None:
            return None

        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb_crop)
        if not result.multi_face_landmarks:
            return None

        face_landmarks = result.multi_face_landmarks[0]
        crop_height, crop_width = face_crop.shape[:2]
        source = np.array(
            [
                self._landmark_to_point(face_landmarks.landmark[33], crop_width, crop_height),
                self._landmark_to_point(face_landmarks.landmark[263], crop_width, crop_height),
                self._landmark_to_point(face_landmarks.landmark[1], crop_width, crop_height),
                self._landmark_to_point(face_landmarks.landmark[61], crop_width, crop_height),
                self._landmark_to_point(face_landmarks.landmark[291], crop_width, crop_height),
            ],
            dtype=np.float32,
        )

        transform, _ = cv2.estimateAffinePartial2D(source, self._template_landmarks)
        if transform is None:
            return None

        return cv2.warpAffine(
            face_crop,
            transform,
            self._input_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    @staticmethod
    def _crop_face(frame_bgr: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
        margin_x = int(bounding_box.width * 0.15)
        margin_y = int(bounding_box.height * 0.15)
        x0 = max(bounding_box.x - margin_x, 0)
        y0 = max(bounding_box.y - margin_y, 0)
        x1 = min(bounding_box.x + bounding_box.width + margin_x, frame_bgr.shape[1])
        y1 = min(bounding_box.y + bounding_box.height + margin_y, frame_bgr.shape[0])
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

    @staticmethod
    def _landmark_to_point(landmark: object, width: int, height: int) -> tuple[float, float]:
        return float(landmark.x * width), float(landmark.y * height)

    @staticmethod
    def _create_face_mesh() -> object | None:
        if mp is None:
            return None

        solutions = getattr(mp, "solutions", None)
        face_mesh_module = getattr(solutions, "face_mesh", None) if solutions is not None else None
        if face_mesh_module is None:
            for module_name in (
                "mediapipe.python.solutions.face_mesh",
                "mediapipe.solutions.face_mesh",
            ):
                try:
                    face_mesh_module = importlib.import_module(module_name)
                    break
                except Exception:
                    continue

        if face_mesh_module is None:
            return None

        try:
            return face_mesh_module.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            return None
