"""Head pose estimation for tracked faces using MediaPipe Tasks Face Landmarker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drone_ai.vision.schemas import BoundingBox

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker,
        FaceLandmarkerOptions,
    )
except ImportError:
    BaseOptions = None
    FaceLandmarker = None
    FaceLandmarkerOptions = None


@dataclass(frozen=True)
class HeadPoseEstimate:
    mesh_ready: bool
    pose_ready: bool
    yaw_deg: float | None
    pitch_deg: float | None
    confidence: float
    failure_reason: str | None
    debug_message: str | None
    yaw_source: str | None
    mesh_points: tuple[tuple[int, int], ...]


class MediaPipeHeadPoseEstimator:
    """Estimates head yaw and pitch from a face ROI."""

    _LANDMARK_INDEX = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "mouth_left": 61,
        "mouth_right": 291,
    }

    _MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        *,
        enabled: bool,
        min_confidence: float,
        model_path: Path,
    ) -> None:
        self._enabled = enabled and mp is not None
        self._min_confidence = min_confidence
        self._model_path = Path(model_path)
        self._init_debug = "ok"
        self._landmarker = self._create_landmarker() if self._enabled else None
        if not enabled:
            self._init_debug = "disabled_by_config"
        elif mp is None:
            self._init_debug = "mediapipe_import_failed"
        elif self._landmarker is None and self._init_debug == "ok":
            self._init_debug = "landmarker_creation_returned_none"

    @property
    def enabled(self) -> bool:
        return self._landmarker is not None

    def estimate(
        self,
        frame_bgr: np.ndarray,
        face_box: BoundingBox,
        *,
        include_mesh_points: bool = True,
    ) -> HeadPoseEstimate | None:
        if self._landmarker is None:
            return HeadPoseEstimate(
                mesh_ready=False,
                pose_ready=False,
                yaw_deg=None,
                pitch_deg=None,
                confidence=0.0,
                failure_reason="mesh_unavailable",
                debug_message=self._init_debug,
                yaw_source=None,
                mesh_points=(),
            )

        roi_info = self._extract_face_roi(frame_bgr, face_box)
        if roi_info is None:
            return HeadPoseEstimate(
                mesh_ready=False,
                pose_ready=False,
                yaw_deg=None,
                pitch_deg=None,
                confidence=0.0,
                failure_reason="empty_roi",
                debug_message=f"empty_roi box={face_box.width}x{face_box.height}",
                yaw_source=None,
                mesh_points=(),
            )
        roi, offset_x, offset_y = roi_info
        landmarker_roi, scale_x, scale_y = self._prepare_landmarker_roi(roi)

        roi_rgb = cv2.cvtColor(landmarker_roi, cv2.COLOR_BGR2RGB)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
            result = self._landmarker.detect(mp_image)
        except Exception as exc:
            return HeadPoseEstimate(
                mesh_ready=False,
                pose_ready=False,
                yaw_deg=None,
                pitch_deg=None,
                confidence=0.0,
                failure_reason="mesh_process_failed",
                debug_message=f"{type(exc).__name__}: {exc}",
                yaw_source=None,
                mesh_points=(),
            )
        landmarks_list = getattr(result, "face_landmarks", None)
        if not landmarks_list:
            return HeadPoseEstimate(
                mesh_ready=False,
                pose_ready=False,
                yaw_deg=None,
                pitch_deg=None,
                confidence=0.0,
                failure_reason="no_landmarks",
                debug_message=f"roi={landmarker_roi.shape[1]}x{landmarker_roi.shape[0]}",
                yaw_source=None,
                mesh_points=(),
            )

        landmarks_obj = landmarks_list[0]
        landmarks = landmarks_obj.landmark if hasattr(landmarks_obj, "landmark") else landmarks_obj
        if not landmarks:
            return HeadPoseEstimate(
                mesh_ready=False,
                pose_ready=False,
                yaw_deg=None,
                pitch_deg=None,
                confidence=0.0,
                failure_reason="no_landmarks",
                debug_message="landmarks_empty",
                yaw_source=None,
                mesh_points=(),
            )
        confidence = self._estimate_confidence(result)
        if confidence < self._min_confidence:
            return HeadPoseEstimate(
                mesh_ready=False,
                pose_ready=False,
                yaw_deg=None,
                pitch_deg=None,
                confidence=confidence,
                failure_reason="low_confidence",
                debug_message=f"confidence={confidence:.3f} < min={self._min_confidence:.3f}",
                yaw_source=None,
                mesh_points=(),
            )

        image_points = []
        roi_height, roi_width = roi.shape[:2]
        proc_height, proc_width = landmarker_roi.shape[:2]
        for key in (
            "nose_tip",
            "chin",
            "left_eye_outer",
            "right_eye_outer",
            "mouth_left",
            "mouth_right",
        ):
            landmark = landmarks[self._LANDMARK_INDEX[key]]
            image_points.append(
                (
                    float(landmark.x * proc_width * scale_x),
                    float(landmark.y * proc_height * scale_y),
                )
            )

        mesh_points: tuple[tuple[int, int], ...] = ()
        if include_mesh_points:
            mesh_points = tuple(
                (
                    int(round(offset_x + (landmark.x * proc_width * scale_x))),
                    int(round(offset_y + (landmark.y * proc_height * scale_y))),
                )
                for landmark in landmarks
            )
        rotation = self._solve_pose(
            np.array(image_points, dtype=np.float64),
            roi_width,
            roi_height,
        )
        failure_reason = None
        yaw_source = "pnp"
        if rotation is None:
            yaw_deg = self._estimate_yaw_2d(landmarks, roi_width)
            pitch_deg = None
            failure_reason = "pnp_failed"
            yaw_source = "mesh_2d"
            debug_message = (
                f"fallback={yaw_source}"
                if yaw_deg is not None
                else "fallback_failed=no_2d_yaw"
            )
        else:
            yaw_deg, pitch_deg = rotation
            debug_message = f"source={yaw_source}"

        return HeadPoseEstimate(
            mesh_ready=True,
            pose_ready=yaw_deg is not None,
            yaw_deg=float(yaw_deg) if yaw_deg is not None else None,
            pitch_deg=float(pitch_deg) if pitch_deg is not None else None,
            confidence=confidence,
            failure_reason=failure_reason,
            debug_message=debug_message,
            yaw_source=yaw_source if yaw_deg is not None else None,
            mesh_points=mesh_points,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()

    def _create_landmarker(self) -> Any | None:
        if mp is None:
            self._init_debug = "mediapipe_module_missing"
            return None

        if BaseOptions is None or FaceLandmarker is None or FaceLandmarkerOptions is None:
            self._init_debug = "face_landmarker_tasks_unavailable"
            return None

        if not self._model_path.exists():
            self._init_debug = f"model_missing:{self._model_path}"
            return None

        try:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._model_path)),
                num_faces=1,
                min_face_detection_confidence=min(self._min_confidence, 0.35),
                min_face_presence_confidence=min(self._min_confidence, 0.35),
                min_tracking_confidence=min(self._min_confidence, 0.35),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            return FaceLandmarker.create_from_options(options)
        except Exception as exc:
            self._init_debug = f"task_init_failed:{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def _extract_face_roi(
        frame_bgr: np.ndarray,
        face_box: BoundingBox,
    ) -> tuple[np.ndarray, int, int] | None:
        frame_height, frame_width = frame_bgr.shape[:2]
        pad_x = int(face_box.width * 0.35)
        pad_y = int(face_box.height * 0.4)
        x1 = max(face_box.x - pad_x, 0)
        y1 = max(face_box.y - pad_y, 0)
        x2 = min(face_box.x + face_box.width + pad_x, frame_width)
        y2 = min(face_box.y + face_box.height + pad_y, frame_height)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2], x1, y1

    
    @staticmethod
    def _prepare_landmarker_roi(roi: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = roi.shape[:2]
        max_side = max(height, width)
        max_supported_side = 320
        if max_side <= max_supported_side:
            return roi, 1.0, 1.0

        scale = max_supported_side / float(max_side)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(roi, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        scale_x = width / float(resized_width)
        scale_y = height / float(resized_height)
        return resized, scale_x, scale_y

    @staticmethod
    def _estimate_confidence(result: Any) -> float:
        # Tasks API returns `face_landmarks`; legacy solution API used `multi_face_landmarks`.
        tasks_faces = getattr(result, "face_landmarks", None)
        solution_faces = getattr(result, "multi_face_landmarks", None)
        if not tasks_faces and not solution_faces:
            return 0.0
        # Neither API exposes a robust per-face confidence for this model output.
        return 1.0

    def _solve_pose(
        self,
        image_points: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float] | None:
        focal_length = frame_width
        camera_matrix = np.array(
            [
                [focal_length, 0.0, frame_width / 2.0],
                [0.0, focal_length, frame_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vector, _ = cv2.solvePnP(
            self._MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.degrees(math.atan2(-rotation_matrix[2, 1], rotation_matrix[2, 2]))
            yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
        else:
            pitch = math.degrees(math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
            yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))

        return float(np.clip(yaw, -90.0, 90.0)), float(np.clip(pitch, -90.0, 90.0))

    def _estimate_yaw_2d(self, landmarks: list[Any], roi_width: int) -> float | None:
        if roi_width <= 0:
            return None

        nose = landmarks[self._LANDMARK_INDEX["nose_tip"]]
        left_eye = landmarks[self._LANDMARK_INDEX["left_eye_outer"]]
        right_eye = landmarks[self._LANDMARK_INDEX["right_eye_outer"]]

        left_dist = abs(float(nose.x - left_eye.x))
        right_dist = abs(float(right_eye.x - nose.x))
        baseline = left_dist + right_dist
        if baseline <= 1e-6:
            return None

        balance = (right_dist - left_dist) / baseline
        # Practical, bounded 2D fallback when solvePnP is unstable.
        return float(np.clip(balance * 90.0, -45.0, 45.0))
