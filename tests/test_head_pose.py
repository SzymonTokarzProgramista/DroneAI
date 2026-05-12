from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import drone_ai.vision.head_pose as head_pose_module
from drone_ai.vision.head_pose import MediaPipeHeadPoseEstimator
from drone_ai.vision.schemas import BoundingBox


class HeadPoseEstimatorTests(unittest.TestCase):
    def test_returns_reason_when_backend_is_unavailable(self) -> None:
        estimator = MediaPipeHeadPoseEstimator(
            enabled=False,
            min_confidence=0.5,
            model_path=Path("models/face_landmarker.task"),
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        result = estimator.estimate(frame, BoundingBox(x=10, y=10, width=80, height=80))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.mesh_ready)
        self.assertFalse(result.pose_ready)
        self.assertEqual(result.failure_reason, "mesh_unavailable")

    def test_returns_model_missing_when_task_model_is_absent(self) -> None:
        estimator = MediaPipeHeadPoseEstimator(
            enabled=True,
            min_confidence=0.5,
            model_path=Path("models/does-not-exist.task"),
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        result = estimator.estimate(frame, BoundingBox(x=10, y=10, width=80, height=80))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.failure_reason, "mesh_unavailable")
        if head_pose_module.mp is None:
            self.assertEqual(result.debug_message, "mediapipe_import_failed")
        else:
            self.assertIn("model_missing:", result.debug_message or "")

    def test_2d_yaw_fallback_returns_value(self) -> None:
        estimator = MediaPipeHeadPoseEstimator(
            enabled=False,
            min_confidence=0.5,
            model_path=Path("models/face_landmarker.task"),
        )
        landmarks = [
            type("Landmark", (), {"x": 0.0, "y": 0.0})()
            for _ in range(300)
        ]
        landmarks[1] = type("Landmark", (), {"x": 0.42, "y": 0.5})()
        landmarks[33] = type("Landmark", (), {"x": 0.18, "y": 0.4})()
        landmarks[263] = type("Landmark", (), {"x": 0.86, "y": 0.4})()

        yaw = estimator._estimate_yaw_2d(landmarks, 100)

        self.assertIsNotNone(yaw)
        assert yaw is not None
        self.assertGreater(yaw, 0.0)

    def test_normalize_landmarks_accepts_tasks_shape(self) -> None:
        landmarks = [
            type("Landmark", (), {"x": 0.1, "y": 0.2})(),
            type("Landmark", (), {"x": 0.3, "y": 0.4})(),
        ]

        normalized = MediaPipeHeadPoseEstimator._normalize_landmarks(landmarks)

        self.assertEqual(normalized, landmarks)

    def test_estimate_confidence_uses_face_landmarks_when_blendshapes_are_missing(self) -> None:
        result = type("Result", (), {"face_landmarks": [[object(), object()]]})()

        confidence = MediaPipeHeadPoseEstimator._estimate_confidence(result)

        self.assertEqual(confidence, 1.0)

    def test_tracking_anchor_y_uses_eye_line(self) -> None:
        estimator = MediaPipeHeadPoseEstimator(
            enabled=False,
            min_confidence=0.5,
            model_path=Path("models/face_landmarker.task"),
        )
        landmarks = [
            type("Landmark", (), {"x": 0.0, "y": 0.0})()
            for _ in range(300)
        ]
        landmarks[33] = type("Landmark", (), {"x": 0.18, "y": 0.30})()
        landmarks[263] = type("Landmark", (), {"x": 0.82, "y": 0.34})()

        anchor_y = estimator._estimate_tracking_anchor_y(
            landmarks,
            roi_height=200,
            offset_y=40,
        )

        self.assertEqual(anchor_y, 104.0)


if __name__ == "__main__":
    unittest.main()
