from __future__ import annotations

from pathlib import Path
import sys
from threading import RLock
import unittest

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_ai.application import DroneApplication
from drone_ai.config import AppConfig
from drone_ai.tracking.face_tracker import FaceTracker
from drone_ai.vision.schemas import BoundingBox, RecognizedFace


class _DummyController:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []
        self.stop_calls = 0

    def send_rc_control(
        self,
        left_right_velocity: int,
        forward_backward_velocity: int,
        up_down_velocity: int,
        yaw_velocity: int,
    ) -> None:
        self.calls.append(
            (
                left_right_velocity,
                forward_backward_velocity,
                up_down_velocity,
                yaw_velocity,
            )
        )

    def stop_motion(self) -> None:
        self.stop_calls += 1


class _DummyHeadPose:
    def __init__(
        self,
        yaw_deg: float = 24.0,
        pitch_deg: float = -3.0,
        *,
        mesh_ready: bool = True,
        pose_ready: bool = True,
        failure_reason: str | None = None,
    ) -> None:
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg
        self.mesh_ready = mesh_ready
        self.pose_ready = pose_ready
        self.failure_reason = failure_reason
        self.calls = 0

    def estimate(self, frame_bgr: np.ndarray, face_box: BoundingBox):
        self.calls += 1
        return type(
            "Pose",
            (),
            {
                "mesh_ready": self.mesh_ready,
                "pose_ready": self.pose_ready,
                "yaw_deg": self.yaw_deg,
                "pitch_deg": self.pitch_deg,
                "failure_reason": self.failure_reason,
                "debug_message": "dummy-debug",
                "yaw_source": "mesh_2d" if self.failure_reason else "pnp",
                "mesh_points": ((10, 10), (12, 12)),
            },
        )()

    def close(self) -> None:
        return None


def make_config(**overrides: object) -> AppConfig:
    values = {
        "database_path": Path("data/drone_ai.sqlite3"),
        "embedder_model_path": Path("models/face_recognition_sface_2021dec_int8.onnx"),
        "detector_model_path": Path("models/blaze_face_short_range.tflite"),
        "tracking_head_pose_enabled": True,
        "tracking_head_yaw_deadband_deg": 10.0,
        "tracking_lateral_gain": 1.0,
        "tracking_min_lateral_speed": 0,
        "tracking_max_lateral_speed": 25,
        "tracking_min_vertical_speed": 8,
        "tracking_orbit_yaw_assist_px_per_deg": 0.0,
        "tracking_head_yaw_turn_gain": 0.28,
    }
    values.update(overrides)
    return AppConfig(**values)


class ApplicationTrackingTests(unittest.TestCase):
    def _make_application(self) -> DroneApplication:
        application = object.__new__(DroneApplication)
        application._config = make_config()
        application._tracker = FaceTracker(application._config)
        application._controller = _DummyController()
        application._head_pose = _DummyHeadPose()
        application._tracking_enabled = True
        application._tracking_target_visible = False
        application._tracking_target_distance_m = None
        application._show_head_mesh = False
        application._frame_lock = RLock()
        return application

    def test_apply_tracking_enriches_target_and_sends_lateral_velocity(self) -> None:
        application = self._make_application()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [
            RecognizedFace(
                bounding_box=BoundingBox(x=250, y=130, width=120, height=140),
                confidence=0.99,
                label="Maks",
                similarity=0.91,
                embedding_ready=True,
            )
        ]

        tracked_faces = application._apply_tracking(frame, faces)

        self.assertEqual(len(tracked_faces), 1)
        self.assertTrue(tracked_faces[0].is_tracking_target)
        self.assertTrue(tracked_faces[0].head_mesh_ready)
        self.assertTrue(tracked_faces[0].head_pose_ready)
        self.assertEqual(tracked_faces[0].head_yaw_deg, 24.0)
        self.assertEqual(tracked_faces[0].head_pose_debug, "dummy-debug")
        self.assertEqual(tracked_faces[0].head_mesh_points, ((10, 10), (12, 12)))
        self.assertLess(application._controller.calls[0][0], 0)
        self.assertLess(application._controller.calls[0][3], 0)
        self.assertEqual(application._head_pose.calls, 1)


    def test_apply_tracking_adjusts_height_when_face_is_too_low(self) -> None:
        application = self._make_application()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [
            RecognizedFace(
                bounding_box=BoundingBox(x=250, y=270, width=120, height=140),
                confidence=0.99,
                label="Maks",
                similarity=0.91,
                embedding_ready=True,
            )
        ]

        application._apply_tracking(frame, faces)

        self.assertNotEqual(application._controller.calls[0][2], 0)
        self.assertGreaterEqual(abs(application._controller.calls[0][2]), 8)

    def test_apply_tracking_skips_head_pose_without_target(self) -> None:
        application = self._make_application()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [
            RecognizedFace(
                bounding_box=BoundingBox(x=250, y=130, width=120, height=140),
                confidence=0.99,
                label="unknown",
                similarity=0.2,
                embedding_ready=True,
            )
        ]

        tracked_faces = application._apply_tracking(frame, faces)

        self.assertFalse(tracked_faces[0].is_tracking_target)
        self.assertEqual(application._head_pose.calls, 0)
        self.assertEqual(application._controller.calls[0][0], 0)

    def test_apply_tracking_shows_mesh_for_largest_face_when_overlay_is_enabled(self) -> None:
        application = self._make_application()
        application._show_head_mesh = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [
            RecognizedFace(
                bounding_box=BoundingBox(x=50, y=80, width=80, height=80),
                confidence=0.97,
                label="unknown",
                similarity=0.15,
                embedding_ready=True,
            ),
            RecognizedFace(
                bounding_box=BoundingBox(x=220, y=100, width=150, height=160),
                confidence=0.99,
                label="unknown",
                similarity=0.21,
                embedding_ready=True,
            ),
        ]

        tracked_faces = application._apply_tracking(frame, faces)

        self.assertEqual(application._head_pose.calls, 1)
        self.assertFalse(tracked_faces[0].head_pose_ready)
        self.assertTrue(tracked_faces[1].head_pose_ready)
        self.assertEqual(tracked_faces[1].head_mesh_points, ((10, 10), (12, 12)))

    def test_apply_tracking_keeps_mesh_when_pose_falls_back(self) -> None:
        application = self._make_application()
        application._show_head_mesh = True
        application._head_pose = _DummyHeadPose(
            yaw_deg=11.0,
            pitch_deg=None,
            mesh_ready=True,
            pose_ready=True,
            failure_reason="pnp_failed",
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [
            RecognizedFace(
                bounding_box=BoundingBox(x=220, y=100, width=150, height=160),
                confidence=0.99,
                label="unknown",
                similarity=0.21,
                embedding_ready=True,
            ),
        ]

        tracked_faces = application._apply_tracking(frame, faces)

        self.assertTrue(tracked_faces[0].head_mesh_ready)
        self.assertTrue(tracked_faces[0].head_pose_ready)
        self.assertEqual(tracked_faces[0].head_yaw_deg, 11.0)
        self.assertEqual(tracked_faces[0].head_pose_failure_reason, "pnp_failed")
        self.assertEqual(tracked_faces[0].head_pose_debug, "dummy-debug")


if __name__ == "__main__":
    unittest.main()
