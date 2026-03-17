from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_ai.config import AppConfig
from drone_ai.tracking.face_tracker import FaceTracker
from drone_ai.vision.schemas import BoundingBox, RecognizedFace


def make_config(**overrides: object) -> AppConfig:
    values = {
        "database_path": Path("data/drone_ai.sqlite3"),
        "embedder_model_path": Path("models/face_recognition_sface_2021dec_int8.onnx"),
        "detector_model_path": Path("models/blaze_face_short_range.tflite"),
        "tracking_head_pose_enabled": True,
        "tracking_head_yaw_deadband_deg": 10.0,
        "tracking_lateral_gain": 1.0,
        "tracking_min_lateral_speed": 12,
        "tracking_max_lateral_speed": 20,
        "tracking_yaw_deadband_px": 40.0,
        "tracking_yaw_gain": 0.1,
        "tracking_orbit_yaw_assist_px_per_deg": 0.0,
    }
    values.update(overrides)
    return AppConfig(**values)


def make_face(**overrides: object) -> RecognizedFace:
    values = {
        "bounding_box": BoundingBox(x=270, y=120, width=100, height=120),
        "confidence": 0.98,
        "label": "Maks",
        "similarity": 0.88,
        "embedding_ready": True,
        "head_mesh_ready": True,
        "head_pose_ready": True,
        "head_yaw_deg": 0.0,
        "head_pitch_deg": 0.0,
    }
    values.update(overrides)
    return RecognizedFace(**values)


class FaceTrackerTests(unittest.TestCase):
    def test_no_target_produces_zero_velocities(self) -> None:
        tracker = FaceTracker(make_config())

        command = tracker.build_command_full(640, 480, None)

        self.assertEqual(command.left_right_velocity, 0)
        self.assertEqual(command.forward_backward_velocity, 0)
        self.assertEqual(command.up_down_velocity, 0)
        self.assertEqual(command.yaw_velocity, 0)
        self.assertFalse(command.target_visible)

    def test_centered_frontal_face_has_no_lateral_motion(self) -> None:
        tracker = FaceTracker(make_config())
        face = make_face(bounding_box=BoundingBox(x=270, y=142, width=100, height=120), head_yaw_deg=0.0)

        command = tracker.build_command_full(640, 480, face)

        self.assertEqual(command.left_right_velocity, 0)

    def test_positive_head_yaw_moves_drone_right(self) -> None:
        tracker = FaceTracker(make_config(tracking_min_lateral_speed=0))
        face = make_face(head_yaw_deg=18.0)

        command = tracker.build_command_full(640, 480, face)

        self.assertEqual(command.left_right_velocity, 18)

    def test_speed_is_clamped_for_large_head_yaw(self) -> None:
        tracker = FaceTracker(
            make_config(
                tracking_lateral_gain=2.0,
                tracking_min_lateral_speed=0,
                tracking_max_lateral_speed=12,
            )
        )
        face = make_face(head_yaw_deg=30.0)

        command = tracker.build_command_full(640, 480, face)

        self.assertEqual(command.left_right_velocity, 12)

    def test_deadband_prevents_small_orbit_adjustments(self) -> None:
        tracker = FaceTracker(make_config(tracking_head_yaw_deadband_deg=15.0))
        face = make_face(head_yaw_deg=10.0)

        command = tracker.build_command_full(640, 480, face)

        self.assertEqual(command.left_right_velocity, 0)

    def test_orbit_assist_biases_yaw_for_centered_face(self) -> None:
        tracker = FaceTracker(
            make_config(
                tracking_min_lateral_speed=0,
                tracking_yaw_deadband_px=10.0,
                tracking_orbit_yaw_assist_px_per_deg=3.0,
            )
        )
        face = make_face(
            bounding_box=BoundingBox(x=270, y=142, width=100, height=120),
            head_yaw_deg=20.0,
        )

        command = tracker.build_command_full(640, 480, face)

        self.assertEqual(command.yaw_velocity, -6)


if __name__ == "__main__":
    unittest.main()
