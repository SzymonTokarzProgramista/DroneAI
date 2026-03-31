"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    database_path: Path
    embedder_model_path: Path
    detector_model_path: Path
    face_landmarker_model_path: Path = Path("models/face_landmarker.task")
    preview_window_name: str = "DroneAI Tello Front Camera"
    recognition_threshold: float = 0.46
    recognition_margin_threshold: float = 0.03
    min_detection_confidence: float = 0.9
    recovery_detection_confidence: float = 0.62
    detection_nms_threshold: float = 0.25
    tracking_target_name: str = "Maks"
    tracking_target_distance_m: float = 0.3
    tracking_face_width_m: float = 0.16
    tracking_camera_hfov_deg: float = 82.6
    tracking_yaw_deadband_px: float = 60.0
    tracking_vertical_deadband_px: float = 45.0
    tracking_distance_deadband_m: float = 0.08
    tracking_forward_gain: float = 90.0
    tracking_yaw_gain: float = 0.12
    tracking_vertical_gain: float = 0.10
    tracking_lateral_gain: float = 0.65
    tracking_min_lateral_speed: int = 12
    tracking_max_forward_speed: int = 30
    tracking_max_yaw_speed: int = 35
    tracking_max_vertical_speed: int = 25
    tracking_min_vertical_speed: int = 8
    tracking_max_lateral_speed: int = 20
    tracking_head_pose_enabled: bool = True
    tracking_head_yaw_deadband_deg: float = 12.0
    tracking_head_pose_min_confidence: float = 0.5
    tracking_orbit_yaw_assist_px_per_deg: float = 3.5
    tracking_reacquire_timeout_seconds: float = 1.8
    tracking_search_yaw_speed: int = 18
    tracking_reacquire_match_max_distance_px: float = 220.0
    tracking_reacquire_min_confidence: float = 0.9
    tracking_reacquire_min_score: float = 0.12
    tracking_preferred_frontal_yaw_deg: float = 12.0
    tracking_profile_recenter_yaw_gain: float = 0.60
    tracking_head_yaw_turn_gain: float = 0.333
    takeoff_extra_rise_cm: int = 30

    @classmethod
    def from_env(cls, root_dir: Path) -> "AppConfig":
        database_path = Path(
            os.environ.get("DRONE_AI_DB_PATH", root_dir / "data" / "drone_ai.sqlite3")
        )
        embedder_model_path = Path(
            os.environ.get(
                "DRONE_AI_EMBEDDER_MODEL",
                root_dir / "models" / "face_recognition_sface_2021dec_int8.onnx",
            )
        )
        detector_model_path = Path(
            os.environ.get(
                "DRONE_AI_DETECTOR_MODEL",
                root_dir / "models" / "blaze_face_short_range.tflite",
            )
        )
        face_landmarker_model_path = Path(
            os.environ.get(
                "DRONE_AI_FACE_LANDMARKER_MODEL",
                root_dir / "models" / "face_landmarker.task",
            )
        )
        return cls(
            database_path=database_path,
            embedder_model_path=embedder_model_path,
            detector_model_path=detector_model_path,
            face_landmarker_model_path=face_landmarker_model_path,
            preview_window_name=os.environ.get(
                "DRONE_AI_PREVIEW_WINDOW", "DroneAI Tello Front Camera"
            ),
            recognition_threshold=float(
                os.environ.get("DRONE_AI_RECOGNITION_THRESHOLD", "0.6")
            ),
            recognition_margin_threshold=float(
                os.environ.get("DRONE_AI_RECOGNITION_MARGIN_THRESHOLD", "0.03")
            ),
            min_detection_confidence=float(
                os.environ.get("DRONE_AI_MIN_DETECTION_CONFIDENCE", "0.9")
            ),
            recovery_detection_confidence=float(
                os.environ.get("DRONE_AI_RECOVERY_DETECTION_CONFIDENCE", "0.62")
            ),
            detection_nms_threshold=float(
                os.environ.get("DRONE_AI_DETECTION_NMS_THRESHOLD", "0.4")
            ),
            tracking_target_name=os.environ.get("DRONE_AI_TRACKING_TARGET_NAME", "Oskar"),
            tracking_target_distance_m=float(
                os.environ.get("DRONE_AI_TRACKING_TARGET_DISTANCE_M", "0.6")
            ),
            tracking_face_width_m=float(
                os.environ.get("DRONE_AI_TRACKING_FACE_WIDTH_M", "0.16")
            ),
            tracking_camera_hfov_deg=float(
                os.environ.get("DRONE_AI_TRACKING_CAMERA_HFOV_DEG", "82.6")
            ),
            tracking_yaw_deadband_px=float(
                os.environ.get("DRONE_AI_TRACKING_YAW_DEADBAND_PX", "60")
            ),
            tracking_vertical_deadband_px=float(
                os.environ.get("DRONE_AI_TRACKING_VERTICAL_DEADBAND_PX", "45")
            ),
            tracking_distance_deadband_m=float(
                os.environ.get("DRONE_AI_TRACKING_DISTANCE_DEADBAND_M", "0.08")
            ),
            tracking_forward_gain=float(
                os.environ.get("DRONE_AI_TRACKING_FORWARD_GAIN", "90")
            ),
            tracking_yaw_gain=float(
                os.environ.get("DRONE_AI_TRACKING_YAW_GAIN", "0.12")
            ),
            tracking_vertical_gain=float(
                os.environ.get("DRONE_AI_TRACKING_VERTICAL_GAIN", "0.10")
            ),
            tracking_lateral_gain=float(
                os.environ.get("DRONE_AI_TRACKING_LATERAL_GAIN", "0.65")
            ),
            tracking_min_lateral_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MIN_LATERAL_SPEED", "12")
            ),
            tracking_max_forward_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MAX_FORWARD_SPEED", "30")
            ),
            tracking_max_yaw_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MAX_YAW_SPEED", "35")
            ),
            tracking_max_vertical_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MAX_VERTICAL_SPEED", "25")
            ),
            tracking_min_vertical_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MIN_VERTICAL_SPEED", "8")
            ),
            tracking_max_lateral_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MAX_LATERAL_SPEED", "20")
            ),
            tracking_head_pose_enabled=os.environ.get(
                "DRONE_AI_TRACKING_HEAD_POSE_ENABLED", "true"
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            tracking_head_yaw_deadband_deg=float(
                os.environ.get("DRONE_AI_TRACKING_HEAD_YAW_DEADBAND_DEG", "12")
            ),
            tracking_head_pose_min_confidence=float(
                os.environ.get("DRONE_AI_TRACKING_HEAD_POSE_MIN_CONFIDENCE", "0.5")
            ),
            tracking_orbit_yaw_assist_px_per_deg=float(
                os.environ.get("DRONE_AI_TRACKING_ORBIT_YAW_ASSIST_PX_PER_DEG", "3.5")
            ),
            tracking_reacquire_timeout_seconds=float(
                os.environ.get("DRONE_AI_TRACKING_REACQUIRE_TIMEOUT_SECONDS", "1.8")
            ),
            tracking_search_yaw_speed=int(
                os.environ.get("DRONE_AI_TRACKING_SEARCH_YAW_SPEED", "18")
            ),
            tracking_reacquire_match_max_distance_px=float(
                os.environ.get("DRONE_AI_TRACKING_REACQUIRE_MATCH_MAX_DISTANCE_PX", "220")
            ),
            tracking_reacquire_min_confidence=float(
                os.environ.get("DRONE_AI_TRACKING_REACQUIRE_MIN_CONFIDENCE", "0.9")
            ),
            tracking_reacquire_min_score=float(
                os.environ.get("DRONE_AI_TRACKING_REACQUIRE_MIN_SCORE", "0.12")
            ),
            tracking_preferred_frontal_yaw_deg=float(
                os.environ.get("DRONE_AI_TRACKING_PREFERRED_FRONTAL_YAW_DEG", "12")
            ),
            tracking_profile_recenter_yaw_gain=float(
                os.environ.get("DRONE_AI_TRACKING_PROFILE_RECENTER_YAW_GAIN", "0.60")
            ),
            tracking_head_yaw_turn_gain=float(
                os.environ.get("DRONE_AI_TRACKING_HEAD_YAW_TURN_GAIN", "0.38")
            ),
            takeoff_extra_rise_cm=int(
                os.environ.get("DRONE_AI_TAKEOFF_EXTRA_RISE_CM", "30")
            ),
        )
