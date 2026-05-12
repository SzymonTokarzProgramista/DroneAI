"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from drone_ai.constants import config_defaults as defaults


@dataclass(frozen=True)
class AppConfig:
    database_path: Path
    embedder_model_path: Path
    detector_model_path: Path
    face_landmarker_model_path: Path = defaults.DEFAULT_FACE_LANDMARKER_MODEL_PATH
    preview_window_name: str = defaults.DEFAULT_PREVIEW_WINDOW_NAME
    recognition_threshold: float = defaults.DEFAULT_RECOGNITION_THRESHOLD
    recognition_margin_threshold: float = defaults.DEFAULT_RECOGNITION_MARGIN_THRESHOLD
    min_detection_confidence: float = defaults.DEFAULT_MIN_DETECTION_CONFIDENCE
    recovery_detection_confidence: float = defaults.DEFAULT_RECOVERY_DETECTION_CONFIDENCE
    detection_nms_threshold: float = defaults.DEFAULT_DETECTION_NMS_THRESHOLD
    tracking_target_name: str = defaults.DEFAULT_TRACKING_TARGET_NAME
    tracking_target_distance_m: float = defaults.DEFAULT_TRACKING_TARGET_DISTANCE_M
    tracking_face_width_m: float = defaults.DEFAULT_TRACKING_FACE_WIDTH_M
    tracking_camera_hfov_deg: float = defaults.DEFAULT_TRACKING_CAMERA_HFOV_DEG
    tracking_yaw_deadband_px: float = defaults.DEFAULT_TRACKING_YAW_DEADBAND_PX
    tracking_vertical_deadband_px: float = defaults.DEFAULT_TRACKING_VERTICAL_DEADBAND_PX
    tracking_vertical_target_y_ratio: float = defaults.DEFAULT_TRACKING_VERTICAL_TARGET_Y_RATIO
    tracking_bbox_anchor_y_ratio: float = defaults.DEFAULT_TRACKING_BBOX_ANCHOR_Y_RATIO
    tracking_distance_deadband_m: float = defaults.DEFAULT_TRACKING_DISTANCE_DEADBAND_M
    tracking_forward_gain: float = defaults.DEFAULT_TRACKING_FORWARD_GAIN
    tracking_yaw_gain: float = defaults.DEFAULT_TRACKING_YAW_GAIN
    tracking_vertical_gain: float = defaults.DEFAULT_TRACKING_VERTICAL_GAIN
    tracking_lateral_gain: float = defaults.DEFAULT_TRACKING_LATERAL_GAIN
    tracking_min_lateral_speed: int = defaults.DEFAULT_TRACKING_MIN_LATERAL_SPEED
    tracking_max_forward_speed: int = defaults.DEFAULT_TRACKING_MAX_FORWARD_SPEED
    tracking_max_yaw_speed: int = defaults.DEFAULT_TRACKING_MAX_YAW_SPEED
    tracking_max_vertical_speed: int = defaults.DEFAULT_TRACKING_MAX_VERTICAL_SPEED
    tracking_min_vertical_speed: int = defaults.DEFAULT_TRACKING_MIN_VERTICAL_SPEED
    tracking_max_lateral_speed: int = defaults.DEFAULT_TRACKING_MAX_LATERAL_SPEED
    tracking_head_pose_enabled: bool = defaults.DEFAULT_TRACKING_HEAD_POSE_ENABLED
    tracking_head_yaw_deadband_deg: float = defaults.DEFAULT_TRACKING_HEAD_YAW_DEADBAND_DEG
    tracking_head_pose_min_confidence: float = defaults.DEFAULT_TRACKING_HEAD_POSE_MIN_CONFIDENCE
    tracking_orbit_yaw_assist_px_per_deg: float = defaults.DEFAULT_TRACKING_ORBIT_YAW_ASSIST_PX_PER_DEG
    tracking_reacquire_timeout_seconds: float = defaults.DEFAULT_TRACKING_REACQUIRE_TIMEOUT_SECONDS
    tracking_loss_search_timeout_seconds: float = defaults.DEFAULT_TRACKING_LOSS_SEARCH_TIMEOUT_SECONDS
    tracking_search_yaw_speed: int = defaults.DEFAULT_TRACKING_SEARCH_YAW_SPEED
    tracking_reacquire_match_max_distance_px: float = defaults.DEFAULT_TRACKING_REACQUIRE_MATCH_MAX_DISTANCE_PX
    tracking_reacquire_min_confidence: float = defaults.DEFAULT_TRACKING_REACQUIRE_MIN_CONFIDENCE
    tracking_reacquire_min_score: float = defaults.DEFAULT_TRACKING_REACQUIRE_MIN_SCORE
    tracking_preferred_frontal_yaw_deg: float = defaults.DEFAULT_TRACKING_PREFERRED_FRONTAL_YAW_DEG
    tracking_profile_recenter_yaw_gain: float = defaults.DEFAULT_TRACKING_PROFILE_RECENTER_YAW_GAIN
    tracking_head_yaw_turn_gain: float = defaults.DEFAULT_TRACKING_HEAD_YAW_TURN_GAIN
    takeoff_extra_rise_cm: int = defaults.DEFAULT_TAKEOFF_EXTRA_RISE_CM

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
                "DRONE_AI_PREVIEW_WINDOW", defaults.DEFAULT_PREVIEW_WINDOW_NAME
            ),
            recognition_threshold=float(
                os.environ.get("DRONE_AI_RECOGNITION_THRESHOLD", defaults.ENV_RECOGNITION_THRESHOLD)
            ),
            recognition_margin_threshold=float(
                os.environ.get("DRONE_AI_RECOGNITION_MARGIN_THRESHOLD", defaults.ENV_RECOGNITION_MARGIN_THRESHOLD)
            ),
            min_detection_confidence=float(
                os.environ.get("DRONE_AI_MIN_DETECTION_CONFIDENCE", defaults.ENV_MIN_DETECTION_CONFIDENCE)
            ),
            recovery_detection_confidence=float(
                os.environ.get("DRONE_AI_RECOVERY_DETECTION_CONFIDENCE", defaults.ENV_RECOVERY_DETECTION_CONFIDENCE)
            ),
            detection_nms_threshold=float(
                os.environ.get("DRONE_AI_DETECTION_NMS_THRESHOLD", defaults.ENV_DETECTION_NMS_THRESHOLD)
            ),
            tracking_target_name=os.environ.get(
                "DRONE_AI_TRACKING_TARGET_NAME", defaults.DEFAULT_TRACKING_TARGET_NAME
            ),
            tracking_target_distance_m=float(
                os.environ.get("DRONE_AI_TRACKING_TARGET_DISTANCE_M", defaults.ENV_TRACKING_TARGET_DISTANCE_M)
            ),
            tracking_face_width_m=float(
                os.environ.get("DRONE_AI_TRACKING_FACE_WIDTH_M", defaults.ENV_TRACKING_FACE_WIDTH_M)
            ),
            tracking_camera_hfov_deg=float(
                os.environ.get("DRONE_AI_TRACKING_CAMERA_HFOV_DEG", defaults.ENV_TRACKING_CAMERA_HFOV_DEG)
            ),
            tracking_yaw_deadband_px=float(
                os.environ.get("DRONE_AI_TRACKING_YAW_DEADBAND_PX", defaults.ENV_TRACKING_YAW_DEADBAND_PX)
            ),
            tracking_vertical_deadband_px=float(
                os.environ.get("DRONE_AI_TRACKING_VERTICAL_DEADBAND_PX", defaults.ENV_TRACKING_VERTICAL_DEADBAND_PX)
            ),
            tracking_vertical_target_y_ratio=float(
                os.environ.get("DRONE_AI_TRACKING_VERTICAL_TARGET_Y_RATIO", defaults.ENV_TRACKING_VERTICAL_TARGET_Y_RATIO)
            ),
            tracking_bbox_anchor_y_ratio=float(
                os.environ.get("DRONE_AI_TRACKING_BBOX_ANCHOR_Y_RATIO", defaults.ENV_TRACKING_BBOX_ANCHOR_Y_RATIO)
            ),
            tracking_distance_deadband_m=float(
                os.environ.get("DRONE_AI_TRACKING_DISTANCE_DEADBAND_M", defaults.ENV_TRACKING_DISTANCE_DEADBAND_M)
            ),
            tracking_forward_gain=float(
                os.environ.get("DRONE_AI_TRACKING_FORWARD_GAIN", defaults.ENV_TRACKING_FORWARD_GAIN)
            ),
            tracking_yaw_gain=float(
                os.environ.get("DRONE_AI_TRACKING_YAW_GAIN", defaults.ENV_TRACKING_YAW_GAIN)
            ),
            tracking_vertical_gain=float(
                os.environ.get("DRONE_AI_TRACKING_VERTICAL_GAIN", defaults.ENV_TRACKING_VERTICAL_GAIN)
            ),
            tracking_lateral_gain=float(
                os.environ.get("DRONE_AI_TRACKING_LATERAL_GAIN", defaults.ENV_TRACKING_LATERAL_GAIN)
            ),
            tracking_min_lateral_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MIN_LATERAL_SPEED", defaults.ENV_TRACKING_MIN_LATERAL_SPEED)
            ),
            tracking_max_forward_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MAX_FORWARD_SPEED", defaults.ENV_TRACKING_MAX_FORWARD_SPEED)
            ),
            tracking_max_yaw_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MAX_YAW_SPEED", defaults.ENV_TRACKING_MAX_YAW_SPEED)
            ),
            tracking_max_vertical_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MAX_VERTICAL_SPEED", defaults.ENV_TRACKING_MAX_VERTICAL_SPEED)
            ),
            tracking_min_vertical_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MIN_VERTICAL_SPEED", defaults.ENV_TRACKING_MIN_VERTICAL_SPEED)
            ),
            tracking_max_lateral_speed=int(
                os.environ.get("DRONE_AI_TRACKING_MAX_LATERAL_SPEED", defaults.ENV_TRACKING_MAX_LATERAL_SPEED)
            ),
            tracking_head_pose_enabled=os.environ.get(
                "DRONE_AI_TRACKING_HEAD_POSE_ENABLED", defaults.ENV_TRACKING_HEAD_POSE_ENABLED
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            tracking_head_yaw_deadband_deg=float(
                os.environ.get("DRONE_AI_TRACKING_HEAD_YAW_DEADBAND_DEG", defaults.ENV_TRACKING_HEAD_YAW_DEADBAND_DEG)
            ),
            tracking_head_pose_min_confidence=float(
                os.environ.get("DRONE_AI_TRACKING_HEAD_POSE_MIN_CONFIDENCE", defaults.ENV_TRACKING_HEAD_POSE_MIN_CONFIDENCE)
            ),
            tracking_orbit_yaw_assist_px_per_deg=float(
                os.environ.get("DRONE_AI_TRACKING_ORBIT_YAW_ASSIST_PX_PER_DEG", defaults.ENV_TRACKING_ORBIT_YAW_ASSIST_PX_PER_DEG)
            ),
            tracking_reacquire_timeout_seconds=float(
                os.environ.get("DRONE_AI_TRACKING_REACQUIRE_TIMEOUT_SECONDS", defaults.ENV_TRACKING_REACQUIRE_TIMEOUT_SECONDS)
            ),
            tracking_loss_search_timeout_seconds=float(
                os.environ.get("DRONE_AI_TRACKING_LOSS_SEARCH_TIMEOUT_SECONDS", defaults.ENV_TRACKING_LOSS_SEARCH_TIMEOUT_SECONDS)
            ),
            tracking_search_yaw_speed=int(
                os.environ.get("DRONE_AI_TRACKING_SEARCH_YAW_SPEED", defaults.ENV_TRACKING_SEARCH_YAW_SPEED)
            ),
            tracking_reacquire_match_max_distance_px=float(
                os.environ.get("DRONE_AI_TRACKING_REACQUIRE_MATCH_MAX_DISTANCE_PX", defaults.ENV_TRACKING_REACQUIRE_MATCH_MAX_DISTANCE_PX)
            ),
            tracking_reacquire_min_confidence=float(
                os.environ.get("DRONE_AI_TRACKING_REACQUIRE_MIN_CONFIDENCE", defaults.ENV_TRACKING_REACQUIRE_MIN_CONFIDENCE)
            ),
            tracking_reacquire_min_score=float(
                os.environ.get("DRONE_AI_TRACKING_REACQUIRE_MIN_SCORE", defaults.ENV_TRACKING_REACQUIRE_MIN_SCORE)
            ),
            tracking_preferred_frontal_yaw_deg=float(
                os.environ.get("DRONE_AI_TRACKING_PREFERRED_FRONTAL_YAW_DEG", defaults.ENV_TRACKING_PREFERRED_FRONTAL_YAW_DEG)
            ),
            tracking_profile_recenter_yaw_gain=float(
                os.environ.get("DRONE_AI_TRACKING_PROFILE_RECENTER_YAW_GAIN", defaults.ENV_TRACKING_PROFILE_RECENTER_YAW_GAIN)
            ),
            tracking_head_yaw_turn_gain=float(
                os.environ.get("DRONE_AI_TRACKING_HEAD_YAW_TURN_GAIN", defaults.ENV_TRACKING_HEAD_YAW_TURN_GAIN)
            ),
            takeoff_extra_rise_cm=int(
                os.environ.get("DRONE_AI_TAKEOFF_EXTRA_RISE_CM", defaults.ENV_TAKEOFF_EXTRA_RISE_CM)
            ),
        )
