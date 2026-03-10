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
    preview_window_name: str = "DroneAI Tello Front Camera"
    recognition_threshold: float = 0.45
    min_detection_confidence: float = 0.8
    detection_nms_threshold: float = 0.35

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
        return cls(
            database_path=database_path,
            embedder_model_path=embedder_model_path,
            detector_model_path=detector_model_path,
            preview_window_name=os.environ.get(
                "DRONE_AI_PREVIEW_WINDOW", "DroneAI Tello Front Camera"
            ),
            recognition_threshold=float(
                os.environ.get("DRONE_AI_RECOGNITION_THRESHOLD", "0.45")
            ),
            min_detection_confidence=float(
                os.environ.get("DRONE_AI_MIN_DETECTION_CONFIDENCE", "0.8")
            ),
            detection_nms_threshold=float(
                os.environ.get("DRONE_AI_DETECTION_NMS_THRESHOLD", "0.35")
            ),
        )
