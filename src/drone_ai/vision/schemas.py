"""Shared types for face detection and recognition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(self.width, 0) * max(self.height, 0)


@dataclass(frozen=True)
class FaceDetection:
    bounding_box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class RecognizedFace:
    bounding_box: BoundingBox
    confidence: float
    label: str
    similarity: float | None
    embedding_ready: bool
    estimated_distance_m: float | None = None
    is_tracking_target: bool = False
    head_mesh_ready: bool = False
    head_pose_ready: bool = False
    head_yaw_deg: float | None = None
    head_pitch_deg: float | None = None
    head_pose_failure_reason: str | None = None
    head_pose_debug: str | None = None
    head_mesh_points: tuple[tuple[int, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FrameAnalysis:
    raw_frame: np.ndarray
    annotated_frame: np.ndarray
    faces: list[RecognizedFace] = field(default_factory=list)


@dataclass(frozen=True)
class ApiStatus:
    connected: bool
    battery: int | None
    stream_enabled: bool
    flying: bool
    known_identities: int
    visible_faces: int
    tracking_enabled: bool = False
    tracking_target_name: str | None = None
    tracking_target_visible: bool = False
    tracking_target_distance_m: float | None = None
    api_url: str | None = None
