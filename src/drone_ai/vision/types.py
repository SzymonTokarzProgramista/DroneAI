"""Shared types for face detection and recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    known_identities: int
    visible_faces: int
    api_url: str | None = None
