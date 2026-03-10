"""Face processing pipeline."""

from __future__ import annotations

import numpy as np

from drone_ai.vision.detector import MediaPipeFaceDetector
from drone_ai.vision.overlay import FaceOverlayRenderer
from drone_ai.vision.recognizer import FaceRecognitionService
from drone_ai.vision.schemas import FaceDetection, FrameAnalysis


class FacePipeline:
    """Runs face detection, recognition, and overlay rendering."""

    def __init__(
        self,
        detector: MediaPipeFaceDetector,
        recognizer: FaceRecognitionService,
        renderer: FaceOverlayRenderer,
    ) -> None:
        self._detector = detector
        self._recognizer = recognizer
        self._renderer = renderer

    def process_frame(self, frame_bgr: np.ndarray) -> FrameAnalysis:
        raw_frame = frame_bgr.copy()
        detections = self._detector.detect(raw_frame)
        faces = self._recognizer.recognize_faces(raw_frame, detections)
        annotated_frame = self._renderer.render(raw_frame, faces)
        return FrameAnalysis(
            raw_frame=raw_frame,
            annotated_frame=annotated_frame,
            faces=faces,
        )

    @staticmethod
    def choose_face(detections: list[FaceDetection]) -> FaceDetection:
        if not detections:
            raise RuntimeError("No face detections available for registration.")
        return max(detections, key=lambda detection: detection.bounding_box.area)

    def close(self) -> None:
        self._detector.close()
