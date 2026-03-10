"""Face detector with MediaPipe-first strategy and OpenCV fallback."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import cv2

from drone_ai.vision.schemas import BoundingBox, FaceDetection

try:
    import mediapipe as mp
except ImportError:
    mp = None


class MediaPipeFaceDetector:
    """Face detector backed by MediaPipe when available, with OpenCV fallback."""

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.8,
        detector_model_path: Path | None = None,
        nms_threshold: float = 0.35,
    ) -> None:
        self._backend = "unknown"
        self._min_detection_confidence = min_detection_confidence
        self._nms_threshold = nms_threshold
        self._detector = self._create_detector(
            min_detection_confidence=min_detection_confidence,
            detector_model_path=detector_model_path,
        )

    def detect(self, frame_bgr: Any) -> list[FaceDetection]:
        frame_height, frame_width = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":
            result = self._detector.process(rgb_frame)
            if not result.detections:
                return []

            detections: list[FaceDetection] = []
            for detection in result.detections:
                rel_box = detection.location_data.relative_bounding_box
                x = max(int(rel_box.xmin * frame_width), 0)
                y = max(int(rel_box.ymin * frame_height), 0)
                width = min(int(rel_box.width * frame_width), frame_width - x)
                height = min(int(rel_box.height * frame_height), frame_height - y)
                detections.append(
                    FaceDetection(
                        bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
                        confidence=float(detection.score[0]) if detection.score else 0.0,
                    )
                )
            return self._filter_detections(detections)

        if self._backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._detector.detect(mp_image)
            if not result.detections:
                return []

            detections: list[FaceDetection] = []
            for detection in result.detections:
                bbox = detection.bounding_box
                x = max(int(bbox.origin_x), 0)
                y = max(int(bbox.origin_y), 0)
                width = min(int(bbox.width), frame_width - x)
                height = min(int(bbox.height), frame_height - y)
                confidence = 0.0
                if detection.categories:
                    confidence = float(detection.categories[0].score)
                detections.append(
                    FaceDetection(
                        bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
                        confidence=confidence,
                    )
                )
            return self._filter_detections(detections)

        if self._backend == "haar":
            grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._detector.detectMultiScale(
                grayscale,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            detections = [
                FaceDetection(
                    bounding_box=BoundingBox(
                        x=int(x),
                        y=int(y),
                        width=int(width),
                        height=int(height),
                    ),
                    confidence=1.0,
                )
                for (x, y, width, height) in faces
            ]
            return self._filter_detections(detections)

        raise RuntimeError("Face detector backend was not initialized correctly.")

    def close(self) -> None:
        close = getattr(self._detector, "close", None)
        if callable(close):
            close()

    def _create_detector(
        self,
        *,
        min_detection_confidence: float,
        detector_model_path: Path | None,
    ) -> Any:
        if mp is not None:
            solutions_module = self._load_solutions_module()
            if solutions_module is not None:
                self._backend = "solutions"
                return solutions_module.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=min_detection_confidence,
                )

            tasks_detector = self._load_tasks_detector(
                min_detection_confidence=min_detection_confidence,
                detector_model_path=detector_model_path,
            )
            if tasks_detector is not None:
                self._backend = "tasks"
                return tasks_detector

        haar_detector = self._load_haar_detector()
        if haar_detector is not None:
            self._backend = "haar"
            return haar_detector

        raise RuntimeError(
            "No supported face detector backend is available. "
            "Tried MediaPipe solutions, MediaPipe tasks, and OpenCV Haar cascade."
        )

    @staticmethod
    def _load_solutions_module() -> Any | None:
        if mp is None:
            return None

        solutions = getattr(mp, "solutions", None)
        face_detection = getattr(solutions, "face_detection", None) if solutions is not None else None
        if face_detection is not None:
            return face_detection

        for module_name in (
            "mediapipe.python.solutions.face_detection",
            "mediapipe.solutions.face_detection",
        ):
            try:
                return importlib.import_module(module_name)
            except Exception:
                continue
        return None

    @staticmethod
    def _load_tasks_detector(
        *,
        min_detection_confidence: float,
        detector_model_path: Path | None,
    ) -> Any | None:
        if mp is None:
            return None

        vision_module = None
        for module_name in ("mediapipe.tasks.python.vision", "mediapipe.tasks.vision"):
            try:
                vision_module = importlib.import_module(module_name)
                break
            except Exception:
                continue
        if vision_module is None:
            return None

        tasks_module = None
        for module_name in ("mediapipe.tasks.python", "mediapipe.tasks"):
            try:
                tasks_module = importlib.import_module(module_name)
                break
            except Exception:
                continue
        if tasks_module is None:
            return None

        if detector_model_path is None:
            return None

        model_path = Path(detector_model_path)
        if not model_path.exists():
            return None

        base_options_cls = getattr(tasks_module, "BaseOptions", None)
        running_mode = getattr(vision_module, "RunningMode", None)
        face_detector_cls = getattr(vision_module, "FaceDetector", None)
        options_cls = getattr(vision_module, "FaceDetectorOptions", None)
        if (
            base_options_cls is None
            or running_mode is None
            or face_detector_cls is None
            or options_cls is None
        ):
            return None

        options = options_cls(
            base_options=base_options_cls(model_asset_path=str(model_path)),
            running_mode=running_mode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        return face_detector_cls.create_from_options(options)

    @staticmethod
    def _load_haar_detector() -> Any | None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None

        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return None
        return detector

    def _filter_detections(self, detections: list[FaceDetection]) -> list[FaceDetection]:
        filtered = [
            detection
            for detection in detections
            if detection.confidence >= self._min_detection_confidence
            and detection.bounding_box.width > 0
            and detection.bounding_box.height > 0
        ]
        if len(filtered) < 2:
            return filtered

        filtered.sort(
            key=lambda detection: (detection.confidence, detection.bounding_box.area),
            reverse=True,
        )

        selected: list[FaceDetection] = []
        for candidate in filtered:
            if any(
                self._iou(candidate.bounding_box, existing.bounding_box) >= self._nms_threshold
                for existing in selected
            ):
                continue
            selected.append(candidate)

        return selected

    @staticmethod
    def _iou(left: BoundingBox, right: BoundingBox) -> float:
        left_x2 = left.x + left.width
        left_y2 = left.y + left.height
        right_x2 = right.x + right.width
        right_y2 = right.y + right.height

        inter_x1 = max(left.x, right.x)
        inter_y1 = max(left.y, right.y)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)

        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        if inter_area == 0:
            return 0.0

        union_area = left.area + right.area - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area
