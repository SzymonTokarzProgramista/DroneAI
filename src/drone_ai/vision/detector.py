"""Face detector with MediaPipe-first strategy and OpenCV profile fallback."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2

from drone_ai.constants.vision import (
    DETECTOR_DEFAULT_MIN_CONFIDENCE,
    DETECTOR_DEFAULT_NMS_THRESHOLD,
    DETECTOR_FALLBACK_MIN_CONFIDENCE,
    DETECTOR_RECOVERY_CONFIDENCE_DELTA,
    HAAR_FRONTAL_CONFIDENCE,
    HAAR_FRONTAL_MIN_NEIGHBORS,
    HAAR_FRONTAL_MIN_SIZE,
    HAAR_FRONTAL_SCALE_FACTOR,
    HAAR_PROFILE_CONFIDENCE,
    HAAR_PROFILE_MIN_NEIGHBORS,
    HAAR_PROFILE_MIN_SIZE,
    HAAR_PROFILE_SCALE_FACTOR,
    MEDIAPIPE_DETECTOR_CONFIDENCE_CAP,
    MEDIAPIPE_FULL_RANGE_MODEL_SELECTION,
    MEDIAPIPE_SHORT_RANGE_MODEL_SELECTION,
    NMS_MIN_CANDIDATES,
    RECOVERY_FRONTAL_MAX_ASPECT_RATIO,
    RECOVERY_FRONTAL_MIN_ASPECT_RATIO,
    RECOVERY_MIN_FACE_AREA_PX,
    RECOVERY_MIN_FACE_SIZE_PX,
    RECOVERY_PROFILE_MAX_ASPECT_RATIO,
    RECOVERY_PROFILE_MIN_ASPECT_RATIO,
)
from drone_ai.vision.schemas import BoundingBox, FaceDetection

try:
    import mediapipe as mp
except ImportError:
    mp = None


@dataclass(frozen=True)
class _DetectorBackend:
    name: str
    detector: Any


class MediaPipeFaceDetector:
    """Face detector backed by multiple detectors with cautious recovery mode."""

    def __init__(
        self,
        *,
        min_detection_confidence: float = DETECTOR_DEFAULT_MIN_CONFIDENCE,
        recovery_detection_confidence: Optional[float] = None,
        detector_model_path: Optional[Path] = None,
        nms_threshold: float = DETECTOR_DEFAULT_NMS_THRESHOLD,
    ) -> None:
        self._min_detection_confidence = min_detection_confidence
        self._recovery_detection_confidence = min(
            min_detection_confidence,
            recovery_detection_confidence
            if recovery_detection_confidence is not None
            else max(
                DETECTOR_FALLBACK_MIN_CONFIDENCE,
                min_detection_confidence - DETECTOR_RECOVERY_CONFIDENCE_DELTA,
            ),
        )
        self._nms_threshold = nms_threshold
        self._detectors = self._create_detectors(
            min_detection_confidence=min_detection_confidence,
            detector_model_path=detector_model_path,
        )

    def detect(self, frame_bgr: Any) -> list[FaceDetection]:
        frame_height, frame_width = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        primary_detections = self._collect_stage(
            {"solutions_short", "solutions_full", "tasks"},
            rgb_frame=rgb_frame,
            grayscale=grayscale,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        filtered_primary = self._filter_detections(primary_detections, allow_recovery=False)
        if filtered_primary:
            return filtered_primary

        frontal_detections = self._collect_stage(
            {"haar_frontal"},
            rgb_frame=rgb_frame,
            grayscale=grayscale,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        filtered_frontal = self._filter_detections(frontal_detections, allow_recovery=True)
        if filtered_frontal:
            return filtered_frontal

        profile_detections = self._collect_stage(
            {"haar_profile"},
            rgb_frame=rgb_frame,
            grayscale=grayscale,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return self._filter_detections(profile_detections, allow_recovery=True, profile_mode=True)

    def close(self) -> None:
        for backend in self._detectors:
            close = getattr(backend.detector, "close", None)
            if callable(close):
                close()

    def _collect_stage(
        self,
        backend_names: set[str],
        *,
        rgb_frame: Any,
        grayscale: Any,
        frame_width: int,
        frame_height: int,
    ) -> list[FaceDetection]:
        detections: list[FaceDetection] = []
        for backend in self._detectors:
            if backend.name not in backend_names:
                continue
            try:
                detections.extend(
                    self._detect_with_backend(
                        backend,
                        rgb_frame=rgb_frame,
                        grayscale=grayscale,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )
            except Exception:
                continue
        return detections

    def _create_detectors(
        self,
        *,
        min_detection_confidence: float,
        detector_model_path: Optional[Path],
    ) -> list[_DetectorBackend]:
        detectors: list[_DetectorBackend] = []

        if mp is not None:
            solutions_module = self._load_solutions_module()
            if solutions_module is not None:
                for model_selection, name in (
                    (MEDIAPIPE_SHORT_RANGE_MODEL_SELECTION, "solutions_short"),
                    (MEDIAPIPE_FULL_RANGE_MODEL_SELECTION, "solutions_full"),
                ):
                    try:
                        detector = solutions_module.FaceDetection(
                            model_selection=model_selection,
                            min_detection_confidence=min(
                                min_detection_confidence,
                                MEDIAPIPE_DETECTOR_CONFIDENCE_CAP,
                            ),
                        )
                        detectors.append(_DetectorBackend(name=name, detector=detector))
                    except Exception:
                        continue

            tasks_detector = self._load_tasks_detector(
                min_detection_confidence=min(
                    min_detection_confidence,
                    MEDIAPIPE_DETECTOR_CONFIDENCE_CAP,
                ),
                detector_model_path=detector_model_path,
            )
            if tasks_detector is not None:
                detectors.append(_DetectorBackend(name="tasks", detector=tasks_detector))

        frontal_detector = self._load_haar_detector("haarcascade_frontalface_default.xml")
        if frontal_detector is not None:
            detectors.append(_DetectorBackend(name="haar_frontal", detector=frontal_detector))

        profile_detector = self._load_haar_detector("haarcascade_profileface.xml")
        if profile_detector is not None:
            detectors.append(_DetectorBackend(name="haar_profile", detector=profile_detector))

        if detectors:
            return detectors

        raise RuntimeError(
            "No supported face detector backend is available. "
            "Tried MediaPipe solutions, MediaPipe tasks, and OpenCV Haar cascades."
        )

    def _detect_with_backend(
        self,
        backend: _DetectorBackend,
        *,
        rgb_frame: Any,
        grayscale: Any,
        frame_width: int,
        frame_height: int,
    ) -> list[FaceDetection]:
        if backend.name.startswith("solutions"):
            result = backend.detector.process(rgb_frame)
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
            return detections

        if backend.name == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = backend.detector.detect(mp_image)
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
            return detections

        if backend.name == "haar_frontal":
            faces = backend.detector.detectMultiScale(
                grayscale,
                scaleFactor=HAAR_FRONTAL_SCALE_FACTOR,
                minNeighbors=HAAR_FRONTAL_MIN_NEIGHBORS,
                minSize=HAAR_FRONTAL_MIN_SIZE,
            )
            return [
                FaceDetection(
                    bounding_box=BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height)),
                    confidence=HAAR_FRONTAL_CONFIDENCE,
                )
                for (x, y, width, height) in faces
            ]

        if backend.name == "haar_profile":
            detections = self._detect_profile_faces(backend.detector, grayscale, frame_width)
            return [
                FaceDetection(
                    bounding_box=box,
                    confidence=HAAR_PROFILE_CONFIDENCE,
                )
                for box in detections
            ]

        return []

    @staticmethod
    def _load_solutions_module() -> Optional[Any]:
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
        detector_model_path: Optional[Path],
    ) -> Optional[Any]:
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
    def _load_haar_detector(filename: str) -> Optional[Any]:
        cascade_path = Path(cv2.data.haarcascades) / filename
        if not cascade_path.exists():
            return None

        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return None
        return detector

    @staticmethod
    def _detect_profile_faces(detector: Any, grayscale: Any, frame_width: int) -> list[BoundingBox]:
        detections: list[BoundingBox] = []
        faces = detector.detectMultiScale(
            grayscale,
            scaleFactor=HAAR_PROFILE_SCALE_FACTOR,
            minNeighbors=HAAR_PROFILE_MIN_NEIGHBORS,
            minSize=HAAR_PROFILE_MIN_SIZE,
        )
        detections.extend(
            BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))
            for (x, y, width, height) in faces
        )

        flipped = cv2.flip(grayscale, 1)
        mirrored_faces = detector.detectMultiScale(
            flipped,
            scaleFactor=HAAR_PROFILE_SCALE_FACTOR,
            minNeighbors=HAAR_PROFILE_MIN_NEIGHBORS,
            minSize=HAAR_PROFILE_MIN_SIZE,
        )
        detections.extend(
            MediaPipeFaceDetector._mirror_bounding_box(
                BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height)),
                frame_width,
            )
            for (x, y, width, height) in mirrored_faces
        )
        return detections

    @staticmethod
    def _mirror_bounding_box(box: BoundingBox, frame_width: int) -> BoundingBox:
        mirrored_x = frame_width - (box.x + box.width)
        return BoundingBox(x=int(mirrored_x), y=box.y, width=box.width, height=box.height)

    def _filter_detections(
        self,
        detections: list[FaceDetection],
        *,
        allow_recovery: bool,
        profile_mode: bool = False,
    ) -> list[FaceDetection]:
        valid = [
            detection
            for detection in detections
            if detection.bounding_box.width > 0 and detection.bounding_box.height > 0
        ]
        if not valid:
            return []

        filtered = [
            detection
            for detection in valid
            if detection.confidence >= self._min_detection_confidence
        ]
        if not filtered and allow_recovery:
            filtered = [
                detection
                for detection in valid
                if detection.confidence >= self._recovery_detection_confidence
                and self._passes_recovery_geometry(detection.bounding_box, profile_mode=profile_mode)
            ]
        if len(filtered) < NMS_MIN_CANDIDATES:
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
    def _passes_recovery_geometry(box: BoundingBox, *, profile_mode: bool) -> bool:
        if (
            box.width < RECOVERY_MIN_FACE_SIZE_PX
            or box.height < RECOVERY_MIN_FACE_SIZE_PX
            or box.area < RECOVERY_MIN_FACE_AREA_PX
        ):
            return False
        aspect_ratio = box.width / float(box.height)
        if profile_mode:
            return RECOVERY_PROFILE_MIN_ASPECT_RATIO <= aspect_ratio <= RECOVERY_PROFILE_MAX_ASPECT_RATIO
        return RECOVERY_FRONTAL_MIN_ASPECT_RATIO <= aspect_ratio <= RECOVERY_FRONTAL_MAX_ASPECT_RATIO

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
