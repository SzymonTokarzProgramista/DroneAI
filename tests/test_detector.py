from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_ai.vision.detector import MediaPipeFaceDetector
from drone_ai.vision.schemas import BoundingBox, FaceDetection


class FaceDetectorTests(unittest.TestCase):
    def _make_detector(self) -> MediaPipeFaceDetector:
        detector = object.__new__(MediaPipeFaceDetector)
        detector._min_detection_confidence = 0.9
        detector._recovery_detection_confidence = 0.62
        detector._nms_threshold = 0.35
        detector._detectors = []
        return detector

    def test_recovery_threshold_returns_weaker_detection_when_geometry_is_good(self) -> None:
        detector = self._make_detector()
        detections = [
            FaceDetection(
                bounding_box=BoundingBox(x=10, y=20, width=80, height=84),
                confidence=0.7,
            )
        ]

        filtered = detector._filter_detections(detections, allow_recovery=True)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].confidence, 0.7)

    def test_recovery_threshold_rejects_bad_geometry_false_positive(self) -> None:
        detector = self._make_detector()
        detections = [
            FaceDetection(
                bounding_box=BoundingBox(x=10, y=20, width=130, height=50),
                confidence=0.7,
            )
        ]

        filtered = detector._filter_detections(detections, allow_recovery=True)

        self.assertEqual(filtered, [])

    def test_profile_mode_rejects_small_detection(self) -> None:
        detector = self._make_detector()
        detections = [
            FaceDetection(
                bounding_box=BoundingBox(x=10, y=20, width=40, height=60),
                confidence=0.7,
            )
        ]

        filtered = detector._filter_detections(detections, allow_recovery=True, profile_mode=True)

        self.assertEqual(filtered, [])

    def test_mirror_bounding_box_maps_profile_detection_back_to_original_frame(self) -> None:
        mirrored = MediaPipeFaceDetector._mirror_bounding_box(
            BoundingBox(x=30, y=20, width=50, height=60),
            320,
        )

        self.assertEqual(mirrored.x, 240)
        self.assertEqual(mirrored.y, 20)
        self.assertEqual(mirrored.width, 50)
        self.assertEqual(mirrored.height, 60)


if __name__ == "__main__":
    unittest.main()
