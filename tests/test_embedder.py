from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_ai.vision.embedder import SFaceEmbedder
from drone_ai.vision.schemas import BoundingBox


class SFaceEmbedderTests(unittest.TestCase):
    def test_crop_face_shifts_window_for_one_shot_variants(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(120, dtype=np.uint8)[None, :]
        box = BoundingBox(x=30, y=20, width=40, height=50)

        centered = SFaceEmbedder._crop_face(
            frame,
            box,
            margin_ratio=0.1,
            offset_x_ratio=0.0,
        )
        shifted = SFaceEmbedder._crop_face(
            frame,
            box,
            margin_ratio=0.1,
            offset_x_ratio=0.2,
        )

        self.assertEqual(centered.shape[:2], shifted.shape[:2])
        self.assertEqual(int(centered[0, 0, 0]), 26)
        self.assertEqual(int(shifted[0, 0, 0]), 34)

    def test_extract_face_chip_resizes_variant_crop_to_model_input(self) -> None:
        embedder = object.__new__(SFaceEmbedder)
        embedder._input_size = (112, 112)
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        box = BoundingBox(x=30, y=20, width=40, height=50)

        chip = embedder._extract_face_chip(
            frame,
            box,
            margin_ratio=0.2,
            offset_x_ratio=0.1,
            offset_y_ratio=-0.1,
        )

        self.assertEqual(chip.shape, (112, 112, 3))

    def test_invalid_box_returns_empty_crop(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)

        crop = SFaceEmbedder._crop_face(
            frame,
            BoundingBox(x=10, y=10, width=0, height=50),
        )

        self.assertEqual(crop.size, 0)


if __name__ == "__main__":
    unittest.main()
