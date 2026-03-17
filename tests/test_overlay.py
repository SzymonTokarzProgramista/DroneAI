from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_ai.vision.overlay import FaceOverlayRenderer
from drone_ai.vision.schemas import BoundingBox, RecognizedFace


class OverlayRendererTests(unittest.TestCase):
    def test_mesh_is_drawn_even_without_pose_ready(self) -> None:
        renderer = FaceOverlayRenderer()
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        face = RecognizedFace(
            bounding_box=BoundingBox(x=20, y=20, width=60, height=60),
            confidence=0.99,
            label="unknown",
            similarity=0.1,
            embedding_ready=True,
            head_mesh_ready=True,
            head_pose_ready=False,
            head_pose_failure_reason="pnp_failed",
            head_mesh_points=((40, 40), (45, 45), (50, 50)),
        )

        annotated = renderer.render(frame, [face], show_head_mesh=True)

        self.assertGreater(int(annotated[:, :, 1].sum()), 0)


if __name__ == "__main__":
    unittest.main()
