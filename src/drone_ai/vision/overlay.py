"""Rendering helpers for annotated frames."""

from __future__ import annotations

import cv2
import numpy as np

from drone_ai.vision.schemas import RecognizedFace


class FaceOverlayRenderer:
    """Draws face boxes and labels on frames."""

    def render(self, frame_bgr: np.ndarray, faces: list[RecognizedFace]) -> np.ndarray:
        annotated = frame_bgr.copy()
        for face in faces:
            box = face.bounding_box
            color = (0, 200, 0) if face.label != "unknown" else (0, 165, 255)
            if not face.embedding_ready:
                color = (0, 0, 255)

            cv2.rectangle(
                annotated,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                color,
                2,
            )

            similarity = f"{face.similarity:.2f}" if face.similarity is not None else "--"
            label = f"{face.label} | det={face.confidence:.2f} | sim={similarity}"
            text_origin = (box.x, max(box.y - 10, 20))
            cv2.putText(
                annotated,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated
