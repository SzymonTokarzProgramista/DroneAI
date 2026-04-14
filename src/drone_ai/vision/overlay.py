"""Rendering helpers for annotated frames."""

from __future__ import annotations

import cv2
import numpy as np

from drone_ai.vision.schemas import RecognizedFace


class FaceOverlayRenderer:
    """Draws face boxes and labels on frames."""

    def render(
        self,
        frame_bgr: np.ndarray,
        faces: list[RecognizedFace],
        *,
        show_head_mesh: bool = False,
    ) -> np.ndarray:
        annotated = frame_bgr.copy()
        for face in faces:
            box = face.bounding_box
            color = (0, 200, 0) if face.label != "unknown" else (0, 165, 255)
            if not face.embedding_ready:
                color = (0, 0, 255)
            if face.is_tracking_target:
                color = (255, 140, 0)

            cv2.rectangle(
                annotated,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                color,
                2,
            )

            if show_head_mesh and face.head_mesh_ready and face.head_mesh_points:
                for point_x, point_y in face.head_mesh_points:
                    if 0 <= point_x < annotated.shape[1] and 0 <= point_y < annotated.shape[0]:
                        cv2.circle(annotated, (point_x, point_y), 1, (255, 255, 0), -1)
                for point_x, point_y in face.head_mesh_points[::40]:
                    if 0 <= point_x < annotated.shape[1] and 0 <= point_y < annotated.shape[0]:
                        cv2.circle(annotated, (point_x, point_y), 2, (0, 255, 255), -1)

            similarity = f"{face.similarity:.2f}" if face.similarity is not None else "--"
            distance = (
                f"{face.estimated_distance_m:.2f}m"
                if face.estimated_distance_m is not None
                else "--"
            )
            head_yaw = (
                f"{face.head_yaw_deg:+.0f}deg"
                if face.head_pose_ready and face.head_yaw_deg is not None
                else "--"
            )
            mesh_status = "ok" if face.head_mesh_ready else "--"
            pose_status = face.head_pose_failure_reason or "ok"
            debug_status = face.head_pose_debug or "--"
            tracking_anchor_source = face.tracking_anchor_source or "--"
            tracking_suffix = " | TRACK" if face.is_tracking_target else ""
            label = (
                f"{face.label} | det={face.confidence:.2f} | sim={similarity} | dist={distance} | yaw={head_yaw} | mesh={mesh_status} | pose={pose_status} | vsrc={tracking_anchor_source} | dbg={debug_status}"
                f"{tracking_suffix}"
            )
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
