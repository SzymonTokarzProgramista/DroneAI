"""Rendering helpers for annotated frames."""

from __future__ import annotations

import cv2
import numpy as np

from drone_ai.constants.vision import (
    BOX_LINE_THICKNESS_PX,
    COLOR_EMBEDDING_ERROR_BGR,
    COLOR_KNOWN_FACE_BGR,
    COLOR_MESH_KEYPOINT_BGR,
    COLOR_MESH_POINT_BGR,
    COLOR_TRACKING_TARGET_BGR,
    COLOR_UNKNOWN_FACE_BGR,
    FILLED_CIRCLE_THICKNESS,
    LABEL_FONT_SCALE,
    LABEL_LINE_THICKNESS_PX,
    LABEL_MIN_Y_PX,
    LABEL_Y_OFFSET_PX,
    MESH_KEYPOINT_RADIUS_PX,
    MESH_KEYPOINT_STRIDE,
    MESH_POINT_RADIUS_PX,
)
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
            color = COLOR_KNOWN_FACE_BGR if face.label != "unknown" else COLOR_UNKNOWN_FACE_BGR
            if not face.embedding_ready:
                color = COLOR_EMBEDDING_ERROR_BGR
            if face.is_tracking_target:
                color = COLOR_TRACKING_TARGET_BGR

            cv2.rectangle(
                annotated,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                color,
                BOX_LINE_THICKNESS_PX,
            )

            if show_head_mesh and face.head_mesh_ready and face.head_mesh_points:
                for point_x, point_y in face.head_mesh_points:
                    if 0 <= point_x < annotated.shape[1] and 0 <= point_y < annotated.shape[0]:
                        cv2.circle(
                            annotated,
                            (point_x, point_y),
                            MESH_POINT_RADIUS_PX,
                            COLOR_MESH_POINT_BGR,
                            FILLED_CIRCLE_THICKNESS,
                        )
                for point_x, point_y in face.head_mesh_points[::MESH_KEYPOINT_STRIDE]:
                    if 0 <= point_x < annotated.shape[1] and 0 <= point_y < annotated.shape[0]:
                        cv2.circle(
                            annotated,
                            (point_x, point_y),
                            MESH_KEYPOINT_RADIUS_PX,
                            COLOR_MESH_KEYPOINT_BGR,
                            FILLED_CIRCLE_THICKNESS,
                        )

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
            text_origin = (box.x, max(box.y - LABEL_Y_OFFSET_PX, LABEL_MIN_Y_PX))
            cv2.putText(
                annotated,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                color,
                LABEL_LINE_THICKNESS_PX,
                cv2.LINE_AA,
            )

        return annotated
