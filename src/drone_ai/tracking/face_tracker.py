"""Face-based tracking controller for DJI Tello."""

from __future__ import annotations

import math
from dataclasses import dataclass

from drone_ai.config import AppConfig
from drone_ai.vision.schemas import RecognizedFace


@dataclass(frozen=True)
class TrackingCommand:
    forward_backward_velocity: int
    up_down_velocity: int
    yaw_velocity: int
    estimated_distance_m: float | None
    target_visible: bool


class FaceTracker:
    """Computes RC commands to keep the target face centered and at a target distance."""

    def __init__(self, config: AppConfig) -> None:
        self._target_name = config.tracking_target_name
        self._target_distance_m = config.tracking_target_distance_m
        self._face_width_m = config.tracking_face_width_m
        self._camera_hfov_deg = config.tracking_camera_hfov_deg
        self._yaw_deadband_px = config.tracking_yaw_deadband_px
        self._vertical_deadband_px = config.tracking_vertical_deadband_px
        self._distance_deadband_m = config.tracking_distance_deadband_m
        self._forward_gain = config.tracking_forward_gain
        self._yaw_gain = config.tracking_yaw_gain
        self._vertical_gain = config.tracking_vertical_gain
        self._max_forward_speed = config.tracking_max_forward_speed
        self._max_yaw_speed = config.tracking_max_yaw_speed
        self._max_vertical_speed = config.tracking_max_vertical_speed

    @property
    def target_name(self) -> str:
        return self._target_name

    def select_target(self, faces: list[RecognizedFace]) -> RecognizedFace | None:
        candidates = [face for face in faces if face.label == self._target_name]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda face: (
                face.similarity if face.similarity is not None else -1.0,
                face.bounding_box.area,
            ),
        )

    def estimate_distance_m(self, frame_width: int, face_width_px: int) -> float | None:
        if frame_width <= 0 or face_width_px <= 0:
            return None
        focal_length_px = frame_width / (2.0 * math.tan(math.radians(self._camera_hfov_deg) / 2.0))
        return (self._face_width_m * focal_length_px) / float(face_width_px)

    def build_command(self, frame_width: int, target_face: RecognizedFace | None) -> TrackingCommand:
        if target_face is None:
            return TrackingCommand(0, 0, 0, None, False)

        estimated_distance_m = self.estimate_distance_m(
            frame_width,
            target_face.bounding_box.width,
        )

        yaw_velocity = 0
        face_center_x = target_face.bounding_box.x + target_face.bounding_box.width / 2.0
        frame_center_x = frame_width / 2.0
        horizontal_error_px = face_center_x - frame_center_x
        if abs(horizontal_error_px) > self._yaw_deadband_px:
            yaw_velocity = self._clamp_speed(
                horizontal_error_px * self._yaw_gain,
                self._max_yaw_speed,
            )

        forward_backward_velocity = 0
        if estimated_distance_m is not None:
            distance_error_m = estimated_distance_m - self._target_distance_m
            if abs(distance_error_m) > self._distance_deadband_m:
                forward_backward_velocity = self._clamp_speed(
                    distance_error_m * self._forward_gain,
                    self._max_forward_speed,
                )

        return TrackingCommand(
            forward_backward_velocity=forward_backward_velocity,
            up_down_velocity=0,
            yaw_velocity=yaw_velocity,
            estimated_distance_m=estimated_distance_m,
            target_visible=True,
        )

    def build_command_full(
        self,
        frame_width: int,
        frame_height: int,
        target_face: RecognizedFace | None,
    ) -> TrackingCommand:
        command = self.build_command(frame_width, target_face)
        if target_face is None:
            return command

        face_center_y = target_face.bounding_box.y + target_face.bounding_box.height / 2.0
        desired_face_center_y = frame_height * 0.42
        vertical_error_px = desired_face_center_y - face_center_y

        up_down_velocity = 0
        if abs(vertical_error_px) > self._vertical_deadband_px:
            up_down_velocity = self._clamp_speed(
                vertical_error_px * self._vertical_gain,
                self._max_vertical_speed,
            )

        return TrackingCommand(
            forward_backward_velocity=command.forward_backward_velocity,
            up_down_velocity=up_down_velocity,
            yaw_velocity=command.yaw_velocity,
            estimated_distance_m=command.estimated_distance_m,
            target_visible=command.target_visible,
        )

    @staticmethod
    def _clamp_speed(value: float, limit: int) -> int:
        if value > 0:
            return min(int(round(value)), limit)
        return max(int(round(value)), -limit)
