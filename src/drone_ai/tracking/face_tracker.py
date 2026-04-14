"""Face-based tracking controller for DJI Tello."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from drone_ai.config import AppConfig
from drone_ai.vision.schemas import BoundingBox, RecognizedFace


@dataclass(frozen=True)
class TrackingCommand:
    left_right_velocity: int
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
        self._vertical_target_y_ratio = config.tracking_vertical_target_y_ratio
        self._bbox_anchor_y_ratio = config.tracking_bbox_anchor_y_ratio
        self._distance_deadband_m = config.tracking_distance_deadband_m
        self._forward_gain = config.tracking_forward_gain
        self._yaw_gain = config.tracking_yaw_gain
        self._vertical_gain = config.tracking_vertical_gain
        self._lateral_gain = config.tracking_lateral_gain
        self._min_lateral_speed = config.tracking_min_lateral_speed
        self._max_forward_speed = config.tracking_max_forward_speed
        self._max_yaw_speed = config.tracking_max_yaw_speed
        self._max_vertical_speed = config.tracking_max_vertical_speed
        self._min_vertical_speed = config.tracking_min_vertical_speed
        self._max_lateral_speed = config.tracking_max_lateral_speed
        self._head_pose_enabled = config.tracking_head_pose_enabled
        self._head_yaw_deadband_deg = config.tracking_head_yaw_deadband_deg
        self._orbit_yaw_assist_px_per_deg = config.tracking_orbit_yaw_assist_px_per_deg
        self._reacquire_timeout_seconds = config.tracking_reacquire_timeout_seconds
        self._search_yaw_speed = config.tracking_search_yaw_speed
        self._match_max_distance_px = config.tracking_reacquire_match_max_distance_px
        self._reacquire_min_confidence = config.tracking_reacquire_min_confidence
        self._reacquire_min_score = config.tracking_reacquire_min_score
        self._preferred_frontal_yaw_deg = config.tracking_preferred_frontal_yaw_deg
        self._profile_recenter_yaw_gain = config.tracking_profile_recenter_yaw_gain
        self._head_yaw_turn_gain = config.tracking_head_yaw_turn_gain
        self._search_direction = 1
        self._last_seen_at = 0.0
        self._last_seen_box: BoundingBox | None = None
        self._last_seen_head_yaw_deg: float | None = None

    @property
    def target_name(self) -> str:
        return self._target_name

    def select_target(self, faces: list[RecognizedFace]) -> RecognizedFace | None:
        candidates = [face for face in faces if face.label == self._target_name]
        if candidates:
            target = max(
                candidates,
                key=lambda face: (
                    face.similarity if face.similarity is not None else -1.0,
                    face.bounding_box.area,
                ),
            )
            self._remember_target(target)
            return target

        fallback = self._select_reacquire_candidate(faces)
        if fallback is not None:
            self._remember_target(fallback)
        return fallback

    def estimate_distance_m(self, frame_width: int, face_width_px: int) -> float | None:
        if frame_width <= 0 or face_width_px <= 0:
            return None
        focal_length_px = frame_width / (2.0 * math.tan(math.radians(self._camera_hfov_deg) / 2.0))
        return (self._face_width_m * focal_length_px) / float(face_width_px)

    def build_command(self, frame_width: int, target_face: RecognizedFace | None) -> TrackingCommand:
        if target_face is None:
            search_command = self._build_search_command()
            if search_command is not None:
                return search_command
            return TrackingCommand(0, 0, 0, 0, None, False)

        self._remember_target(target_face)
        estimated_distance_m = self.estimate_distance_m(
            frame_width,
            target_face.bounding_box.width,
        )

        yaw_velocity = 0
        face_center_x = target_face.bounding_box.x + target_face.bounding_box.width / 2.0
        frame_center_x = frame_width / 2.0
        desired_face_center_x = frame_center_x
        profile_recenter_active = False
        head_yaw_error_deg = None
        motion_head_yaw_deg = None
        if self._head_pose_enabled and target_face.head_pose_ready and target_face.head_yaw_deg is not None:
            head_yaw_error_deg = target_face.head_yaw_deg
            motion_head_yaw_deg = -head_yaw_error_deg
            desired_face_center_x += motion_head_yaw_deg * self._orbit_yaw_assist_px_per_deg
            profile_recenter_active = abs(motion_head_yaw_deg) >= self._preferred_frontal_yaw_deg

        horizontal_error_px = face_center_x - desired_face_center_x
        self._update_search_direction(horizontal_error_px, motion_head_yaw_deg)
        if abs(horizontal_error_px) > self._yaw_deadband_px:
            yaw_velocity = self._clamp_speed(
                horizontal_error_px * self._yaw_gain,
                self._max_yaw_speed,
            )

        if motion_head_yaw_deg is not None and abs(motion_head_yaw_deg) > self._head_yaw_deadband_deg:
            yaw_turn_gain = self._head_yaw_turn_gain
            if profile_recenter_active:
                yaw_turn_gain += self._profile_recenter_yaw_gain
            yaw_velocity += self._clamp_speed(
                motion_head_yaw_deg * yaw_turn_gain,
                self._max_yaw_speed,
            )
            yaw_velocity = self._clamp_speed(yaw_velocity, self._max_yaw_speed)

        forward_backward_velocity = 0
        if estimated_distance_m is not None:
            distance_error_m = estimated_distance_m - self._target_distance_m
            if abs(distance_error_m) > self._distance_deadband_m:
                forward_backward_velocity = self._clamp_speed(
                    distance_error_m * self._forward_gain,
                    self._max_forward_speed,
                )

        left_right_velocity = 0
        if (
            self._head_pose_enabled
            and target_face.head_pose_ready
            and motion_head_yaw_deg is not None
            and abs(motion_head_yaw_deg) > self._head_yaw_deadband_deg
        ):
            left_right_velocity = self._clamp_speed_with_minimum(
                motion_head_yaw_deg * self._lateral_gain,
                self._min_lateral_speed,
                self._max_lateral_speed,
            )

        return TrackingCommand(
            left_right_velocity=left_right_velocity,
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

        tracking_anchor_y = self._resolve_tracking_anchor_y(target_face)
        desired_face_center_y = frame_height * self._vertical_target_y_ratio
        vertical_error_px = desired_face_center_y - tracking_anchor_y

        up_down_velocity = 0
        if abs(vertical_error_px) > self._vertical_deadband_px:
            up_down_velocity = self._clamp_speed_with_minimum(
                vertical_error_px * self._vertical_gain,
                self._min_vertical_speed,
                self._max_vertical_speed,
            )

        return TrackingCommand(
            left_right_velocity=command.left_right_velocity,
            forward_backward_velocity=command.forward_backward_velocity,
            up_down_velocity=up_down_velocity,
            yaw_velocity=command.yaw_velocity,
            estimated_distance_m=command.estimated_distance_m,
            target_visible=command.target_visible,
        )

    def _resolve_tracking_anchor_y(self, target_face: RecognizedFace) -> float:
        if target_face.tracking_anchor_y_px is not None:
            return target_face.tracking_anchor_y_px
        return target_face.bounding_box.y + (
            target_face.bounding_box.height * self._bbox_anchor_y_ratio
        )

    def _select_reacquire_candidate(self, faces: list[RecognizedFace]) -> RecognizedFace | None:
        if not self._can_reacquire() or self._last_seen_box is None:
            return None
        if not faces:
            return None

        candidates: list[tuple[float, RecognizedFace]] = []
        for face in faces:
            if not face.embedding_ready or face.confidence < self._reacquire_min_confidence:
                continue
            score = self._reacquire_score(face.bounding_box)
            if score is None or score < self._reacquire_min_score:
                continue
            candidates.append((score, face))

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (item[0], item[1].confidence, item[1].bounding_box.area),
            reverse=True,
        )
        return candidates[0][1]

    def _reacquire_score(self, candidate_box: BoundingBox) -> float | None:
        if self._last_seen_box is None:
            return None

        iou = self._iou(candidate_box, self._last_seen_box)
        candidate_center_x, candidate_center_y = self._box_center(candidate_box)
        last_center_x, last_center_y = self._box_center(self._last_seen_box)
        center_distance_px = math.hypot(candidate_center_x - last_center_x, candidate_center_y - last_center_y)
        if center_distance_px > self._match_max_distance_px:
            return None

        return (iou * 2.2) - (center_distance_px / max(self._match_max_distance_px, 1.0))

    def _remember_target(self, face: RecognizedFace) -> None:
        self._last_seen_at = time.monotonic()
        self._last_seen_box = face.bounding_box
        self._last_seen_head_yaw_deg = face.head_yaw_deg

    def _can_reacquire(self) -> bool:
        if self._last_seen_at <= 0.0:
            return False
        return (time.monotonic() - self._last_seen_at) <= self._reacquire_timeout_seconds

    def _build_search_command(self) -> TrackingCommand | None:
        if not self._can_reacquire() or self._search_yaw_speed <= 0:
            return None
        return TrackingCommand(
            left_right_velocity=0,
            forward_backward_velocity=0,
            up_down_velocity=0,
            yaw_velocity=self._search_direction * self._search_yaw_speed,
            estimated_distance_m=None,
            target_visible=False,
        )

    def _update_search_direction(
        self,
        horizontal_error_px: float,
        head_yaw_deg: float | None,
    ) -> None:
        if abs(horizontal_error_px) > self._yaw_deadband_px / 2.0:
            self._search_direction = 1 if horizontal_error_px > 0 else -1
            return
        if head_yaw_deg is not None and abs(head_yaw_deg) > self._head_yaw_deadband_deg:
            self._search_direction = 1 if head_yaw_deg > 0 else -1

    @staticmethod
    def _box_center(box: BoundingBox) -> tuple[float, float]:
        return box.x + (box.width / 2.0), box.y + (box.height / 2.0)

    @staticmethod
    def _iou(left: BoundingBox, right: BoundingBox) -> float:
        x1 = max(left.x, right.x)
        y1 = max(left.y, right.y)
        x2 = min(left.x + left.width, right.x + right.width)
        y2 = min(left.y + left.height, right.y + right.height)
        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)
        intersection = intersection_width * intersection_height
        if intersection == 0:
            return 0.0
        union = left.area + right.area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _clamp_speed(value: float, limit: int) -> int:
        if value > 0:
            return min(int(round(value)), limit)
        return max(int(round(value)), -limit)

    @staticmethod
    def _clamp_speed_with_minimum(value: float, minimum: int, limit: int) -> int:
        if value == 0:
            return 0

        clamped = FaceTracker._clamp_speed(value, limit)
        if clamped > 0:
            return max(clamped, minimum)
        if clamped < 0:
            return min(clamped, -minimum)
        return 0
