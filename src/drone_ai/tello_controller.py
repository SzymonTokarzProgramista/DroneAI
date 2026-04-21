"""Thin integration layer over DJITelloPy."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
import time
from typing import Any, Optional

import cv2


@dataclass
class TelloStatus:
    connected: bool
    battery: Optional[int] = None
    stream_enabled: bool = False
    flying: bool = False


class TelloController:
    """Encapsulates the initial Tello handshake used by the project."""

    def __init__(self, *, takeoff_extra_rise_cm: int = 30) -> None:
        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise RuntimeError(
                "DJITelloPy is not available. Run the project through `uv sync` or `./start.sh` first."
            ) from exc

        try:
            self._tello = Tello()
        except PermissionError as exc:
            raise RuntimeError(
                "Tello socket initialization failed. This environment blocks UDP sockets, so run the command on your local machine connected to the drone Wi-Fi."
            ) from exc

        self._connected = False
        self._stream_enabled = False
        self._flying = False
        self._frame_reader: Optional[Any] = None
        self._lock = RLock()
        self._takeoff_extra_rise_cm = max(int(takeoff_extra_rise_cm), 0)

    def connect(self, *, enable_stream: bool = False) -> TelloStatus:
        with self._lock:
            if not self._connected:
                self._tello.connect()
                self._connected = True

            battery = self._tello.get_battery()

            if enable_stream:
                self._enable_stream_locked()

            return TelloStatus(
                connected=True,
                battery=battery,
                stream_enabled=self._stream_enabled,
                flying=self._flying,
            )

    def start_video_stream(self) -> TelloStatus:
        with self._lock:
            if not self._connected:
                self.connect(enable_stream=True)
            elif not self._stream_enabled:
                self._enable_stream_locked()

            return self.status()

    def stop_video_stream(self) -> TelloStatus:
        with self._lock:
            if self._connected and self._stream_enabled:
                self._tello.streamoff()
                self._stream_enabled = False
                self._frame_reader = None

            return self.status()

    def get_latest_frame(self) -> Any:
        with self._lock:
            if not self._connected:
                raise RuntimeError("Tello is not connected.")
            if not self._stream_enabled:
                raise RuntimeError("Tello video stream is not enabled.")

            if self._frame_reader is None:
                self._frame_reader = self._tello.get_frame_read()

            frame = self._frame_reader.frame
            if frame is None:
                raise RuntimeError("Tello video frame is not available yet.")

            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def takeoff(self) -> TelloStatus:
        with self._lock:
            if not self._connected:
                raise RuntimeError("Tello is not connected.")
            if not self._flying:
                self._tello.takeoff()
                self._flying = True
                if self._takeoff_extra_rise_cm > 0:
                    self._tello.move_up(self._takeoff_extra_rise_cm)
                if self._stream_enabled:
                    time.sleep(0.5)
                    self._frame_reader = self._tello.get_frame_read()
            return self.status()

    def land(self) -> TelloStatus:
        with self._lock:
            if not self._connected:
                raise RuntimeError("Tello is not connected.")
            if self._flying:
                self._tello.land()
                self._flying = False
            self._tello.send_rc_control(0, 0, 0, 0)
            return self.status()

    def send_rc_control(
        self,
        left_right_velocity: int,
        forward_backward_velocity: int,
        up_down_velocity: int,
        yaw_velocity: int,
    ) -> None:
        with self._lock:
            if not self._connected or not self._flying:
                return
            self._tello.send_rc_control(
                int(left_right_velocity),
                int(forward_backward_velocity),
                int(up_down_velocity),
                int(yaw_velocity),
            )

    def stop_motion(self) -> None:
        self.send_rc_control(0, 0, 0, 0)

    def status(self) -> TelloStatus:
        with self._lock:
            battery = None
            if self._connected:
                try:
                    battery = self._tello.get_battery()
                except Exception:
                    battery = None

            return TelloStatus(
                connected=self._connected,
                battery=battery,
                stream_enabled=self._stream_enabled,
                flying=self._flying,
            )

    def disconnect(self) -> None:
        with self._lock:
            if not self._connected:
                return

            if self._stream_enabled:
                try:
                    self._tello.streamoff()
                except Exception:
                    pass
                self._stream_enabled = False
                self._frame_reader = None

            if self._flying:
                try:
                    self._tello.send_rc_control(0, 0, 0, 0)
                except Exception:
                    pass
                self._flying = False

            try:
                self._tello.end()
            except Exception:
                pass

            self._connected = False

    def _enable_stream_locked(self) -> None:
        self._tello.streamon()
        self._stream_enabled = True
        time.sleep(0.2)
        self._frame_reader = self._tello.get_frame_read()
