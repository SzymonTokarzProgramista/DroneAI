"""Thin integration layer over DJITelloPy."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

import cv2


@dataclass
class TelloStatus:
    connected: bool
    battery: int | None = None
    stream_enabled: bool = False


class TelloController:
    """Encapsulates the initial Tello handshake used by the project."""

    def __init__(self) -> None:
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
        self._lock = RLock()

    def connect(self, *, enable_stream: bool = False) -> TelloStatus:
        with self._lock:
            if not self._connected:
                self._tello.connect()
                self._connected = True

            battery = self._tello.get_battery()

            if enable_stream:
                self._tello.streamon()
                self._stream_enabled = True

            return TelloStatus(
                connected=True,
                battery=battery,
                stream_enabled=self._stream_enabled,
            )

    def start_video_stream(self) -> TelloStatus:
        with self._lock:
            if not self._connected:
                self.connect(enable_stream=True)
            elif not self._stream_enabled:
                self._tello.streamon()
                self._stream_enabled = True

            return self.status()

    def stop_video_stream(self) -> TelloStatus:
        with self._lock:
            if self._connected and self._stream_enabled:
                self._tello.streamoff()
                self._stream_enabled = False

            return self.status()

    def get_latest_frame(self) -> Any:
        with self._lock:
            if not self._connected:
                raise RuntimeError("Tello is not connected.")
            if not self._stream_enabled:
                raise RuntimeError("Tello video stream is not enabled.")

            frame = self._tello.get_frame_read().frame
            if frame is None:
                raise RuntimeError("Tello video frame is not available yet.")

            # DJITelloPy frames can arrive in RGB order; normalize to BGR for OpenCV processing/display.
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

            try:
                self._tello.end()
            except Exception:
                pass

            self._connected = False
