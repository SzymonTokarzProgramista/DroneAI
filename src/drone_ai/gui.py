"""Desktop GUI for preview, enrollment, and recognition status."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import cv2
from PIL import Image, ImageTk

from drone_ai.constants.runtime import (
    GUI_COLUMN_PREVIEW_WEIGHT,
    GUI_COLUMN_SIDEBAR_WEIGHT,
    GUI_MIN_HEIGHT_PX,
    GUI_MIN_WIDTH_PX,
    GUI_OUTER_PADDING_PX,
    GUI_PREVIEW_THUMBNAIL_SIZE,
    GUI_REFRESH_INTERVAL_MS,
    GUI_SECTION_FONT_SIZE,
    GUI_TITLE_FONT_SIZE,
    GUI_WINDOW_GEOMETRY,
    GUI_WRAP_LENGTH_PX,
    REGISTRATION_SERIES_SAMPLE_COUNT,
)
from drone_ai.storage.face_repository import IdentitySummary
from drone_ai.vision.schemas import ApiStatus, RecognizedFace


class DroneAIGUI:
    """Tkinter-based operator GUI."""

    def __init__(
        self,
        *,
        title: str,
        get_frame: Callable[[], object | None],
        get_faces: Callable[[], list[RecognizedFace]],
        get_status: Callable[[], ApiStatus],
        list_identities: Callable[[], list[IdentitySummary]],
        register_face: Callable[[str], IdentitySummary],
        register_face_series: Callable[[str], IdentitySummary],
        takeoff: Callable[[], object],
        land: Callable[[], object],
        enable_tracking: Callable[[], None],
        disable_tracking: Callable[[], None],
        head_mesh_enabled: Callable[[], bool],
        toggle_head_mesh: Callable[[], bool],
    ) -> None:
        self._get_frame = get_frame
        self._get_faces = get_faces
        self._get_status = get_status
        self._list_identities = list_identities
        self._register_face = register_face
        self._register_face_series = register_face_series
        self._takeoff = takeoff
        self._land = land
        self._enable_tracking = enable_tracking
        self._disable_tracking = disable_tracking
        self._head_mesh_enabled = head_mesh_enabled
        self._toggle_head_mesh = toggle_head_mesh

        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry(GUI_WINDOW_GEOMETRY)
        self._root.minsize(GUI_MIN_WIDTH_PX, GUI_MIN_HEIGHT_PX)

        self._preview_image: Optional[ImageTk.PhotoImage] = None
        self._status_var = tk.StringVar(value="Starting...")
        self._message_var = tk.StringVar(value="Ready.")
        self._tracking_button_var = tk.StringVar(value="Enable Tracking")
        self._name_var = tk.StringVar()
        self._visible_faces_var = tk.StringVar(value="No detections yet.")
        self._identities_var = tk.StringVar(value="No saved identities.")
        self._head_mesh_button_var = tk.StringVar(value="Show FaceMesh")

        self._build_layout()

    def run(self) -> int:
        self._schedule_refresh()
        self._root.mainloop()
        return 0

    def close(self) -> None:
        if self._root.winfo_exists():
            self._root.quit()
            self._root.destroy()

    def _build_layout(self) -> None:
        self._root.columnconfigure(0, weight=GUI_COLUMN_PREVIEW_WEIGHT)
        self._root.columnconfigure(1, weight=GUI_COLUMN_SIDEBAR_WEIGHT)
        self._root.rowconfigure(0, weight=1)

        preview_frame = ttk.Frame(self._root, padding=GUI_OUTER_PADDING_PX)
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self._preview_label = ttk.Label(preview_frame, text="Waiting for video stream...")
        self._preview_label.grid(row=0, column=0, sticky="nsew")

        sidebar = ttk.Frame(self._root, padding=GUI_OUTER_PADDING_PX)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="DroneAI", font=("TkDefaultFont", GUI_TITLE_FONT_SIZE, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            sidebar,
            textvariable=self._status_var,
            wraplength=GUI_WRAP_LENGTH_PX,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 12))

        flight_controls = ttk.Frame(sidebar)
        flight_controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        flight_controls.columnconfigure(0, weight=1)
        flight_controls.columnconfigure(1, weight=1)
        ttk.Button(flight_controls, text="Takeoff", command=self._handle_takeoff).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(flight_controls, text="Land", command=self._handle_land).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        ttk.Button(
            sidebar,
            textvariable=self._tracking_button_var,
            command=self._toggle_tracking,
        ).grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(
            sidebar,
            textvariable=self._head_mesh_button_var,
            command=self._toggle_head_mesh_overlay,
        ).grid(row=4, column=0, sticky="ew", pady=(0, 8))

        ttk.Separator(sidebar).grid(row=5, column=0, sticky="ew", pady=8)
        ttk.Label(
            sidebar,
            text="Capture Face",
            font=("TkDefaultFont", GUI_SECTION_FONT_SIZE, "bold"),
        ).grid(
            row=6, column=0, sticky="w"
        )
        ttk.Entry(sidebar, textvariable=self._name_var).grid(
            row=7, column=0, sticky="ew", pady=(8, 6)
        )
        ttk.Button(
            sidebar,
            text="Capture Largest Face",
            command=self._capture_face,
        ).grid(row=8, column=0, sticky="ew")
        ttk.Button(
            sidebar,
            text=f"Capture {REGISTRATION_SERIES_SAMPLE_COUNT} Samples",
            command=self._capture_face_series,
        ).grid(row=9, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(
            sidebar,
            text="Capture one photo to build a one-shot face profile, or capture a short series for extra robustness.",
            wraplength=GUI_WRAP_LENGTH_PX,
            justify="left",
        ).grid(row=10, column=0, sticky="ew", pady=(8, 12))
        ttk.Label(
            sidebar,
            textvariable=self._message_var,
            foreground="#1f4f99",
            wraplength=GUI_WRAP_LENGTH_PX,
            justify="left",
        ).grid(row=11, column=0, sticky="ew", pady=(0, 12))

        ttk.Separator(sidebar).grid(row=12, column=0, sticky="ew", pady=8)
        ttk.Label(
            sidebar,
            text="Visible Faces",
            font=("TkDefaultFont", GUI_SECTION_FONT_SIZE, "bold"),
        ).grid(
            row=13, column=0, sticky="w"
        )
        ttk.Label(
            sidebar,
            textvariable=self._visible_faces_var,
            wraplength=GUI_WRAP_LENGTH_PX,
            justify="left",
        ).grid(row=14, column=0, sticky="ew", pady=(8, 12))

        ttk.Separator(sidebar).grid(row=15, column=0, sticky="ew", pady=8)
        identities_header = ttk.Frame(sidebar)
        identities_header.grid(row=16, column=0, sticky="ew")
        identities_header.columnconfigure(0, weight=1)
        ttk.Label(
            identities_header,
            text="Known Identities",
            font=("TkDefaultFont", GUI_SECTION_FONT_SIZE, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            identities_header,
            text="Refresh",
            command=self._refresh_identity_list,
        ).grid(row=0, column=1, sticky="e")
        ttk.Label(
            sidebar,
            textvariable=self._identities_var,
            wraplength=GUI_WRAP_LENGTH_PX,
            justify="left",
        ).grid(row=17, column=0, sticky="ew", pady=(8, 0))

        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh_identity_list()

    def _schedule_refresh(self) -> None:
        self._update_preview()
        self._update_status()
        self._root.after(GUI_REFRESH_INTERVAL_MS, self._schedule_refresh)

    def _update_preview(self) -> None:
        frame = self._get_frame()
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_image.thumbnail(GUI_PREVIEW_THUMBNAIL_SIZE)
        photo = ImageTk.PhotoImage(frame_image)
        self._preview_label.configure(image=photo, text="")
        self._preview_label.image = photo
        self._preview_image = photo

    def _update_status(self) -> None:
        status = self._get_status()
        self._tracking_button_var.set(
            "Disable Tracking" if status.tracking_enabled else "Enable Tracking"
        )
        self._head_mesh_button_var.set(
            "Hide FaceMesh" if self._head_mesh_enabled() else "Show FaceMesh"
        )
        self._status_var.set(
            "\n".join(
                [
                    f"Connected: {status.connected}",
                    f"Battery: {status.battery if status.battery is not None else '--'}%",
                    f"Stream: {status.stream_enabled}",
                    f"Flying: {status.flying}",
                    f"Visible faces: {status.visible_faces}",
                    f"Known identities: {status.known_identities}",
                    f"Tracking: {status.tracking_enabled}",
                    f"Target: {status.tracking_target_name or '--'}",
                    f"Target visible: {status.tracking_target_visible}",
                    f"Searching: {status.tracking_search_active}",
                    f"Search direction: {status.tracking_search_direction or '--'}",
                    f"Target distance: "
                    f"{status.tracking_target_distance_m:.2f}m"
                    if status.tracking_target_distance_m is not None
                    else "Target distance: --",
                    f"API: {status.api_url or 'not started'}",
                ]
            )
        )
        self._visible_faces_var.set(self._format_visible_faces(self._get_faces()))

    def _handle_takeoff(self) -> None:
        try:
            self._takeoff()
            self._message_var.set("Takeoff command sent.")
        except Exception as exc:
            self._message_var.set(f"Takeoff failed: {exc}")

    def _handle_land(self) -> None:
        try:
            self._land()
            self._message_var.set("Land command sent.")
        except Exception as exc:
            self._message_var.set(f"Land failed: {exc}")

    def _toggle_tracking(self) -> None:
        status = self._get_status()
        try:
            if status.tracking_enabled:
                self._disable_tracking()
                self._message_var.set("Tracking disabled.")
            else:
                self._enable_tracking()
                self._message_var.set(
                    f"Tracking enabled for target '{status.tracking_target_name}'."
                )
        except Exception as exc:
            self._message_var.set(f"Tracking toggle failed: {exc}")

    def _toggle_head_mesh_overlay(self) -> None:
        try:
            enabled = self._toggle_head_mesh()
            self._message_var.set(
                "FaceMesh overlay enabled." if enabled else "FaceMesh overlay disabled."
            )
        except Exception as exc:
            self._message_var.set(f"FaceMesh toggle failed: {exc}")

    def _capture_face(self) -> None:
        identity_name = self._name_var.get().strip()
        if not identity_name:
            self._message_var.set("Enter a name before capturing a face.")
            return

        try:
            summary = self._register_face(identity_name)
        except Exception as exc:
            self._message_var.set(f"Capture failed: {exc}")
            return

        self._message_var.set(
            f"Saved one-shot face profile for '{summary.name}'. "
            f"Total embeddings: {summary.embedding_count}."
        )
        self._refresh_identity_list()

    def _capture_face_series(self) -> None:
        identity_name = self._name_var.get().strip()
        if not identity_name:
            self._message_var.set("Enter a name before capturing a face series.")
            return

        self._message_var.set(
            f"Capturing {REGISTRATION_SERIES_SAMPLE_COUNT} samples. "
            "Keep one face centered and still for a moment."
        )
        self._root.update_idletasks()

        try:
            summary = self._register_face_series(identity_name)
        except Exception as exc:
            self._message_var.set(f"Series capture failed: {exc}")
            return

        self._message_var.set(
            f"Saved {REGISTRATION_SERIES_SAMPLE_COUNT} samples for '{summary.name}'. "
            f"Total embeddings: {summary.embedding_count}."
        )
        self._refresh_identity_list()

    def _refresh_identity_list(self) -> None:
        identities = self._list_identities()
        if not identities:
            self._identities_var.set("No saved identities.")
            return

        lines = [
            f"{identity.name} ({identity.embedding_count} embeddings)"
            for identity in identities
        ]
        self._identities_var.set("\n".join(lines))

    @staticmethod
    def _format_visible_faces(faces: list[RecognizedFace]) -> str:
        if not faces:
            return "No detections."

        lines = []
        for index, face in enumerate(faces, start=1):
            similarity = f"{face.similarity:.3f}" if face.similarity is not None else "--"
            yaw = f"{face.head_yaw_deg:+.0f}deg" if face.head_pose_ready and face.head_yaw_deg is not None else "--"
            mesh = "ok" if face.head_mesh_ready else "--"
            pose = face.head_pose_failure_reason or "ok"
            debug = face.head_pose_debug or "--"
            lines.append(
                f"{index}. {face.label} | det={face.confidence:.2f} | cos={similarity} | yaw={yaw} | mesh={mesh} | pose={pose} | dbg={debug}"
            )
        return "\n".join(lines)
