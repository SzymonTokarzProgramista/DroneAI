"""Application runtime and HTTP API."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from drone_ai.config import AppConfig
from drone_ai.storage.face_repository import IdentitySummary, SQLiteFaceRepository
from drone_ai.tello_controller import TelloController, TelloStatus
from drone_ai.tracking.face_tracker import FaceTracker
from drone_ai.vision.detector import MediaPipeFaceDetector
from drone_ai.vision.head_pose import MediaPipeHeadPoseEstimator
from drone_ai.vision.embedder import SFaceEmbedder
from drone_ai.vision.overlay import FaceOverlayRenderer
from drone_ai.vision.pipeline import FacePipeline
from drone_ai.vision.recognizer import FaceRecognitionService
from drone_ai.vision.schemas import ApiStatus, FaceDetection, FrameAnalysis, RecognizedFace

if TYPE_CHECKING:
    from drone_ai.gui import DroneAIGUI

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None
    HTTPException = RuntimeError
    BaseModel = object
    uvicorn = None


class RegisterFaceRequest(BaseModel):
    name: str


class DroneApplication:
    """Coordinates drone IO, face recognition, preview, and API."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._controller = TelloController(
            takeoff_extra_rise_cm=config.takeoff_extra_rise_cm
        )
        self._tracker = FaceTracker(config)
        self._repository = SQLiteFaceRepository(config.database_path)
        self._recognizer = FaceRecognitionService(
            self._repository,
            SFaceEmbedder(config.embedder_model_path),
            similarity_threshold=config.recognition_threshold,
            margin_threshold=config.recognition_margin_threshold,
        )
        self._pipeline = FacePipeline(
            MediaPipeFaceDetector(
                min_detection_confidence=config.min_detection_confidence,
                recovery_detection_confidence=config.recovery_detection_confidence,
                detector_model_path=config.detector_model_path,
                nms_threshold=config.detection_nms_threshold,
            ),
            self._recognizer,
            FaceOverlayRenderer(),
        )
        self._head_pose = MediaPipeHeadPoseEstimator(
            enabled=config.tracking_head_pose_enabled,
            min_confidence=config.tracking_head_pose_min_confidence,
            model_path=config.face_landmarker_model_path,
        )
        self._latest_analysis: FrameAnalysis | None = None
        self._latest_detections: list[FaceDetection] = []
        self._latest_frame_id = 0
        self._tracking_enabled = False
        self._tracking_target_visible = False
        self._tracking_target_distance_m: float | None = None
        self._show_head_mesh = False
        self._frame_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._processing_thread: threading.Thread | None = None
        self._api_server: Any | None = None
        self._api_thread: threading.Thread | None = None
        self._api_url: str | None = None
        self._gui: DroneAIGUI | None = None

    def connect(self, *, enable_stream: bool = True) -> TelloStatus:
        status = self._controller.connect(enable_stream=enable_stream)
        if self._processing_thread is None or not self._processing_thread.is_alive():
            self._stop_event.clear()
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                name="droneai-frame-processor",
                daemon=True,
            )
            self._processing_thread.start()
        return status

    def disconnect(self) -> None:
        self._tracking_enabled = False
        self._tracking_target_visible = False
        self._tracking_target_distance_m = None
        self._stop_event.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None
        self._controller.disconnect()

    def takeoff(self) -> TelloStatus:
        return self._controller.takeoff()

    def land(self) -> TelloStatus:
        self._tracking_enabled = False
        self._tracking_target_visible = False
        self._tracking_target_distance_m = None
        return self._controller.land()

    def enable_tracking(self) -> None:
        self._tracking_enabled = True

    def disable_tracking(self) -> None:
        self._tracking_enabled = False
        self._tracking_target_visible = False
        self._tracking_target_distance_m = None
        self._controller.stop_motion()

    def start_api(self, host: str, port: int) -> None:
        if FastAPI is None or uvicorn is None:
            raise RuntimeError(
                "FastAPI or Uvicorn is not installed. Run `uv sync` or `./start.sh` first."
            )

        if self._api_thread and self._api_thread.is_alive():
            return

        api = create_api(self)
        config = uvicorn.Config(api, host=host, port=port, log_level="info")
        self._api_server = uvicorn.Server(config)
        self._api_thread = threading.Thread(
            target=self._api_server.run,
            name="droneai-api",
            daemon=True,
        )
        self._api_thread.start()

        deadline = time.time() + 5
        while time.time() < deadline:
            if self._api_server.started:
                self._api_url = f"http://{host}:{port}"
                return
            if not self._api_thread.is_alive():
                break
            time.sleep(0.05)

        raise RuntimeError("API server failed to start.")

    def stop_api(self) -> None:
        if self._api_server is None:
            return

        self._api_server.should_exit = True
        if self._api_thread is not None:
            self._api_thread.join(timeout=5)
        self._api_server = None
        self._api_thread = None
        self._api_url = None

    def status(self) -> ApiStatus:
        tello_status = self._controller.status()
        identities = self._repository.list_identities()
        with self._frame_lock:
            visible_faces = len(self._latest_analysis.faces) if self._latest_analysis else 0
        return ApiStatus(
            connected=tello_status.connected,
            battery=tello_status.battery,
            stream_enabled=tello_status.stream_enabled,
            flying=tello_status.flying,
            known_identities=len(identities),
            visible_faces=visible_faces,
            tracking_enabled=self._tracking_enabled,
            tracking_target_name=self._tracker.target_name,
            tracking_target_visible=self._tracking_target_visible,
            tracking_target_distance_m=self._tracking_target_distance_m,
            api_url=self._api_url,
        )

    def list_identities(self) -> list[IdentitySummary]:
        return self._repository.list_identities()

    def register_face(self, name: str) -> IdentitySummary:
        with self._frame_lock:
            if self._latest_analysis is None:
                raise RuntimeError("No processed frame is available yet.")
            frame = self._latest_analysis.raw_frame.copy()
            detections = list(self._latest_detections)

        selected_detection = FacePipeline.choose_face(detections)
        return self._recognizer.register_face(name, frame, selected_detection)

    def register_face_series(
        self,
        name: str,
        *,
        sample_count: int = 5,
        timeout_seconds: float = 6.0,
    ) -> IdentitySummary:
        deadline = time.time() + timeout_seconds
        last_frame_id = -1
        captured = 0
        latest_summary: IdentitySummary | None = None

        while captured < sample_count and time.time() < deadline:
            with self._frame_lock:
                analysis = self._latest_analysis
                detections = list(self._latest_detections)
                frame_id = self._latest_frame_id

                if analysis is not None:
                    frame = analysis.raw_frame.copy()
                else:
                    frame = None

            if analysis is None or frame is None or not detections or frame_id == last_frame_id:
                time.sleep(0.08)
                continue

            detection = FacePipeline.choose_face(detections)
            if detection.bounding_box.area < 10_000:
                last_frame_id = frame_id
                time.sleep(0.08)
                continue

            latest_summary = self._recognizer.register_face(name, frame, detection)
            captured += 1
            last_frame_id = frame_id
            time.sleep(0.12)

        if latest_summary is None:
            raise RuntimeError("Failed to capture a usable face series from the current preview.")

        if captured < sample_count:
            raise RuntimeError(
                f"Captured only {captured}/{sample_count} usable samples. Move closer and keep one face visible."
            )

        return latest_summary

    def latest_annotated_frame(self) -> Any:
        with self._frame_lock:
            if self._latest_analysis is None:
                return None
            return self._latest_analysis.annotated_frame.copy()

    def head_mesh_enabled(self) -> bool:
        return self._show_head_mesh

    def toggle_head_mesh(self) -> bool:
        self._show_head_mesh = not self._show_head_mesh
        return self._show_head_mesh

    def latest_faces(self) -> list[RecognizedFace]:
        with self._frame_lock:
            if self._latest_analysis is None:
                return []
            return list(self._latest_analysis.faces)

    def preview_loop(self) -> int:
        try:
            from drone_ai.gui import DroneAIGUI
        except ImportError as exc:
            raise RuntimeError(
                "Tk GUI is not available. Install the system tkinter package and rerun the app."
            ) from exc

        self._gui = DroneAIGUI(
            title=self._config.preview_window_name,
            get_frame=self.latest_annotated_frame,
            get_faces=self.latest_faces,
            get_status=self.status,
            list_identities=self.list_identities,
            register_face=self.register_face,
            register_face_series=self.register_face_series,
            takeoff=self.takeoff,
            land=self.land,
            enable_tracking=self.enable_tracking,
            disable_tracking=self.disable_tracking,
            head_mesh_enabled=self.head_mesh_enabled,
            toggle_head_mesh=self.toggle_head_mesh,
        )
        return self._gui.run()

    def close(self) -> None:
        if self._gui is not None:
            self._gui.close()
            self._gui = None
        self.stop_api()
        self.disconnect()
        self._head_pose.close()
        self._pipeline.close()

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._controller.get_latest_frame()
                analysis = self._pipeline.process_frame(frame)
                tracked_faces = self._apply_tracking(frame, analysis.faces)
                detections = [
                    FaceDetection(
                        bounding_box=face.bounding_box,
                        confidence=face.confidence,
                    )
                    for face in tracked_faces
                ]
                annotated_frame = self._pipeline.render(
                    frame,
                    tracked_faces,
                    show_head_mesh=self._show_head_mesh,
                )
                with self._frame_lock:
                    self._latest_analysis = FrameAnalysis(
                        raw_frame=analysis.raw_frame,
                        annotated_frame=annotated_frame,
                        faces=tracked_faces,
                    )
                    self._latest_detections = detections
                    self._latest_frame_id += 1
            except Exception:
                self._tracking_target_visible = False
                self._tracking_target_distance_m = None
                self._controller.stop_motion()
                time.sleep(0.05)
                continue

    def _apply_tracking(
        self,
        frame_bgr: Any,
        faces: list[RecognizedFace],
    ) -> list[RecognizedFace]:
        target_face = self._tracker.select_target(faces)
        mesh_preview_face = target_face
        if mesh_preview_face is None and self._show_head_mesh and faces:
            mesh_preview_face = max(faces, key=lambda face: face.bounding_box.area)

        head_pose = None
        if mesh_preview_face is not None:
            head_pose = self._head_pose.estimate(frame_bgr, mesh_preview_face.bounding_box)

        enriched_target_face = target_face
        fallback_tracking_anchor_y = None
        fallback_tracking_anchor_source = None
        if target_face is not None:
            fallback_tracking_anchor_y = self._bbox_tracking_anchor_y(target_face.bounding_box)
            fallback_tracking_anchor_source = "bbox"
        if (
            target_face is not None
            and mesh_preview_face is target_face
            and head_pose is not None
        ):
            tracking_anchor_y = head_pose.tracking_anchor_y_px
            tracking_anchor_source = (
                "mesh" if tracking_anchor_y is not None and head_pose.mesh_ready else fallback_tracking_anchor_source
            )
            if tracking_anchor_y is None:
                tracking_anchor_y = fallback_tracking_anchor_y
            enriched_target_face = RecognizedFace(
                bounding_box=target_face.bounding_box,
                confidence=target_face.confidence,
                label=target_face.label,
                similarity=target_face.similarity,
                embedding_ready=target_face.embedding_ready,
                estimated_distance_m=target_face.estimated_distance_m,
                is_tracking_target=target_face.is_tracking_target,
                head_mesh_ready=head_pose.mesh_ready,
                head_pose_ready=head_pose.pose_ready,
                head_yaw_deg=head_pose.yaw_deg,
                head_pitch_deg=head_pose.pitch_deg,
                head_pose_failure_reason=head_pose.failure_reason,
                head_pose_debug=head_pose.debug_message,
                tracking_anchor_y_px=tracking_anchor_y,
                tracking_anchor_source=tracking_anchor_source,
                head_mesh_points=head_pose.mesh_points,
            )
        elif target_face is not None:
            enriched_target_face = RecognizedFace(
                bounding_box=target_face.bounding_box,
                confidence=target_face.confidence,
                label=target_face.label,
                similarity=target_face.similarity,
                embedding_ready=target_face.embedding_ready,
                estimated_distance_m=target_face.estimated_distance_m,
                is_tracking_target=target_face.is_tracking_target,
                head_mesh_ready=target_face.head_mesh_ready,
                head_pose_ready=target_face.head_pose_ready,
                head_yaw_deg=target_face.head_yaw_deg,
                head_pitch_deg=target_face.head_pitch_deg,
                head_pose_failure_reason=target_face.head_pose_failure_reason,
                head_pose_debug=target_face.head_pose_debug,
                tracking_anchor_y_px=fallback_tracking_anchor_y,
                tracking_anchor_source=fallback_tracking_anchor_source,
                head_mesh_points=target_face.head_mesh_points,
            )

        command = self._tracker.build_command_full(
            frame_bgr.shape[1],
            frame_bgr.shape[0],
            enriched_target_face,
        )

        tracked_faces: list[RecognizedFace] = []
        for face in faces:
            is_target = target_face is not None and face is target_face
            has_mesh_preview = mesh_preview_face is not None and face is mesh_preview_face
            estimated_distance_m = None
            head_mesh_ready = False
            head_pose_ready = False
            head_yaw_deg = None
            head_pitch_deg = None
            head_pose_failure_reason = None
            head_pose_debug = None
            tracking_anchor_y_px = None
            tracking_anchor_source = None
            head_mesh_points: tuple[tuple[int, int], ...] = ()
            if is_target:
                estimated_distance_m = command.estimated_distance_m
                tracking_anchor_y_px = (
                    enriched_target_face.tracking_anchor_y_px
                    if enriched_target_face is not None
                    else None
                )
                tracking_anchor_source = (
                    enriched_target_face.tracking_anchor_source
                    if enriched_target_face is not None
                    else None
                )
            if has_mesh_preview and head_pose is not None:
                head_mesh_ready = head_pose.mesh_ready
                head_pose_ready = head_pose.pose_ready
                head_yaw_deg = head_pose.yaw_deg
                head_pitch_deg = head_pose.pitch_deg
                head_pose_failure_reason = head_pose.failure_reason
                head_pose_debug = head_pose.debug_message
                head_mesh_points = head_pose.mesh_points
            tracked_faces.append(
                RecognizedFace(
                    bounding_box=face.bounding_box,
                    confidence=face.confidence,
                    label=face.label,
                    similarity=face.similarity,
                    embedding_ready=face.embedding_ready,
                    estimated_distance_m=estimated_distance_m,
                    is_tracking_target=is_target,
                    head_mesh_ready=head_mesh_ready,
                    head_pose_ready=head_pose_ready,
                    head_yaw_deg=head_yaw_deg,
                    head_pitch_deg=head_pitch_deg,
                    head_pose_failure_reason=head_pose_failure_reason,
                    head_pose_debug=head_pose_debug,
                    tracking_anchor_y_px=tracking_anchor_y_px,
                    tracking_anchor_source=tracking_anchor_source,
                    head_mesh_points=head_mesh_points,
                )
            )

        self._tracking_target_visible = command.target_visible
        self._tracking_target_distance_m = command.estimated_distance_m

        if self._tracking_enabled:
            self._controller.send_rc_control(
                command.left_right_velocity,
                command.forward_backward_velocity,
                command.up_down_velocity,
                command.yaw_velocity,
            )
        else:
            self._controller.stop_motion()

        return tracked_faces

    def _bbox_tracking_anchor_y(self, bounding_box: Any) -> float:
        return bounding_box.y + (
            bounding_box.height * self._config.tracking_bbox_anchor_y_ratio
        )


def create_api(application: DroneApplication) -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Run `uv sync` or `./start.sh` first."
        )

    api = FastAPI(title="DroneAI API", version="0.2.0")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/status")
    def status() -> dict[str, Any]:
        return asdict(application.status())

    @api.get("/identities")
    def identities() -> list[dict[str, Any]]:
        return [asdict(identity) for identity in application.list_identities()]

    @api.post("/connect")
    def connect() -> dict[str, Any]:
        try:
            return asdict(application.connect(enable_stream=True))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.post("/disconnect")
    def disconnect() -> dict[str, str]:
        application.disconnect()
        return {"status": "disconnected"}

    @api.post("/takeoff")
    def takeoff() -> dict[str, Any]:
        try:
            return asdict(application.takeoff())
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.post("/land")
    def land() -> dict[str, Any]:
        try:
            return asdict(application.land())
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.post("/tracking/start")
    def start_tracking() -> dict[str, Any]:
        application.enable_tracking()
        return asdict(application.status())

    @api.post("/tracking/stop")
    def stop_tracking() -> dict[str, Any]:
        application.disable_tracking()
        return asdict(application.status())

    @api.post("/faces/register")
    def register_face(request: RegisterFaceRequest) -> dict[str, Any]:
        try:
            return asdict(application.register_face(request.name))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return api
