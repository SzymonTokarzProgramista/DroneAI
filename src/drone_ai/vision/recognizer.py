"""Face recognition service."""

from __future__ import annotations

from threading import RLock

import numpy as np

from drone_ai.storage.face_repository import IdentitySummary, SQLiteFaceRepository, StoredEmbedding
from drone_ai.vision.embedder import SFaceEmbedder
from drone_ai.vision.types import FaceDetection, RecognizedFace


class FaceRecognitionService:
    """Handles embedding extraction, gallery matching, and enrollment."""

    def __init__(
        self,
        repository: SQLiteFaceRepository,
        embedder: SFaceEmbedder,
        *,
        similarity_threshold: float,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._lock = RLock()
        self._gallery: list[StoredEmbedding] = []
        self.reload_gallery()

    def reload_gallery(self) -> None:
        with self._lock:
            self._gallery = self._repository.load_embeddings()

    def list_identities(self) -> list[IdentitySummary]:
        return self._repository.list_identities()

    def register_face(self, name: str, frame_bgr: np.ndarray, detection: FaceDetection) -> IdentitySummary:
        embedding = self._embedder.embed(frame_bgr, detection.bounding_box)
        summary = self._repository.add_embedding(name, embedding)
        self.reload_gallery()
        return summary

    def recognize_faces(
        self, frame_bgr: np.ndarray, detections: list[FaceDetection]
    ) -> list[RecognizedFace]:
        recognized: list[RecognizedFace] = []
        for detection in detections:
            recognized.append(self._recognize_single(frame_bgr, detection))
        return recognized

    def _recognize_single(
        self, frame_bgr: np.ndarray, detection: FaceDetection
    ) -> RecognizedFace:
        try:
            query_embedding = self._embedder.embed(frame_bgr, detection.bounding_box)
        except Exception:
            return RecognizedFace(
                bounding_box=detection.bounding_box,
                confidence=detection.confidence,
                label="embedding-error",
                similarity=None,
                embedding_ready=False,
            )

        best_name = "unknown"
        best_similarity: float | None = None

        with self._lock:
            gallery = list(self._gallery)

        for candidate in gallery:
            similarity = self._embedder.cosine_similarity(query_embedding, candidate.embedding)
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_name = candidate.name

        if best_similarity is None or best_similarity < self._similarity_threshold:
            best_name = "unknown"

        return RecognizedFace(
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            label=best_name,
            similarity=best_similarity,
            embedding_ready=True,
        )
