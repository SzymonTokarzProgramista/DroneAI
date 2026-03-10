"""Face recognition service."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from drone_ai.storage.face_repository import IdentitySummary, SQLiteFaceRepository, StoredEmbedding
from drone_ai.vision.embedder import SFaceEmbedder
from drone_ai.vision.schemas import FaceDetection, RecognizedFace


@dataclass(frozen=True)
class IdentityPrototype:
    name: str
    embedding: np.ndarray
    sample_count: int


class FaceRecognitionService:
    """Handles embedding extraction, gallery matching, and enrollment."""

    def __init__(
        self,
        repository: SQLiteFaceRepository,
        embedder: SFaceEmbedder,
        *,
        similarity_threshold: float,
        margin_threshold: float,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._margin_threshold = margin_threshold
        self._lock = RLock()
        self._gallery: list[StoredEmbedding] = []
        self._prototypes: list[IdentityPrototype] = []
        self._grouped_gallery: dict[str, list[np.ndarray]] = {}
        self.reload_gallery()

    def reload_gallery(self) -> None:
        with self._lock:
            self._gallery = self._repository.load_embeddings()
            self._grouped_gallery = self._group_embeddings(self._gallery)
            self._prototypes = self._build_prototypes(self._gallery)

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
        second_best_similarity: float | None = None

        with self._lock:
            prototypes = {prototype.name: prototype for prototype in self._prototypes}
            grouped_gallery = {
                name: list(embeddings)
                for name, embeddings in self._grouped_gallery.items()
            }

        for candidate_name, embeddings in grouped_gallery.items():
            prototype = prototypes.get(candidate_name)
            sample_similarities = sorted(
                (
                    self._embedder.cosine_similarity(query_embedding, embedding)
                    for embedding in embeddings
                ),
                reverse=True,
            )
            if not sample_similarities:
                continue

            top_k = sample_similarities[: min(3, len(sample_similarities))]
            similarity = float(np.mean(top_k))
            best_single_similarity = top_k[0]
            similarity = 0.65 * best_single_similarity + 0.35 * similarity

            if prototype is not None:
                prototype_similarity = self._embedder.cosine_similarity(
                    query_embedding,
                    prototype.embedding,
                )
                similarity = max(similarity, prototype_similarity)

            if best_similarity is None or similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_name = candidate_name
            elif second_best_similarity is None or similarity > second_best_similarity:
                second_best_similarity = similarity

        if best_similarity is None or best_similarity < self._similarity_threshold:
            best_name = "unknown"
        elif second_best_similarity is not None:
            if (best_similarity - second_best_similarity) < self._margin_threshold:
                best_name = "unknown"

        if best_name == "unknown" and best_similarity is not None and best_similarity < 0:
            best_name = "unknown"

        return RecognizedFace(
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            label=best_name,
            similarity=best_similarity,
            embedding_ready=True,
        )

    @staticmethod
    def _build_prototypes(gallery: list[StoredEmbedding]) -> list[IdentityPrototype]:
        prototypes: list[IdentityPrototype] = []
        for name, embeddings in FaceRecognitionService._group_embeddings(gallery).items():
            stacked = np.stack(embeddings, axis=0)
            centroid = stacked.mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm == 0.0:
                continue
            prototypes.append(
                IdentityPrototype(
                    name=name,
                    embedding=centroid / norm,
                    sample_count=len(embeddings),
                )
            )

        return sorted(prototypes, key=lambda prototype: prototype.name)

    @staticmethod
    def _group_embeddings(gallery: list[StoredEmbedding]) -> dict[str, list[np.ndarray]]:
        grouped: dict[str, list[np.ndarray]] = {}
        for item in gallery:
            grouped.setdefault(item.name, []).append(item.embedding)
        return grouped
