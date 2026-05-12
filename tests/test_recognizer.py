from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_ai.storage.face_repository import IdentitySummary, StoredEmbedding
from drone_ai.vision.recognizer import FaceRecognitionService
from drone_ai.vision.schemas import BoundingBox, FaceDetection


class _FakeRepository:
    def __init__(self) -> None:
        self.embeddings: list[StoredEmbedding] = []
        self.added_batches: list[list[np.ndarray]] = []

    def load_embeddings(self) -> list[StoredEmbedding]:
        return list(self.embeddings)

    def list_identities(self) -> list[IdentitySummary]:
        return []

    def add_embeddings(self, name: str, embeddings: list[np.ndarray]) -> IdentitySummary:
        normalized_name = name.strip()
        batch = [np.asarray(embedding, dtype=np.float32) for embedding in embeddings]
        self.added_batches.append(batch)
        self.embeddings.extend(
            StoredEmbedding(
                identity_id=1,
                name=normalized_name,
                embedding=embedding,
            )
            for embedding in batch
        )
        return IdentitySummary(
            identity_id=1,
            name=normalized_name,
            embedding_count=len(self.embeddings),
            created_at="now",
        )


class _FakeEmbedder:
    def __init__(self) -> None:
        self.embed_calls = 0
        self.embed_variants_calls = 0

    def embed(self, frame_bgr: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
        self.embed_calls += 1
        return np.array([0.0, 1.0], dtype=np.float32)

    def embed_variants(self, frame_bgr: np.ndarray, bounding_box: BoundingBox) -> list[np.ndarray]:
        self.embed_variants_calls += 1
        return [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.1], dtype=np.float32),
        ]

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))


class FaceRecognitionServiceTests(unittest.TestCase):
    def _make_service(self) -> tuple[FaceRecognitionService, _FakeRepository, _FakeEmbedder]:
        repository = _FakeRepository()
        embedder = _FakeEmbedder()
        service = FaceRecognitionService(
            repository,
            embedder,
            similarity_threshold=0.5,
            margin_threshold=0.03,
        )
        return service, repository, embedder

    def test_register_face_uses_one_shot_embedding_variants_by_default(self) -> None:
        service, repository, embedder = self._make_service()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = FaceDetection(BoundingBox(x=10, y=10, width=50, height=50), 0.99)

        summary = service.register_face(" Maks ", frame, detection)

        self.assertEqual(embedder.embed_variants_calls, 1)
        self.assertEqual(embedder.embed_calls, 0)
        self.assertEqual(len(repository.added_batches[0]), 2)
        self.assertEqual(summary.name, "Maks")
        self.assertEqual(summary.embedding_count, 2)

    def test_register_face_can_skip_variants_for_series_capture(self) -> None:
        service, repository, embedder = self._make_service()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = FaceDetection(BoundingBox(x=10, y=10, width=50, height=50), 0.99)

        summary = service.register_face(
            "Maks",
            frame,
            detection,
            augment_from_single_frame=False,
        )

        self.assertEqual(embedder.embed_variants_calls, 0)
        self.assertEqual(embedder.embed_calls, 1)
        self.assertEqual(len(repository.added_batches[0]), 1)
        self.assertEqual(summary.embedding_count, 1)


if __name__ == "__main__":
    unittest.main()
