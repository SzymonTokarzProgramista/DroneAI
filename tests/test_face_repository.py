from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_ai.storage.face_repository import SQLiteFaceRepository


class SQLiteFaceRepositoryTests(unittest.TestCase):
    def test_add_embeddings_stores_batch_for_one_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = SQLiteFaceRepository(Path(temp_dir) / "faces.sqlite3")
            embeddings = [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.9, 0.1, 0.0], dtype=np.float32),
            ]

            summary = repository.add_embeddings(" Maks ", embeddings)

            self.assertEqual(summary.name, "Maks")
            self.assertEqual(summary.embedding_count, 2)
            loaded = repository.load_embeddings()
            self.assertEqual(len(loaded), 2)
            np.testing.assert_array_equal(loaded[0].embedding, embeddings[0])
            np.testing.assert_array_equal(loaded[1].embedding, embeddings[1])

    def test_add_embeddings_rejects_empty_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = SQLiteFaceRepository(Path(temp_dir) / "faces.sqlite3")

            with self.assertRaises(ValueError):
                repository.add_embeddings("Maks", [])


if __name__ == "__main__":
    unittest.main()
