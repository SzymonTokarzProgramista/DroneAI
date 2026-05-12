"""SQLite repository for face identities and embeddings."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IdentitySummary:
    identity_id: int
    name: str
    embedding_count: int
    created_at: str


@dataclass(frozen=True)
class StoredEmbedding:
    identity_id: int
    name: str
    embedding: np.ndarray


class SQLiteFaceRepository:
    """Persists named face embeddings in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (identity_id) REFERENCES identities (id) ON DELETE CASCADE
                )
                """
            )

    def add_embedding(self, name: str, embedding: np.ndarray) -> IdentitySummary:
        return self.add_embeddings(name, [embedding])

    def add_embeddings(self, name: str, embeddings: list[np.ndarray]) -> IdentitySummary:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Identity name cannot be empty.")

        vectors = [
            np.asarray(embedding, dtype=np.float32).reshape(-1)
            for embedding in embeddings
        ]
        if not vectors:
            raise ValueError("At least one face embedding is required.")

        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                INSERT INTO identities (name, created_at)
                VALUES (?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (normalized_name, created_at),
            )

            row = connection.execute(
                "SELECT id, name, created_at FROM identities WHERE name = ?",
                (normalized_name,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to load identity after insert.")

            connection.executemany(
                """
                INSERT INTO face_embeddings (identity_id, embedding, dimension, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (row["id"], vector.tobytes(), vector.size, created_at)
                    for vector in vectors
                ],
            )

        return self.get_identity_summary(normalized_name)

    def get_identity_summary(self, name: str) -> IdentitySummary:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT i.id, i.name, i.created_at, COUNT(fe.id) AS embedding_count
                FROM identities i
                LEFT JOIN face_embeddings fe ON fe.identity_id = i.id
                WHERE i.name = ?
                GROUP BY i.id, i.name, i.created_at
                """,
                (name,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Identity '{name}' does not exist.")

        return IdentitySummary(
            identity_id=row["id"],
            name=row["name"],
            embedding_count=row["embedding_count"],
            created_at=row["created_at"],
        )

    def list_identities(self) -> list[IdentitySummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.id, i.name, i.created_at, COUNT(fe.id) AS embedding_count
                FROM identities i
                LEFT JOIN face_embeddings fe ON fe.identity_id = i.id
                GROUP BY i.id, i.name, i.created_at
                ORDER BY i.name ASC
                """
            ).fetchall()

        return [
            IdentitySummary(
                identity_id=row["id"],
                name=row["name"],
                embedding_count=row["embedding_count"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def load_embeddings(self) -> list[StoredEmbedding]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.id AS identity_id, i.name, fe.embedding, fe.dimension
                FROM face_embeddings fe
                JOIN identities i ON i.id = fe.identity_id
                ORDER BY i.name ASC, fe.id ASC
                """
            ).fetchall()

        embeddings: list[StoredEmbedding] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32, count=row["dimension"]).copy()
            embeddings.append(
                StoredEmbedding(
                    identity_id=row["identity_id"],
                    name=row["name"],
                    embedding=vector,
                )
            )
        return embeddings
