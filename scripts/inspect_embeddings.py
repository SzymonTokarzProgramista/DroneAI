#!/usr/bin/env python3
"""Inspect face embedding quality stored in SQLite."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingRecord:
    embedding_id: int
    identity_id: int
    name: str
    embedding: np.ndarray
    created_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect face embedding separation in SQLite.")
    parser.add_argument(
        "--db",
        default="data/drone_ai.sqlite3",
        help="Path to the SQLite database file.",
    )
    return parser.parse_args()


def load_embeddings(database_path: Path) -> list[EmbeddingRecord]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT
                fe.id AS embedding_id,
                i.id AS identity_id,
                i.name AS name,
                fe.embedding AS embedding,
                fe.dimension AS dimension,
                fe.created_at AS created_at
            FROM face_embeddings fe
            JOIN identities i ON i.id = fe.identity_id
            ORDER BY i.name ASC, fe.id ASC
            """
        ).fetchall()
    finally:
        connection.close()

    records: list[EmbeddingRecord] = []
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32, count=row["dimension"]).copy()
        norm = np.linalg.norm(vector)
        if norm != 0.0:
            vector = vector / norm
        records.append(
            EmbeddingRecord(
                embedding_id=row["embedding_id"],
                identity_id=row["identity_id"],
                name=row["name"],
                embedding=vector,
                created_at=row["created_at"],
            )
        )
    return records


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def build_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    centroid = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm == 0.0:
        return centroid
    return centroid / norm


def summarize_within_class(records_by_name: dict[str, list[EmbeddingRecord]]) -> list[str]:
    lines = ["Within-class cohesion:"]
    for name, records in sorted(records_by_name.items()):
        embeddings = [record.embedding for record in records]
        centroid = build_centroid(embeddings)
        similarities_to_centroid = [cosine_similarity(embedding, centroid) for embedding in embeddings]

        pairwise: list[float] = []
        for index, left in enumerate(embeddings):
            for right in embeddings[index + 1 :]:
                pairwise.append(cosine_similarity(left, right))

        mean_pairwise = float(np.mean(pairwise)) if pairwise else 1.0
        min_pairwise = float(np.min(pairwise)) if pairwise else 1.0
        mean_centroid = float(np.mean(similarities_to_centroid))
        min_centroid = float(np.min(similarities_to_centroid))

        lines.append(
            f"- {name}: samples={len(records)}, mean_pairwise={mean_pairwise:.4f}, "
            f"min_pairwise={min_pairwise:.4f}, mean_to_centroid={mean_centroid:.4f}, "
            f"min_to_centroid={min_centroid:.4f}"
        )
    return lines


def summarize_between_class(records_by_name: dict[str, list[EmbeddingRecord]]) -> list[str]:
    centroids = {
        name: build_centroid([record.embedding for record in records])
        for name, records in records_by_name.items()
    }

    lines = ["Between-class centroid similarity:"]
    names = sorted(centroids)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            similarity = cosine_similarity(centroids[left_name], centroids[right_name])
            lines.append(f"- {left_name} vs {right_name}: centroid_cos={similarity:.4f}")
    return lines


def summarize_confusions(records_by_name: dict[str, list[EmbeddingRecord]]) -> list[str]:
    centroids = {
        name: build_centroid([record.embedding for record in records])
        for name, records in records_by_name.items()
    }

    lines = ["Nearest competing identity per class:"]
    for name, records in sorted(records_by_name.items()):
        own_centroid = centroids[name]
        own_similarities = [cosine_similarity(record.embedding, own_centroid) for record in records]
        mean_own = float(np.mean(own_similarities))

        nearest_name = None
        nearest_similarity = -1.0
        for other_name, other_centroid in centroids.items():
            if other_name == name:
                continue
            similarity = cosine_similarity(own_centroid, other_centroid)
            if similarity > nearest_similarity:
                nearest_similarity = similarity
                nearest_name = other_name

        margin = mean_own - nearest_similarity
        lines.append(
            f"- {name}: mean_own={mean_own:.4f}, nearest_other={nearest_name}, "
            f"nearest_cos={nearest_similarity:.4f}, margin={margin:.4f}"
        )
    return lines


def summarize_outliers(records_by_name: dict[str, list[EmbeddingRecord]]) -> list[str]:
    lines = ["Potential outliers:"]
    found = False

    for name, records in sorted(records_by_name.items()):
        centroid = build_centroid([record.embedding for record in records])
        for record in records:
            own_similarity = cosine_similarity(record.embedding, centroid)

            best_other_name = None
            best_other_similarity = -1.0
            for other_name, other_records in records_by_name.items():
                if other_name == name:
                    continue
                other_centroid = build_centroid([item.embedding for item in other_records])
                similarity = cosine_similarity(record.embedding, other_centroid)
                if similarity > best_other_similarity:
                    best_other_similarity = similarity
                    best_other_name = other_name

            if best_other_similarity >= own_similarity:
                found = True
                lines.append(
                    f"- embedding_id={record.embedding_id} ({name}) looks suspicious: "
                    f"own_centroid={own_similarity:.4f}, other={best_other_name}:{best_other_similarity:.4f}, "
                    f"created_at={record.created_at}"
                )

    if not found:
        lines.append("- none")

    return lines


def main() -> int:
    args = parse_args()
    database_path = Path(args.db)
    if not database_path.exists():
        raise SystemExit(f"Database file does not exist: {database_path}")

    records = load_embeddings(database_path)
    if not records:
        raise SystemExit("No embeddings found in the database.")

    records_by_name: dict[str, list[EmbeddingRecord]] = defaultdict(list)
    for record in records:
        records_by_name[record.name].append(record)

    sections = [
        [f"Database: {database_path}", f"Embeddings: {len(records)}", f"Identities: {len(records_by_name)}"],
        summarize_within_class(records_by_name),
        summarize_between_class(records_by_name),
        summarize_confusions(records_by_name),
        summarize_outliers(records_by_name),
    ]

    for section in sections:
        for line in section:
            print(line)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
