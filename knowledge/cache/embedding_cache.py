"""SQLite Embedding cache，避免重复对相同文本编码。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import Lock


class EmbeddingCache:
    def __init__(self, db_path: str | Path, namespace: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._lock = Lock()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "namespace TEXT NOT NULL, key TEXT NOT NULL, vector TEXT NOT NULL, "
                "PRIMARY KEY(namespace, key))"
            )

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, texts: list[str]) -> dict[int, list[float]]:
        if not texts:
            return {}
        keys = [self._key(t) for t in texts]
        placeholders = ",".join("?" for _ in keys)
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT key, vector FROM embeddings WHERE namespace=? AND key IN ({placeholders})",
                [self.namespace, *keys],
            ).fetchall()
        mapping = {k: json.loads(v) for k, v in rows}
        return {i: mapping[k] for i, k in enumerate(keys) if k in mapping}

    def put_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        rows = [(self.namespace, self._key(t), json.dumps(v, separators=(",", ":"))) for t, v in zip(texts, vectors)]
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.executemany("INSERT OR REPLACE INTO embeddings(namespace,key,vector) VALUES(?,?,?)", rows)
