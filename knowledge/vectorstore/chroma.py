"""ChromaDB 持久化向量库。"""
from __future__ import annotations
import hashlib
import chromadb
from chromadb.config import Settings
from core import config
from core.types import DocumentChunk, RetrievalResult


class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))

    @property
    def collection(self):
        return self.client.get_or_create_collection(name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _chunk_id(source: str, text: str, idx: int) -> str:
        h = hashlib.sha1(f"{source}\x00{idx}\x00{text}".encode()).hexdigest()
        return f"{h[:24]}-{idx}"

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def reset(self) -> None:
        try:
            self.client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass

    @staticmethod
    def _clean_meta(meta: dict) -> dict:
        return {k: v for k, v in (meta or {}).items() if v is not None and isinstance(v, (str, int, float, bool))}

    def upsert(self, chunks, embeddings, source: str) -> int:
        if not chunks:
            return 0
        self.delete_source(source)
        batch = max(1, config.INGEST_BATCH_SIZE)
        for start in range(0, len(chunks), batch):
            cs = chunks[start:start + batch]
            es = embeddings[start:start + batch]
            ids = [self._chunk_id(source, c.text, start + i) for i, c in enumerate(cs)]
            self.collection.add(
                ids=ids,
                documents=[c.text for c in cs],
                embeddings=es,
                metadatas=[self._clean_meta(c.metadata) for c in cs],
            )
        return len(chunks)

    def query(self, embedding, top_k: int, threshold=None):
        if self.collection.count() == 0:
            return []
        n = min(top_k, self.collection.count())
        res = self.collection.query(query_embeddings=[embedding], n_results=n, include=["documents", "metadatas", "distances"])
        out = []
        for text, meta, dist in zip((res.get("documents") or [[]])[0], (res.get("metadatas") or [[]])[0], (res.get("distances") or [[]])[0]):
            score = 1.0 - float(dist)
            if threshold is not None and score < threshold:
                continue
            meta = meta or {}
            out.append(RetrievalResult(text, meta.get("source", ""), meta.get("page"), meta.get("section", ""), meta.get("kind", "text"), round(score, 6)))
        return out

    def all_records(self) -> list[RetrievalResult]:
        total = self.collection.count()
        if total == 0:
            return []
        out = []
        offset = 0
        limit = 5000
        while offset < total:
            res = self.collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for text, meta in zip(docs, metas):
                meta = meta or {}
                out.append(RetrievalResult(text, meta.get("source", ""), meta.get("page"), meta.get("section", ""), meta.get("kind", "text"), 0.0))
            offset += len(docs)
            if not docs:
                break
        return out

    def stats(self):
        return {"collection": config.COLLECTION_NAME, "count": self.collection.count(), "path": str(config.CHROMA_DIR)}


_store = None

def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
