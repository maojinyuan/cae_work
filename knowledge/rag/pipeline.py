from __future__ import annotations

import hashlib
import json
from pathlib import Path
from core import config
from knowledge import loaders
from knowledge.chunking import CAEChunker
from knowledge.embedding import get_embedder
from knowledge.vectorstore import get_store
from knowledge.retrieval import Retriever
from knowledge.rag.prompt import build_messages, format_sources
from models.llm import get_llm


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class IngestManifest:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text("utf-8")) if path.exists() else {}
        except Exception:
            self.data = {}

    def unchanged(self, source: str, digest: str) -> bool:
        return self.data.get(source, {}).get("sha256") == digest

    def update(self, source: str, digest: str, chunks: int) -> None:
        self.data[source] = {"sha256": digest, "chunks": chunks}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), "utf-8")

    def clear(self) -> None:
        self.data = {}
        if self.path.exists():
            self.path.unlink()


class RAGPipeline:
    def __init__(self, chunker=None, embedder=None, store=None, llm=None):
        self.chunker = chunker or CAEChunker()
        self.embedder = embedder or get_embedder()
        self.store = store or get_store()
        self.retriever = Retriever(self.embedder, self.store)
        self.llm = llm
        self.manifest = IngestManifest(config.INGEST_MANIFEST)

    def ingest_file(self, path, source: str | None = None, force: bool = False):
        path = Path(path)
        source = source or path.name
        digest = _sha256(path)
        if config.INGEST_SKIP_UNCHANGED and not force and self.manifest.unchanged(source, digest):
            return {"file": source, "chunks": 0, "status": "unchanged"}

        blocks = loaders.load_file(path)
        if not blocks:
            return {"file": source, "chunks": 0, "status": "skipped"}
        chunks = self.chunker.chunk(blocks, source)
        if not chunks:
            return {"file": source, "chunks": 0, "status": "empty"}
        vectors = self.embedder.embed([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embedding 数量异常: chunks={len(chunks)}, vectors={len(vectors)}")
        n = self.store.upsert(chunks, vectors, source)
        self.manifest.update(source, digest, n)
        return {"file": source, "chunks": n, "status": "ok"}

    def ingest_path(self, path, recreate=False, force=False):
        root = Path(path).resolve()
        if recreate:
            self.store.reset()
            self.manifest.clear()
        imported, skipped, errors = [], [], []
        for file in loaders.iter_documents(root):
            try:
                source = file.name if root.is_file() else str(file.resolve().relative_to(root))
                result = self.ingest_file(file, source=source, force=force)
                (imported if result["status"] == "ok" else skipped).append(result)
            except Exception as exc:
                errors.append({"file": str(file), "error": str(exc)})
        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "files_imported": len(imported),
            "total_chunks": sum(x["chunks"] for x in imported),
        }

    def search(self, question, top_k=None, threshold=None):
        return self.retriever.retrieve(question, top_k, threshold)

    def ask(self, question, top_k=None, threshold=None):
        hits = self.search(question, top_k, threshold)
        if not hits:
            return {"answer": "资料库中未检索到相关内容。请确认相关文档已导入，或换一种问法。", "sources": [], "retrieved": 0}
        if self.llm is None:
            self.llm = get_llm()
        answer = self.llm.generate(build_messages(question, hits))
        return {"answer": answer, "sources": format_sources(hits), "retrieved": len(hits)}


_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
