from __future__ import annotations
from core import config


def get_embedding_model():
    provider = config.EMBED_PROVIDER.lower()
    if provider == "local":
        from .local import LocalSentenceTransformerEmbedding
        return LocalSentenceTransformerEmbedding()
    if provider in {"api", "openai", "openai_compatible"}:
        from .api import APIEmbedding
        if not config.EMBED_BASE_URL or not config.EMBED_MODEL:
            raise RuntimeError("API Embedding 需要 CAE_EMBED_BASE_URL 和 CAE_EMBED_MODEL")
        return APIEmbedding()
    raise ValueError(f"未知 Embedding provider: {config.EMBED_PROVIDER}；支持 local / api")


_embedding = None

def get_embedder():
    global _embedding
    if _embedding is None:
        _embedding = get_embedding_model()
    return _embedding
