"""本地 SentenceTransformers Embedding provider。"""
from __future__ import annotations

from pathlib import Path

from core import config
from knowledge.cache import EmbeddingCache


class LocalSentenceTransformerEmbedding:
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or config.EMBED_MODEL_PATH).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"本地 Embedding 模型目录不存在: {self.model_path}")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("本地 Embedding 需要 torch 和 sentence-transformers。") from exc

        self.device = self._resolve_device(torch)
        self.model = SentenceTransformer(
            str(self.model_path),
            device=self.device,
            local_files_only=True,
            trust_remote_code=config.EMBED_TRUST_REMOTE_CODE,
        )
        # 避免极长工程文档片段拖慢编码；可由环境变量覆盖。
        if config.EMBED_MAX_LENGTH > 0:
            self.model.max_seq_length = min(config.EMBED_MAX_LENGTH, getattr(self.model, "max_seq_length", config.EMBED_MAX_LENGTH))
        namespace = f"st::{self.model_path}::{config.EMBED_NORMALIZE}::{self.model.max_seq_length}"
        self.cache = EmbeddingCache(config.EMBED_CACHE_DB, namespace) if config.EMBED_CACHE_ENABLED else None

    @staticmethod
    def _resolve_device(torch) -> str:
        requested = config.EMBED_DEVICE.lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CAE_EMBED_DEVICE 指定 CUDA，但当前环境没有可用 CUDA。")
        return requested

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=config.EMBED_BATCH,
            normalize_embeddings=config.EMBED_NORMALIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.cache is None:
            return self._encode(texts)

        cached = self.cache.get_many(texts)
        out: list[list[float] | None] = [cached.get(i) for i in range(len(texts))]
        miss_idx = [i for i, v in enumerate(out) if v is None]
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            miss_vectors = self._encode(miss_texts)
            self.cache.put_many(miss_texts, miss_vectors)
            for i, vec in zip(miss_idx, miss_vectors):
                out[i] = vec
        return [v for v in out if v is not None]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
