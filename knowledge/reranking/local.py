"""可选本地 CrossEncoder reranker。"""
from __future__ import annotations
from pathlib import Path
from core import config


class LocalReranker:
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or config.RERANK_MODEL_PATH).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Reranker 模型目录不存在: {self.model_path}")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Reranker 需要 sentence-transformers 和 torch。") from exc
        device = config.RERANK_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(str(self.model_path), device=device, local_files_only=True)

    def rerank(self, query, hits, top_k):
        if not hits:
            return []
        scores = self.model.predict(
            [(query, h.text) for h in hits],
            batch_size=config.RERANK_BATCH,
            show_progress_bar=False,
        )
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        out = []
        for hit, score in ranked[:top_k]:
            hit.score = round(float(score), 6)
            out.append(hit)
        return out
