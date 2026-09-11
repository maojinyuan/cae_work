"""OpenAI-compatible Embedding API。"""
from __future__ import annotations
from openai import OpenAI
from core import config


class APIEmbedding:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or config.EMBED_MODEL
        self.client = OpenAI(api_key=api_key or config.EMBED_API_KEY or "not-needed", base_url=base_url or config.EMBED_BASE_URL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = []
        for start in range(0, len(texts), config.EMBED_BATCH):
            batch = texts[start:start + config.EMBED_BATCH]
            result = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(item.embedding for item in sorted(result.data, key=lambda x: x.index))
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
