from __future__ import annotations
from typing import Protocol


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_one(self, text: str) -> list[float]: ...
