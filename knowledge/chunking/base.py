from __future__ import annotations
from typing import Protocol
from core.types import DocumentBlock, DocumentChunk


class Chunker(Protocol):
    def chunk(self, blocks: list[DocumentBlock], source: str) -> list[DocumentChunk]: ...
