"""平台层共享的数据结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentBlock:
    text: str
    kind: str = "paragraph"
    level: int = 0
    page: int | None = None


@dataclass
class DocumentChunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    text: str
    source: str
    page: int | None
    section: str
    kind: str
    score: float
