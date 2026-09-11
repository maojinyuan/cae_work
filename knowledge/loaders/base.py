from __future__ import annotations
from pathlib import Path
from typing import Protocol
from core.types import DocumentBlock


class DocumentLoader(Protocol):
    def load(self, path: Path) -> list[DocumentBlock]: ...
