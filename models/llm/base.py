from __future__ import annotations
from typing import Protocol


class LLM(Protocol):
    def generate(self, messages: list[dict], max_tokens: int | None = None) -> str: ...
