from __future__ import annotations
from core import config


def get_llm():
    provider = config.LLM_PROVIDER.lower()
    if provider == "local":
        from .local import LocalTransformersLLM
        return LocalTransformersLLM()
    if provider in {"openai", "openai_compatible", "api"}:
        from .openai_compatible import OpenAICompatibleLLM
        return OpenAICompatibleLLM()
    raise ValueError(f"未知 LLM provider: {config.LLM_PROVIDER}")
