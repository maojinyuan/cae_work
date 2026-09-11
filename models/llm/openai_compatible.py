from __future__ import annotations
from openai import OpenAI
from core import config


class OpenAICompatibleLLM:
    def __init__(self):
        if not config.LLM_BASE_URL or not config.LLM_MODEL:
            raise RuntimeError("API LLM 需要 CAE_LLM_BASE_URL 和 CAE_LLM_MODEL")
        # vLLM/SGLang 等内部服务通常不校验 key；OpenAI SDK 仍要求传一个字符串。
        self.client = OpenAI(api_key=config.LLM_API_KEY or "not-needed", base_url=config.LLM_BASE_URL)

    def generate(self, messages, max_tokens=None):
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()
