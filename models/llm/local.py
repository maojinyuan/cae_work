"""离线本地 LLM provider：Hugging Face Transformers。"""
from __future__ import annotations

from pathlib import Path
from core import config


class LocalTransformersLLM:
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or config.LLM_MODEL_PATH).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"本地 LLM 模型目录不存在: {self.model_path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("本地 LLM 需要 torch、transformers、accelerate。") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=config.LLM_TRUST_REMOTE_CODE
        )
        device = self._resolve_device(torch)
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": config.LLM_TRUST_REMOTE_CODE,
            "torch_dtype": self._resolve_dtype(torch),
        }
        # device_map=auto 适合大模型跨 GPU/CPU 调度；未指定时单设备加载更可控。
        if config.LLM_DEVICE_MAP:
            kwargs["device_map"] = config.LLM_DEVICE_MAP
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        if not config.LLM_DEVICE_MAP:
            self.model.to(device)
        self.model.eval()
        self.device = device

    @staticmethod
    def _resolve_device(torch) -> str:
        requested = config.LLM_DEVICE.lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CAE_LLM_DEVICE 指定 CUDA，但当前环境没有可用 CUDA。")
        return requested

    @staticmethod
    def _resolve_dtype(torch):
        if config.LLM_DTYPE == "auto":
            return "auto"
        dtype = getattr(torch, config.LLM_DTYPE, None)
        if dtype is None:
            raise ValueError(f"不支持的 CAE_LLM_DTYPE={config.LLM_DTYPE}")
        return dtype

    def _build_inputs(self, messages: list[dict]):
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=config.LLM_ENABLE_THINKING,
            )
        else:
            prompt = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages) + "\nassistant:"
            encoded = self.tokenizer(prompt, return_tensors="pt")

        max_input = config.LLM_MAX_INPUT_TOKENS
        if max_input > 0 and encoded["input_ids"].shape[-1] > max_input:
            for key, value in list(encoded.items()):
                if getattr(value, "ndim", 0) == 2:
                    encoded[key] = value[:, -max_input:]
        return encoded

    def generate(self, messages: list[dict], max_tokens: int | None = None) -> str:
        encoded = self._build_inputs(messages)
        model_device = next(self.model.parameters()).device
        encoded = {k: v.to(model_device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[-1]
        temperature = config.LLM_TEMPERATURE
        kwargs = {
            "max_new_tokens": max_tokens or config.LLM_MAX_TOKENS,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=config.LLM_TOP_P)
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **kwargs)
        return self.tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()
