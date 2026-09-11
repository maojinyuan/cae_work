"""检查离线模型目录与关键依赖。"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import importlib.util
from core import config


def yesno(v): return "OK" if v else "MISSING"


def main():
    print("=== CAE AI Platform offline check ===")
    for pkg in ["chromadb", "torch", "transformers", "sentence_transformers", "fastapi"]:
        print(f"{pkg:24s}: {yesno(importlib.util.find_spec(pkg) is not None)}")
    print(f"LLM provider             : {config.LLM_PROVIDER}")
    print(f"LLM path                 : {config.LLM_MODEL_PATH} [{yesno(config.LLM_MODEL_PATH.exists())}]")
    print(f"Embedding provider       : {config.EMBED_PROVIDER}")
    print(f"Embedding path           : {config.EMBED_MODEL_PATH} [{yesno(config.EMBED_MODEL_PATH.exists())}]")
    print(f"Retrieval mode           : {config.RETRIEVAL_MODE}")
    print(f"Reranker enabled         : {config.RERANK_ENABLED}")
    if config.RERANK_ENABLED:
        print(f"Reranker path            : {config.RERANK_MODEL_PATH} [{yesno(config.RERANK_MODEL_PATH.exists())}]")
    try:
        import torch
        print(f"CUDA available           : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device              : {torch.cuda.get_device_name(0)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
