"""CAE AI Platform 全局配置。

设计原则：
1. 默认完全离线：本地 LLM + 本地 Embedding。
2. 通过环境变量切换 API provider，不改业务代码。
3. `.env` 自动加载；所有 Hugging Face 调用强制 local_files_only。
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


DATA_DIR = _path("CAE_DATA_DIR", PROJECT_ROOT / "data" / "documents")
STORAGE_DIR = _path("CAE_STORAGE_DIR", PROJECT_ROOT / "storage")
CHROMA_DIR = STORAGE_DIR / "chroma"
CACHE_DIR = _path("CAE_CACHE_DIR", PROJECT_ROOT / ".cache")
MODEL_DIR = _path("CAE_MODEL_DIR", PROJECT_ROOT / "artifacts" / "models")

# -------------------- LLM --------------------
LLM_PROVIDER = os.getenv("CAE_LLM_PROVIDER", "local").lower()  # local | api
LLM_MODEL_PATH = _path("CAE_LLM_MODEL_PATH", MODEL_DIR / "llm" / "YOUR_LLM_MODEL")
LLM_DEVICE = os.getenv("CAE_LLM_DEVICE", "auto")
LLM_DEVICE_MAP = os.getenv("CAE_LLM_DEVICE_MAP", "").strip()
LLM_DTYPE = os.getenv("CAE_LLM_DTYPE", "auto")
LLM_TRUST_REMOTE_CODE = _bool("CAE_LLM_TRUST_REMOTE_CODE", False)
LLM_MAX_TOKENS = int(os.getenv("CAE_LLM_MAX_TOKENS", "1024"))
LLM_MAX_INPUT_TOKENS = int(os.getenv("CAE_LLM_MAX_INPUT_TOKENS", "0"))  # 0 = tokenizer/model decides
LLM_TEMPERATURE = float(os.getenv("CAE_LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("CAE_LLM_TOP_P", "0.9"))
LLM_ENABLE_THINKING = _bool("CAE_LLM_ENABLE_THINKING", False)
LLM_BASE_URL = os.getenv("CAE_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_API_KEY = os.getenv("CAE_LLM_API_KEY", "")
LLM_MODEL = os.getenv("CAE_LLM_MODEL", "")

# -------------------- Embedding --------------------
EMBED_PROVIDER = os.getenv("CAE_EMBED_PROVIDER", "local").lower()  # local | api
EMBED_MODEL_PATH = _path("CAE_EMBED_MODEL_PATH", MODEL_DIR / "embedding" / "YOUR_EMBEDDING_MODEL")
EMBED_DEVICE = os.getenv("CAE_EMBED_DEVICE", "auto")
EMBED_TRUST_REMOTE_CODE = _bool("CAE_EMBED_TRUST_REMOTE_CODE", False)
EMBED_BATCH = int(os.getenv("CAE_EMBED_BATCH", "32"))
EMBED_MAX_LENGTH = int(os.getenv("CAE_EMBED_MAX_LENGTH", "1024"))
EMBED_NORMALIZE = _bool("CAE_EMBED_NORMALIZE", True)
EMBED_CACHE_ENABLED = _bool("CAE_EMBED_CACHE_ENABLED", True)
EMBED_CACHE_DB = STORAGE_DIR / "embedding_cache.sqlite3"
EMBED_BASE_URL = os.getenv("CAE_EMBED_BASE_URL", "http://127.0.0.1:8001/v1")
EMBED_API_KEY = os.getenv("CAE_EMBED_API_KEY", "")
EMBED_MODEL = os.getenv("CAE_EMBED_MODEL", "")

# -------------------- Chunk / ingest --------------------
CHUNK_SIZE = int(os.getenv("CAE_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CAE_CHUNK_OVERLAP", "120"))
INGEST_BATCH_SIZE = int(os.getenv("CAE_INGEST_BATCH_SIZE", "64"))
INGEST_SKIP_UNCHANGED = _bool("CAE_INGEST_SKIP_UNCHANGED", True)
INGEST_MANIFEST = STORAGE_DIR / "ingest_manifest.json"
SUPPORTED_EXT = {".docx", ".pdf", ".pptx", ".txt", ".md"}

# -------------------- Retrieval --------------------
COLLECTION_NAME = os.getenv("CAE_COLLECTION_NAME", "cae_docs")
RETRIEVAL_MODE = os.getenv("CAE_RETRIEVAL_MODE", "dense").lower()  # dense | hybrid
TOP_K = int(os.getenv("CAE_TOP_K", "6"))
CANDIDATE_K = int(os.getenv("CAE_CANDIDATE_K", "20"))
SIM_THRESHOLD = float(os.getenv("CAE_SIM_THRESHOLD", "0.25"))
RRF_K = int(os.getenv("CAE_RRF_K", "60"))
HYBRID_DENSE_WEIGHT = float(os.getenv("CAE_HYBRID_DENSE_WEIGHT", "0.65"))
HYBRID_LEXICAL_WEIGHT = float(os.getenv("CAE_HYBRID_LEXICAL_WEIGHT", "0.35"))

# -------------------- Optional reranker --------------------
RERANK_ENABLED = _bool("CAE_RERANK_ENABLED", False)
RERANK_MODEL_PATH = _path("CAE_RERANK_MODEL_PATH", MODEL_DIR / "reranker" / "YOUR_RERANKER_MODEL")
RERANK_DEVICE = os.getenv("CAE_RERANK_DEVICE", "auto")
RERANK_BATCH = int(os.getenv("CAE_RERANK_BATCH", "8"))
RERANK_MAX_LENGTH = int(os.getenv("CAE_RERANK_MAX_LENGTH", "512"))

for _d in (DATA_DIR, STORAGE_DIR, CHROMA_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 离线设备上避免 transformers/huggingface_hub 尝试联网。
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "huggingface" / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
