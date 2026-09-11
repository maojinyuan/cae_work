"""不启动 Web 服务的最小运行测试。"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from knowledge.embedding import get_embedder
from knowledge.vectorstore import get_store
from models.llm import get_llm


def main():
    embedder = get_embedder()
    vec = embedder.embed_one("发动机缸盖热应力分析")
    print(f"Embedding: OK, dim={len(vec)}")
    print(f"Vector store: OK, stats={get_store().stats()}")
    llm = get_llm()
    text = llm.generate([{"role": "user", "content": "只回答：CAE模型加载成功"}], max_tokens=32)
    print(f"LLM: OK, output={text!r}")


if __name__ == "__main__":
    main()
