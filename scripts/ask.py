"""命令行问答 CLI。"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from knowledge.rag import get_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="向 CAE AI Platform 提问")
    parser.add_argument("question", nargs="+", help="问题")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-gen", action="store_true", help="只检索")
    args = parser.parse_args()
    question = " ".join(args.question)
    rag = get_pipeline()
    if args.no_gen:
        hits = rag.search(question, args.top_k)
        print(f"检索到 {len(hits)} 条:\n")
        for hit in hits:
            print(f"[{hit.score}] {hit.source} (第{hit.page}页) [{hit.kind}]")
            print(hit.text[:300]); print("-" * 60)
        return 0
    result = rag.ask(question, args.top_k)
    print("=" * 60); print(result["answer"]); print("=" * 60)
    for source in result["sources"]:
        loc = source["source"] + (f" 第{source['page']}页" if source.get("page") else "")
        print(f"  [{source['index']}] {loc} (相似度 {source['score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
