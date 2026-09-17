"""命令行问答 CLI。"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="向 CAE AI Platform 提问")
    parser.add_argument("question", nargs="*", help="问题")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-gen", action="store_true", help="只检索")
    parser.add_argument("--stop-server", action="store_true", help="停止自动启动的后台服务，释放模型内存/显存")
    args = parser.parse_args()
    if args.stop_server:
        if args.question or args.no_gen or args.top_k is not None:
            parser.error("--stop-server 不能与问题或检索参数一起使用")
        from core.local_service import stop
        try:
            print(stop())
            return 0
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"停止失败: {exc}", file=sys.stderr)
            return 1
    if not args.question or not " ".join(args.question).strip():
        parser.error("请输入问题")
    question = " ".join(args.question)
    if args.no_gen:
        from knowledge.rag import get_pipeline
        rag = get_pipeline()
        hits = rag.search(question, args.top_k)
        print(f"检索到 {len(hits)} 条:\n")
        for hit in hits:
            print(f"[{hit.score}] {hit.source} (第{hit.page}页) [{hit.kind}]")
            print(hit.text[:300]); print("-" * 60)
        return 0
    from core.local_service import ask
    try:
        result = ask(question, args.top_k)
    except (RuntimeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("=" * 60); print(result["answer"]); print("=" * 60)
    for source in result["sources"]:
        loc = source["source"] + (f" 第{source['page']}页" if source.get("page") else "")
        print(f"  [{source['index']}] {loc} (相似度 {source['score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
