"""命令行问答 CLI。"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import json
from urllib.error import URLError
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description="向 CAE AI Platform 提问")
    parser.add_argument("question", nargs="+", help="问题")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-gen", action="store_true", help="只检索")
    parser.add_argument("--server", help="常驻 CAE 服务地址，如 http://127.0.0.1:8000；客户端不加载模型")
    args = parser.parse_args()
    question = " ".join(args.question)
    if args.server:
        endpoint = "/api/search" if args.no_gen else "/api/query"
        request = Request(
            args.server.rstrip("/") + endpoint,
            data=json.dumps({"question": question, "top_k": args.top_k}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=600) as response:
                result = json.load(response)
        except (URLError, OSError, ValueError) as exc:
            print(f"服务请求失败: {exc}。请确认 CAE API 服务已启动且地址正确。", file=sys.stderr)
            return 1
    else:
        from knowledge.rag import get_pipeline
        rag = get_pipeline()
        result = ({"results": [vars(hit) for hit in rag.search(question, args.top_k)]}
                  if args.no_gen else rag.ask(question, args.top_k))
    if args.no_gen:
        hits = result["results"]
        print(f"检索到 {len(hits)} 条:\n")
        for hit in hits:
            print(f"[{hit['score']}] {hit['source']} (第{hit['page']}页) [{hit['kind']}]")
            print(hit['text'][:300]); print("-" * 60)
        return 0
    print("=" * 60); print(result["answer"]); print("=" * 60)
    for source in result["sources"]:
        loc = source["source"] + (f" 第{source['page']}页" if source.get("page") else "")
        print(f"  [{source['index']}] {loc} (相似度 {source['score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
