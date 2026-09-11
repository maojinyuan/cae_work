"""文档导入 CLI。"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from core import config
from knowledge.rag import get_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="导入 CAE 文档到知识库")
    parser.add_argument("path", nargs="?", default=str(config.DATA_DIR))
    parser.add_argument("--recreate", action="store_true", help="清空向量库后重建")
    parser.add_argument("--force", action="store_true", help="即使文件未变化也重新编码")
    args = parser.parse_args()
    result = get_pipeline().ingest_path(args.path, recreate=args.recreate, force=args.force)
    print(f"导入完成: 新增/更新 {result['files_imported']} 个文件，共 {result['total_chunks']} 个块")
    for item in result["imported"]:
        print(f"  ✓ {item['file']} ({item['chunks']} 块)")
    for item in result["skipped"]:
        print(f"  - {item['file']} ({item['status']})")
    for item in result["errors"]:
        print(f"  ✗ {item['file']} ({item['error']})")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
