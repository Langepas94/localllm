"""Собрать RAG-индекс localllm из документов в corpus/ (тем же rag.py, что и CLI).

Используется на деплое и локально. Индекс пишется в data/rag_index.json.

  python scripts/build_index.py --url http://127.0.0.1:11434/v1 --embed-model bge-m3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from rag import INDEX_PATH, RagIndex


def main():
    ap = argparse.ArgumentParser(description="Сборка RAG-индекса из corpus/")
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parent.parent / "corpus"))
    ap.add_argument("--fresh", action="store_true", help="стереть существующий индекс перед сборкой")
    args = ap.parse_args()

    if args.fresh and INDEX_PATH.exists():
        INDEX_PATH.unlink()

    client = OpenAI(base_url=args.url, api_key="lm-studio")
    index = RagIndex(client, args.embed_model)
    files, chunks = index.add_path(args.corpus)
    print(f"Проиндексировано файлов: {files}, чанков добавлено: {chunks}, всего в индексе: {len(index.entries)}")
    print(f"Индекс: {INDEX_PATH}")


if __name__ == "__main__":
    main()
