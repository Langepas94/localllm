"""RAG: индексация текстовых файлов и поиск через эмбеддинги LM Studio."""

from __future__ import annotations

import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_PATH = DATA_DIR / "rag_index.json"

CHUNK_SIZE = 800      # символов в чанке
CHUNK_OVERLAP = 200   # перекрытие соседних чанков
EMBED_BATCH = 32      # чанков на один запрос к эмбеддингам
TOP_K = 3             # сколько чанков подставлять в контекст
MIN_SCORE = 0.45      # порог косинусной близости

TEXT_EXTS = {".txt", ".md", ".rst", ".py", ".json", ".csv", ".html", ".log", ".yaml", ".yml"}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class RagIndex:
    """Векторный индекс в одном JSON-файле; хранит чанки и их эмбеддинги."""

    def __init__(self, client, embed_model: str):
        self.client = client
        self.embed_model = embed_model
        self.entries: list[dict] = []  # {"source": str, "text": str, "vector": list[float]}
        if INDEX_PATH.exists():
            self.entries = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]

    def add_path(self, path: str) -> tuple[int, int]:
        """Индексирует файл или папку (рекурсивно). Возвращает (файлов, чанков)."""
        p = Path(path).expanduser()
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in TEXT_EXTS]
        else:
            raise FileNotFoundError(path)
        added_files = added_chunks = 0
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue
            # переиндексация файла: убираем его старые чанки
            self.entries = [e for e in self.entries if e["source"] != str(f)]
            for i in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[i : i + EMBED_BATCH]
                for chunk, vector in zip(batch, self._embed(batch)):
                    self.entries.append({"source": str(f), "text": chunk, "vector": vector})
                added_chunks += len(batch)
            added_files += 1
        self.save()
        return added_files, added_chunks

    def search(self, query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> list[dict]:
        """Топ-k чанков, похожих на запрос. Пустой список, если ничего не нашлось."""
        if not self.entries:
            return []
        qv = self._embed([query])[0]
        scored = sorted(
            ((cosine(qv, e["vector"]), e) for e in self.entries),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [e for score, e in scored[:top_k] if score >= min_score]

    def sources(self) -> dict[str, int]:
        """{путь файла: число чанков} для /rag status."""
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e["source"]] = counts.get(e["source"], 0) + 1
        return counts

    def clear(self) -> None:
        self.entries = []
        self.save()

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(json.dumps(self.entries, ensure_ascii=False), encoding="utf-8")
