"""RAG: индексация текстовых файлов и поиск через эмбеддинги LM Studio."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_PATH = DATA_DIR / "rag_index.json"

CHUNK_SIZE = 800        # символов в чанке
CHUNK_OVERLAP = 200     # перекрытие соседних чанков
EMBED_BATCH = 32        # чанков на один запрос к эмбеддингам
TOP_K = 3               # сколько чанков подставлять в контекст
MIN_SCORE = 0.45        # порог косинусной близости для попадания в выборку
ANSWER_MIN_SCORE = 0.5  # ниже лучшего score — ассистент отвечает «не знаю»
ARTICLE_MAX = 2000      # статья длиннее — дробим окном, сохраняя заголовок в каждом чанке
RERANK_CANDIDATES = 10  # сколько чанков достаём под reranker (потом сужаем до TOP_K)
REL_GAP = 0.12          # фильтр релевантности: отсекаем кандидатов настолько ниже лучшего

TEXT_EXTS = {".txt", ".md", ".rst", ".py", ".json", ".csv", ".html", ".log", ".yaml", ".yml"}
DOC_EXTS = TEXT_EXTS | {".pdf"}

# «Статья 123.» / «Статья 351.1.» в начале блока — граница для structured-нарезки.
ARTICLE_RE = re.compile(r"(Статья\s+\d+(?:\.\d+)?\.)")

# Колонтитулы экспорта КонсультантПлюс: на каждой странице такой блок вклинивается
# в середину предложения. Вырезаем построчно (только для PDF, чтобы не задеть код/текст).
PDF_NOISE_RE = re.compile(
    r"^(?:"
    r"КонсультантПлюс"
    r"|.*www\.consultant\.ru.*"                       # 'надежная правовая поддержка ... Страница N из M'
    r"|Документ предоставлен КонсультантПлюс"
    r"|Дата сохранения:.*"
    r"|Страница\s+\d+\s+из\s+\d+"
    r'|"[^"]*"\s+от\s+\d{2}\.\d{2}\.\d{4}\s+N'         # строка-заголовок: "Трудовой кодекс..." от ... N
    r"|\d+-ФЗ"                                          # перенос номера закона из заголовка (197-ФЗ)
    r"|\(ред\.\s+от\s+\d{2}\.\d{2}\.\d{4}\)"           # колонтитульная '(ред. от ДД.ММ.ГГГГ)'
    r")\s*$",
    re.MULTILINE,
)


def strip_boilerplate(text: str) -> str:
    """Убирает колонтитулы КонсультантПлюс и схлопывает пустые строки от них."""
    text = PDF_NOISE_RE.sub("", text)
    return re.sub(r"\n{2,}", "\n", text)


def extract_text(path: Path) -> str:
    """Текст файла. PDF читается через pypdf (+чистка колонтитулов), остальное — UTF-8."""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Для PDF нужен pypdf: pip install pypdf") from e
        reader = PdfReader(str(path))
        return strip_boilerplate("\n".join(page.extract_text() or "" for page in reader.pages))
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    step = max(1, size - overlap)  # защита от зацикливания при overlap >= size
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return [c.strip() for c in chunks if c.strip()]


def structured_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str] | None:
    """Дробит юртекст по статьям («Статья N.»); заголовок статьи входит в каждый чанк.

    Короткая статья = один чанк; длинная (> ARTICLE_MAX) до-режется скользящим окном,
    но заголовок статьи повторяется в каждом её куске — чтобы контекст не терял привязку.
    Возвращает None, если статей не нашлось (текст не структурный — нужен chunk_text).
    """
    parts = ARTICLE_RE.split(text)  # [преамбула, "Статья 1.", тело1, "Статья 2.", тело2, ...]
    if len(parts) < 3:
        return None
    chunks: list[str] = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if len(header) + len(body) <= ARTICLE_MAX:
            chunks.append(f"{header} {body}".strip())
        else:
            for sub in chunk_text(body, size, overlap):
                chunks.append(f"{header} {sub}")
    return [c for c in chunks if c.strip()]


def section_of(text: str) -> str:
    """Метка раздела для списка источников: «Статья N.» из чанка либо короткий префикс."""
    m = ARTICLE_RE.match(text.strip())
    if m:
        return m.group(1)
    head = " ".join(text.split())[:30]
    return head + "…" if head else "фрагмент"


def looks_binary(text: str) -> bool:
    """Похоже ли на сырой PDF/бинарь, ошибочно прочитанный как текст.

    Ловит частый провал: PDF попал в индекс байтами (`%PDF-...1 0 obj...`) —
    без извлечения через pypdf. Такой «текст» засоряет индекс и ломает ответы.
    """
    if text.lstrip()[:16].startswith("%PDF-"):
        return True
    sample = text[:2000]
    if not sample:
        return False
    weird = sum(1 for ch in sample if ch == "�" or (ord(ch) < 32 and ch not in "\r\n\t"))
    return weird / len(sample) > 0.05


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
            files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in DOC_EXTS]
        else:
            raise FileNotFoundError(path)
        added_files = added_chunks = 0
        for f in files:
            try:
                text = extract_text(f)
            except OSError:
                continue
            # не индексируем бинарь, прочитанный как текст (частый случай — сырой PDF)
            if looks_binary(text):
                if p.is_file():
                    raise RuntimeError(
                        f"Файл похож на сырой PDF/бинарь, не индексирую: {f}\n"
                        f"Для PDF нужен pypdf (pip install pypdf) и переиндексация: /rag clear, затем /rag add."
                    )
                continue  # в папке просто пропускаем битый файл
            # structured-нарезка по статьям для юртекста; иначе — скользящее окно
            chunks = structured_chunks(text) or chunk_text(text)
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
        """Топ-k чанков, похожих на запрос: [{source, text, score}], без вектора.

        Оценка близости (score) возвращается — на неё опираются фильтр
        релевантности и reranker. Пустой список, если ничего не нашлось.
        """
        if not self.entries:
            return []
        qv = self._embed([query])[0]
        scored = sorted(
            ((cosine(qv, e["vector"]), i) for i, e in enumerate(self.entries)),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            {"source": self.entries[i]["source"], "text": self.entries[i]["text"],
             "score": score, "chunk_id": i}
            for score, i in scored[:top_k]
            if score >= min_score
        ]

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


# ── улучшение ретривала: rewriter, фильтр релевантности, reranker ─────────────
# Работают поверх search(): rewrite делает запрос самодостаточным, фильтр дёшево
# отсекает слабые совпадения по косинусу, reranker чат-моделью переупорядочивает
# и выкидывает нерелевантное. Каждый шаг fail-safe: при сбое возвращает вход.

REWRITE_PROMPT = """\
Ты помогаешь искать в базе документов. Ниже недавняя переписка и новый вопрос.
Перепиши вопрос в один самодостаточный поисковый запрос на русском: раскрой \
местоимения и отсылки («это», «а по какому закону») по контексту переписки, \
добавь ключевые термины. Не отвечай на вопрос и не поясняй — выведи только \
переписанный запрос одной строкой.

Переписка:
{history}

Вопрос: {question}
Поисковый запрос:"""

RERANK_PROMPT = """\
Вопрос пользователя: {query}

Ниже пронумерованные фрагменты из базы знаний. Отбери те, что действительно \
помогают ответить на вопрос, и упорядочи по убыванию полезности (не более {k}). \
Выведи только их номера через запятую, например: 3, 1. \
Если ни один фрагмент не относится к вопросу — выведи одно слово: НЕТ.

Фрагменты:
{passages}

Номера релевантных фрагментов:"""


def _service_chat(client, model: str, prompt: str, temperature: float) -> str:
    """Служебный вызов чат-модели: не-стриминг, вырезание <think>. '' при ошибке.

    max_tokens большой: локальные reasoning-модели тратят тысячи токенов на
    размышления до короткого ответа (см. заметку в CLAUDE.md).
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=8000,
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def rewrite_query(client, chat_model: str, history: list[dict], question: str) -> str:
    """Переписывает вопрос в самодостаточный поисковый запрос с учётом истории.

    Follow-up вроде «а по какому закону?» превращает в полноценный запрос.
    При ошибке/пустом ответе возвращает исходный вопрос (fail-safe).
    """
    if not history:
        return question
    convo = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content'][:400]}"
        for m in history[-4:]
    )
    out = _service_chat(client, chat_model, REWRITE_PROMPT.format(history=convo, question=question), 0.2)
    # берём первую непустую строку — модель иногда добавляет лишнее
    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    return line or question


def relevance_filter(hits: list[dict], gap: float = REL_GAP) -> list[dict]:
    """Дёшево (без LLM) отсекает кандидатов, заметно уступающих лучшему по косинусу.

    hits ожидаются отсортированными по убыванию score (как их отдаёт search).
    Убирает «притянутые» чанки, оставляя кучный топ вокруг лучшего совпадения.
    """
    if not hits:
        return []
    top = hits[0]["score"]
    return [h for h in hits if h["score"] >= top - gap]


def _parse_ranking(text: str, n: int) -> list[int] | None:
    """Разбирает ответ reranker'а: [] — явно ничего, None — не распарсили (фолбэк)."""
    low = text.lower()
    if not re.search(r"\d", text) and re.search(r"\b(нет|none|ничего|не подход)", low):
        return []
    seen: list[int] = []
    for x in (int(m) for m in re.findall(r"\d+", text)):
        if 1 <= x <= n and x not in seen:
            seen.append(x)
    return seen or None


def rerank(client, chat_model: str, query: str, hits: list[dict], top_k: int = TOP_K) -> list[dict]:
    """LLM-переранжирование + фильтр релевантности за один вызов.

    Модель выбирает из кандидатов действительно релевантные, в порядке убывания
    пользы, и отбрасывает мусор. Возвращает <= top_k чанков; пустой список — если
    релевантного нет (лучше не подставлять ничего). При сбое разбора — исходный
    порядок (top_k), чтобы не потерять ретривал полностью.
    """
    if not hits:
        return []
    listing = "\n".join(f"[{i}] {h['text'][:500]}" for i, h in enumerate(hits, 1))
    out = _service_chat(client, chat_model, RERANK_PROMPT.format(query=query, passages=listing, k=top_k), 0.0)
    order = _parse_ranking(out, len(hits))
    if order is None:
        return hits[:top_k]
    return [hits[i - 1] for i in order][:top_k]
