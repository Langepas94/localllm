"""Профиль пользователя: факты о нём и селективное обогащение системного промпта.

Как решается «не замусоривать промпт»: при добавлении факта чат-модель один раз
генерирует «триггеры» — примерные вопросы, при которых факт стоит учитывать
(для «Мне 12 лет» — «какие фильмы мне можно смотреть», «посоветуй игру»...).
При каждом сообщении вопрос сравнивается эмбеддингами с триггерами
(вопрос-к-вопросу — это надёжно, в отличие от вопрос-к-факту), и в системный
промпт попадают только факты с достаточно близким триггером. Дорогое рассуждение
о релевантности случается один раз при добавлении, а не при каждом сообщении.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag import cosine

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILE_PATH = DATA_DIR / "profile.json"

MIN_SCORE = 0.85               # порог похожести вопроса на триггер
QUERY_PREFIX = "search_query: "  # префикс nomic-моделей; чужим моделям не мешает
MAX_TRIGGERS = 12
MODES = ("smart", "always", "off")

TRIGGER_PROMPT = """\
Дан факт о пользователе: «{fact}»

Придумай 12 коротких разнообразных вопросов или просьб ассистенту, при ответе на \
которые этот факт стоит учитывать. Охвати разные темы, где факт меняет ответ: \
рекомендации контента (что посмотреть, почитать, во что поиграть), советы, \
планирование, покупки, ограничения и возможности. Пиши от первого лица пользователя. \
Выведи только список, по одному вопросу в строке, без нумерации и без пояснений."""


class Profile:
    """Факты о пользователе с триггерами и их эмбеддингами; хранится в data/profile.json."""

    def __init__(self, client, embed_model: str | None):
        self.client = client
        self.embed_model = embed_model
        self.mode = "smart"
        self.facts: list[dict] = []  # {"text": str, "triggers": [str], "vectors": [[float]]}
        if PROFILE_PATH.exists():
            data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            self.mode = data.get("mode", "smart")
            self.facts = data.get("facts", [])

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(
            json.dumps({"mode": self.mode, "facts": self.facts}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.embed_model:
            return None
        try:
            resp = self.client.embeddings.create(model=self.embed_model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            return None

    def _gen_triggers(self, text: str, chat_model: str) -> list[str]:
        try:
            resp = self.client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": TRIGGER_PROMPT.format(fact=text)}],
                temperature=0.7,
                max_tokens=8000,  # reasoning-модели тратят тысячи токенов на <think>
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            return []
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        triggers = []
        for line in raw.splitlines():
            line = re.sub(r"^[\s\-*•]*\d*[.)]?\s*", "", line.strip()).strip()
            if line:
                triggers.append(line)
        return triggers[:MAX_TRIGGERS]

    def _make_fact(self, text: str, chat_model: str) -> dict:
        triggers = self._gen_triggers(text, chat_model)
        # эмбеддинг самого факта тоже кладём — фолбэк, если триггеров нет
        vectors = self._embed([QUERY_PREFIX + t for t in [text] + triggers]) or []
        return {"text": text, "triggers": triggers, "vectors": vectors}

    def add(self, text: str, chat_model: str) -> list[str]:
        """Добавляет факт. Возвращает сгенерированные триггеры (пусто = не удалось)."""
        fact = self._make_fact(text, chat_model)
        self.facts.append(fact)
        self.save()
        return fact["triggers"]

    def edit(self, index: int, text: str, chat_model: str) -> list[str]:
        fact = self._make_fact(text, chat_model)
        self.facts[index] = fact
        self.save()
        return fact["triggers"]

    def delete(self, index: int) -> None:
        del self.facts[index]
        self.save()

    def add_trigger(self, index: int, trigger: str) -> bool:
        """Вручную дописывает триггер к факту (если авто-список что-то упустил)."""
        vec = self._embed([QUERY_PREFIX + trigger])
        if not vec:
            return False
        fact = self.facts[index]
        fact["triggers"].append(trigger)
        fact["vectors"].append(vec[0])
        self.save()
        return True

    def relevant(self, query: str) -> list[str]:
        """Факты, которые стоит подклеить в системный промпт для этого вопроса."""
        if self.mode == "off" or not self.facts:
            return []
        if self.mode == "always":
            return [f["text"] for f in self.facts]
        qv = self._embed([QUERY_PREFIX + query])
        if not qv:
            # эмбеддинги недоступны — лучше дать модели всё, чем потерять важное
            return [f["text"] for f in self.facts]
        out = []
        for f in self.facts:
            if not f["vectors"]:
                out.append(f["text"])  # факт без векторов включаем всегда
            elif max(cosine(qv[0], v) for v in f["vectors"]) >= MIN_SCORE:
                out.append(f["text"])
        return out
