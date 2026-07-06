"""Сессии: персистентная история, системный промпт и сжатие контекста."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

KEEP_MESSAGES = 10  # сколько последних сообщений оставлять при сжатии

SUMMARY_PROMPT = """\
Ниже — часть диалога пользователя с ассистентом. Сожми её в краткое саммари \
(5-10 предложений): о чём шла речь, какие факты, решения и договорённости важно \
помнить для продолжения разговора. Пиши по-русски, без вступлений, только суть.

{previous}{transcript}"""


class Session:
    """Один диалог: системный промпт, история сообщений, саммари старой части."""

    def __init__(self, system_prompt: str = "", model: str = ""):
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.created = time.strftime("%Y-%m-%d %H:%M:%S")
        self.system_prompt = system_prompt
        self.model = model
        self.summary = ""
        self.messages: list[dict] = []  # только user/assistant

    def build_messages(self, rag_context: str = "", profile_block: str = "") -> list[dict]:
        """Собирает полный список messages для запроса к API.

        Системный промпт, факты профиля и саммари не хранятся в self.messages,
        а подклеиваются здесь; RAG-контекст добавляется только в отправляемую
        копию последнего вопроса, в истории вопрос остаётся чистым.
        """
        system = self.system_prompt
        if profile_block:
            system = (system + "\n\n" if system else "") + (
                "Факты о пользователе, относящиеся к его сообщению:\n" + profile_block
            )
        if self.summary:
            system = (system + "\n\n" if system else "") + (
                "Краткое содержание предыдущей части диалога:\n" + self.summary
            )
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(dict(m) for m in self.messages)
        if rag_context and msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] = (
                f"Контекст из базы знаний:\n{rag_context}\n\nВопрос: {msgs[-1]['content']}"
            )
        return msgs

    def estimate_tokens(self, rag_context: str = "") -> int:
        """Грубая оценка размера запроса в токенах (~3 символа на токен для русского)."""
        chars = sum(len(m["content"]) for m in self.build_messages(rag_context))
        return chars // 3

    @property
    def path(self) -> Path:
        return SESSIONS_DIR / f"{self.id}.json"

    def save(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_file(cls, path: Path) -> "Session":
        session = cls()
        session.__dict__.update(json.loads(Path(path).read_text(encoding="utf-8")))
        return session


def list_sessions() -> list[Path]:
    """Файлы сессий, новые первыми."""
    if not SESSIONS_DIR.exists():
        return []
    return sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def session_preview(path: Path) -> str:
    """Строка для списка сессий: id, модель, число сообщений, начало первого вопроса."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"{path.stem} (повреждена)"
    first = next((m["content"] for m in data.get("messages", []) if m["role"] == "user"), "")
    snippet = first[:60].replace("\n", " ") + ("…" if len(first) > 60 else "")
    return f"{data.get('id', path.stem)} | {data.get('model', '?')} | {len(data.get('messages', []))} сообщ. | {snippet}"


def compress(session: Session, client, keep: int = KEEP_MESSAGES) -> bool:
    """Сжимает историю: всё, кроме последних keep сообщений, превращает в саммари.

    Саммари делает та же модель, что ведёт диалог. Возвращает True при успехе.
    """
    if len(session.messages) <= keep:
        return False
    old, session.messages = session.messages[:-keep], session.messages[-keep:]
    transcript = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}" for m in old
    )
    previous = (
        f"Саммари ещё более ранней части диалога (учти его):\n{session.summary}\n\n"
        if session.summary
        else ""
    )
    try:
        resp = client.chat.completions.create(
            model=session.model,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(previous=previous, transcript=transcript)}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        session.messages = old + session.messages  # откат
        return False
    # у reasoning-моделей вырезаем блок размышлений
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if not text:
        session.messages = old + session.messages
        return False
    session.summary = text
    return True
