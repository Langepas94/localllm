"""Веб-сервис localllm: та же RAG-логика юр-режима (/legal), но по HTTP — для VPS.

Отдельная точка входа от CLI (chat.py), но пайплайн переиспользует rag.py и юр-режим
из chat.py (LEGAL_SYSTEM, LEGAL_RETR) — локально и на сервере одна программа, один git.

Работает против любого OpenAI-совместимого API: LM Studio (`/v1`) или Ollama
(`http://127.0.0.1:11434/v1`). На чистом stdlib (http.server) — без веб-фреймворков,
как и весь проект, намеренно deps-light.

Эндпоинты:
  GET  /            — страница чата (web/index.html)
  GET  /api/health  — статус
  POST /api/chat    — {"message": "..."} -> {answer, sources[], not_known}

Запуск:
  python web.py --url http://127.0.0.1:11434/v1 --model qwen2.5:1.5b --embed-model bge-m3
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import OpenAI, APIConnectionError, APIError

from bench import strip_think, preset_kwargs
from chat import LEGAL_SYSTEM, LEGAL_RETR
from rag import (
    ANSWER_MIN_SCORE, RERANK_CANDIDATES, RagIndex,
    build_category_map, relevance_filter, rerank, section_of, special_case_reorder,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_QUESTION_CHARS = 1000

# Одна тяжёлая генерация за раз: на слабом VPS (2 ядра, обе модели в swap)
# параллельный инференс лишь усиливает трешинг. Остальные запросы ждут -> стабильность.
_inference_lock = threading.Lock()


class Agent:
    """Юр-агент: RAG-поиск по ТК РФ + генерация. Повторяет ветку /legal on из chat.py."""

    def __init__(self, client: OpenAI, model: str, index: RagIndex):
        self.client = client
        self.model = model
        self.index = index
        # Карта «статья -> спец-категории» для special_case_reorder (строится один раз).
        self.catmap = build_category_map(index.entries)

    def answer(self, question: str) -> dict:
        # Пайплайн как в chat.py (legal): поиск+дедуп -> фильтр -> rerank -> порог «не знаю».
        # rewrite пропущен: он для follow-up'ов по истории, а веб-запрос одноходовый.
        hits = self.index.search(question, top_k=RERANK_CANDIDATES, dedup=LEGAL_RETR["dedup"])
        best_score = hits[0]["score"] if hits else 0.0
        if LEGAL_RETR["filter"]:
            hits = relevance_filter(hits)
        if LEGAL_RETR["rerank"]:
            hits = rerank(self.client, self.model, question, hits, top_k=LEGAL_RETR["top_k"])
        else:
            hits = hits[: LEGAL_RETR["top_k"]]
        # Детерминированно ставим общую норму выше специальных (или норму нужной
        # категории — если вопрос про неё). Слабая модель отвечает по первому фрагменту.
        hits = special_case_reorder(question, hits, self.catmap)

        # grounding: если ничего релевантного — детерминированное «Не знаю», без вызова модели
        if not hits or best_score < ANSWER_MIN_SCORE:
            return {
                "answer": "Не знаю — в Трудовом кодексе не нашлось достаточно релевантной "
                          "информации по вашему вопросу. Уточните формулировку.",
                "sources": [],
                "not_known": True,
            }

        chunks = "\n---\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
        context = (
            "Отвечай, опираясь ТОЛЬКО на пронумерованные фрагменты ниже. Если в них "
            "нет ответа — напиши «Не знаю» и попроси уточнить, ничего не выдумывай.\n\n"
            + chunks
        )
        messages = [
            {"role": "system", "content": LEGAL_SYSTEM},
            {"role": "user", "content": f"{context}\n\nВопрос: {question}"},
        ]
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, **preset_kwargs(self.model, "legal")
        )
        answer = strip_think(resp.choices[0].message.content or "").strip()
        sources = [
            {
                "source": Path(h["source"]).name,
                "section": section_of(h["text"]),
                "quote": " ".join(h["text"].split())[:220],
            }
            for h in hits
        ]
        return {"answer": answer or "[пустой ответ]", "sources": sources, "not_known": False}


def make_handler(agent: Agent, embed_model: str):
    index_html = (WEB_DIR / "index.html").read_bytes() if (WEB_DIR / "index.html").is_file() else b"web/index.html not found"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # тише в журнале systemd
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, index_html, "text/html; charset=utf-8")
            elif self.path == "/api/health":
                self._json(200, {"status": "ok", "model": agent.model, "embed_model": embed_model})
            else:
                self._json(404, {"detail": "not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                self._json(404, {"detail": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                question = (payload.get("message") or "").strip()
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"detail": "Некорректный JSON."})
                return
            if not question:
                self._json(400, {"detail": "Пустой вопрос."})
                return
            if len(question) > MAX_QUESTION_CHARS:
                self._json(413, {"detail": f"Вопрос длиннее {MAX_QUESTION_CHARS} символов."})
                return
            try:
                with _inference_lock:
                    result = agent.answer(question)
            except (APIConnectionError, APIError) as e:
                self._json(502, {"detail": f"Модель недоступна: {e}"})
                return
            self._json(200, result)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Веб-сервис агента по трудовому праву (localllm)")
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1", help="OpenAI-совместимый API (Ollama/LM Studio)")
    ap.add_argument("--model", default="qwen2.5:1.5b", help="чат-модель")
    ap.add_argument("--embed-model", default="bge-m3", help="модель эмбеддингов")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    client = OpenAI(base_url=args.url, api_key="lm-studio")
    index = RagIndex(client, args.embed_model)
    if not index.entries:
        raise SystemExit("Индекс пуст. Соберите его: python scripts/build_index.py (нужен PDF в corpus/).")
    agent = Agent(client, args.model, index)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(agent, args.embed_model))
    print(f"Агент по трудовому праву: http://{args.host}:{args.port}  "
          f"(модель {args.model}, эмбеддинги {args.embed_model}, чанков {len(index.entries)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
