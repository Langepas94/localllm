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
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import OpenAI, APIConnectionError, APIError

from bench import strip_think, preset_kwargs
from chat import LEGAL_SYSTEM, LEGAL_RETR
from rag import (
    ANSWER_MIN_SCORE, RERANK_CANDIDATES, RagIndex,
    build_article_full, build_article_lead, relevance_filter, section_of,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_QUESTION_CHARS = 1000
CONTEXT_ARTICLES = 2      # статей (совпавших чанков) в контексте. 2, т.к. чанки крупнее
                          # (см. CONTEXT_CHARS) — держим скорость. Нужная норма обычно в #1.
CONTEXT_CHARS = 1400      # символов на чанк: норма бывает в середине статьи (ст.81 прогул
                          # на позиции ~1150) — короче нельзя, иначе обрежется.

# Одна тяжёлая генерация за раз: на слабом VPS (2 ядра, обе модели в swap)
# параллельный инференс лишь усиливает трешинг. Остальные запросы ждут -> стабильность.
_inference_lock = threading.Lock()


class Agent:
    """Юр-агент: RAG-поиск по ТК РФ + генерация. Повторяет ветку /legal on из chat.py."""

    def __init__(self, client: OpenAI, model: str, index: RagIndex):
        self.client = client
        self.model = model
        self.index = index
        # Строятся один раз при старте: лид-чанки и полный текст статей (для источников).
        self.lead = build_article_lead(index.entries)
        self.full = build_article_full(index.entries)

    def answer(self, question: str) -> dict:
        # Пайплайн как в chat.py (legal): поиск+дедуп -> фильтр -> rerank -> порог «не знаю».
        # rewrite пропущен: он для follow-up'ов по истории, а веб-запрос одноходовый.
        hits = self.index.search(question, top_k=RERANK_CANDIDATES, dedup=LEGAL_RETR["dedup"])
        best_score = hits[0]["score"] if hits else 0.0
        if LEGAL_RETR["filter"]:
            hits = relevance_filter(hits)
        # Порядок — чистый косинус (+dedup по статье). Он и так ставит нужную общую норму
        # первой (проверено: увольнение->ст.80, испыт.срок->ст.70, отпуск за свой счёт->ст.128).
        # LLM-rerank убран (медленно/ненадёжно); special_case_reorder убран — он ошибочно
        # топил общие статьи, чей текст перечисляет категории работников (ст.70 не попадала
        # в контекст, отсюда ложные «недостаточно данных»).
        hits = hits[: LEGAL_RETR["top_k"]]

        # grounding: если ничего релевантного — детерминированное «Не знаю», без вызова модели
        if not hits or best_score < ANSWER_MIN_SCORE:
            return {
                "answer": "Не знаю — в Трудовом кодексе не нашлось достаточно релевантной "
                          "информации по вашему вопросу. Уточните формулировку.",
                "sources": [],
                "not_known": True,
            }

        # В контекст берём именно ТОТ чанк, что совпал с запросом по косинусу — в нём и
        # лежит нужная норма. Лид-чанк (начало статьи) НЕ годится: ключевая норма часто
        # в середине/конце (ст.70 срок — на позиции ~2300, ст.81 прогул — ~1150), и модель
        # видела начало без ответа -> ложное «недостаточно данных».
        ctx_hits = hits[:CONTEXT_ARTICLES]
        chunks = "\n---\n".join(
            f"[{i}] " + " ".join(h["text"].split())[:CONTEXT_CHARS]
            for i, h in enumerate(ctx_hits, 1)
        )
        context = (
            "Ответь работнику простым языком, ТОЛЬКО по фрагментам ниже (фрагмент [1] — "
            "главный). 2–3 предложения: суть нормы + что это значит на практике; сошлись "
            "на «ст. N ТК РФ»; нерелевантные фрагменты игнорируй. Если нормы по теме нет — "
            "ответь одной фразой «Недостаточно данных, уточните вопрос».\n\n"
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
        # Иногда модель даёт содержательный ответ И приписывает «Недостаточно данных» —
        # убираем спорную приписку, если ответ не сводится только к ней (детерминированно).
        if "недостаточно данных" in answer.lower() and len(answer) > 70:
            answer = re.sub(r"[^.!?]*недостаточно данных[^.!?]*[.!?]?", "", answer, flags=re.I).strip()
        sources = [
            {
                "source": Path(h["source"]).name,
                "section": section_of(h["text"]),
                "quote": " ".join(self.lead.get(section_of(h["text"]), h["text"]).split())[:220],
                "full": self.full.get(section_of(h["text"]), " ".join(h["text"].split())),
            }
            for h in ctx_hits
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
            # Слабый box тянет одну генерацию за раз (~100с). Не держим ждущие соединения
            # открытыми (они отваливаются под нагрузкой) — если сервис занят, сразу отвечаем
            # 503 «занято». Так сервис остаётся стабильным при нескольких запросах.
            if not _inference_lock.acquire(blocking=False):
                self._json(503, {"detail": "Сервис сейчас отвечает на другой вопрос "
                                           "(на слабом сервере — по одному за раз). "
                                           "Попробуйте через минуту."})
                return
            try:
                result = agent.answer(question)
            except (APIConnectionError, APIError) as e:
                self._json(502, {"detail": f"Модель временно недоступна, попробуйте ещё раз. ({type(e).__name__})"})
                return
            except Exception as e:  # любая иная ошибка -> graceful JSON, а не обрыв соединения
                traceback.print_exc()
                self._json(500, {"detail": f"Внутренняя ошибка сервиса ({type(e).__name__}). Попробуйте переформулировать вопрос."})
                return
            finally:
                _inference_lock.release()
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

    # Прогрев в ФОНЕ: грузим обе модели (эмбеддер + чат) в память, чтобы запросы шли без
    # холодной загрузки (на слабом VPS это минуты). В отдельном потоке — чтобы порт
    # открылся сразу (запрос во время прогрева будет медленным, но не с ошибкой связи).
    # При MAX_LOADED_MODELS=2 обе модели останутся резидентно -> запросы без перезагрузок.
    def _warmup():
        try:
            index._embed(["прогрев"])
            client.chat.completions.create(model=args.model,
                                           messages=[{"role": "user", "content": "ок"}], max_tokens=1)
            print("Модели прогреты (эмбеддер + чат загружены).")
        except Exception as e:  # прогрев не критичен
            print(f"Прогрев не удался (не критично): {e}")
    threading.Thread(target=_warmup, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(agent, args.embed_model))
    print(f"Агент по трудовому праву: http://{args.host}:{args.port}  "
          f"(модель {args.model}, эмбеддинги {args.embed_model}, чанков {len(index.entries)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
