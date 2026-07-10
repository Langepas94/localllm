"""Бенчмарк «до/после» оптимизации локальной LLM под юридическую задачу.

Гоняет один и тот же набор вопросов в двух режимах — базовом («до») и
оптимизированном («после», пресет legal + юр-промпт + RAG-контекст) — замеряет
время ответа, токены и скорость (ток/с) и сохраняет сами ответы, чтобы можно было
сравнить качество. Результат — таблица метрик в консоли + markdown-отчёт и JSON.

Нужен запущенный сервер LM Studio с загруженной моделью (Developer -> Start Server).

Примеры:
    python bench.py --model <id>
    python bench.py --model <id> --rag                       # + контекст из data/rag_index.json
    python bench.py --model <id> --isolate                   # варьировать ТОЛЬКО параметры
    python bench.py --model <id> --questions q.txt --out docs/bench-legal.md --repeat 2

Квантование сравнивают так: загрузить в LM Studio квант A, прогнать bench, затем
квант B — и сравнить два отчёта (скорость/качество на одном наборе вопросов).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from openai import OpenAI, APIConnectionError, APIError

from chat import LEGAL_RETR, LEGAL_SYSTEM, find_embedding_model  # единые настройки юр-режима
from gen_params import PRESETS, GenParams
from rag import INDEX_PATH, RagIndex

DEFAULT_URL = "http://localhost:1234/v1"

# Набор вопросов по умолчанию (общая гражданско-правовая тематика). Свой список —
# через --questions <файл> (по одному вопросу в строке, # — комментарий).
DEFAULT_QUESTIONS = [
    "Что такое оферта и чем она отличается от приглашения делать оферты?",
    "В каких случаях сделка признаётся ничтожной?",
    "Каков общий срок исковой давности и с какого момента он исчисляется?",
    "Какие условия договора купли-продажи считаются существенными?",
    "Что такое неустойка и как она соотносится с возмещением убытков?",
]


def strip_think(text: str) -> str:
    """Вырезает блок размышлений reasoning-моделей."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    qs = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not qs:
        raise ValueError(f"В файле {path} не найдено вопросов.")
    return qs


def preset_kwargs(model: str, name: str | None) -> dict:
    """kwargs генерации для пресета (name=None -> пусто, дефолты модели)."""
    gp = GenParams()
    gp.by_model = {model: dict(PRESETS[name])} if name else {}
    return gp.request_kwargs(model)


def build_messages(question: str, system: str | None, context: str) -> list[dict]:
    """Собирает messages как в chat.py: система + (контекст ->) вопрос."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    content = question
    if context:
        content = (
            "Отвечай, опираясь ТОЛЬКО на фрагменты ниже; если в них нет ответа — «Не знаю».\n\n"
            f"{context}\n\nВопрос: {question}"
        )
    msgs.append({"role": "user", "content": content})
    return msgs


def retrieve(rag_index: RagIndex | None, question: str) -> str:
    """Контекст юр-режима: top_k + дедуп по статье (как /legal on), без LLM-шагов."""
    if not rag_index or not rag_index.entries:
        return ""
    hits = rag_index.search(question, top_k=LEGAL_RETR["top_k"], dedup=LEGAL_RETR["dedup"])
    return "\n---\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))


def run_once(client: OpenAI, model: str, messages: list[dict], kwargs: dict) -> dict:
    """Один не-стриминг вызов. Возвращает метрики + ответ (или ошибку)."""
    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    except (APIConnectionError, APIError) as e:
        return {"error": str(e), "elapsed": time.perf_counter() - started}
    elapsed = time.perf_counter() - started
    usage = resp.usage
    answer = strip_think(resp.choices[0].message.content or "")
    comp = usage.completion_tokens if usage else 0
    return {
        "elapsed": elapsed,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": comp,
        "total_tokens": usage.total_tokens if usage else 0,
        "tok_s": comp / elapsed if elapsed > 0 else 0.0,
        "answer": answer or "[пустой ответ — весь бюджет max_tokens ушёл на размышления?]",
    }


def run_mode(client, model, question, *, system, context, kwargs, repeat) -> dict:
    """Прогоняет вопрос repeat раз в одном режиме. Агрегирует метрики, ловит недетерминизм."""
    messages = build_messages(question, system, context)
    runs = [run_once(client, model, messages, kwargs) for _ in range(repeat)]
    if any("error" in r for r in runs):
        return {"error": next(r["error"] for r in runs if "error" in r)}
    agg = {
        "elapsed": statistics.mean(r["elapsed"] for r in runs),
        "tok_s": statistics.mean(r["tok_s"] for r in runs),
        "prompt_tokens": runs[0]["prompt_tokens"],
        "completion_tokens": statistics.mean(r["completion_tokens"] for r in runs),
        "answer": runs[0]["answer"],
        "identical": len({r["answer"] for r in runs}) == 1 if repeat > 1 else None,
    }
    return agg


def _avg(rows: list[dict], mode: str, field: str) -> float:
    vals = [r[mode][field] for r in rows if "error" not in r[mode]]
    return statistics.mean(vals) if vals else 0.0


def render_report(meta: dict, rows: list[dict]) -> str:
    """Собирает markdown-отчёт из метрик и ответов (чистая функция — тестируема offline)."""
    L: list[str] = []
    L.append(f"# Бенчмарк оптимизации: {meta['model']}")
    L.append("")
    L.append(f"- Дата: {meta['date']}")
    L.append(f"- Режим: {'изоляция параметров (RAG/промпт одинаковы)' if meta['isolate'] else 'холистический (до = сырая модель, после = пресет+промпт+RAG)'}")
    L.append(f"- RAG-контекст: {'да' if meta['rag'] else 'нет'} · Повторов на вопрос: {meta['repeat']}")
    L.append(f"- Пресет «после»: {meta['preset']} = {meta['preset_kwargs']}")
    L.append("")
    L.append("## Сводка (среднее по всем вопросам)")
    L.append("")
    L.append("| Метрика | До (base) | После (optimized) |")
    L.append("|---|---|---|")
    L.append(f"| Время ответа, с | {_avg(rows,'base','elapsed'):.1f} | {_avg(rows,'opt','elapsed'):.1f} |")
    L.append(f"| Скорость, ток/с | {_avg(rows,'base','tok_s'):.0f} | {_avg(rows,'opt','tok_s'):.0f} |")
    L.append(f"| Токенов в ответе | {_avg(rows,'base','completion_tokens'):.0f} | {_avg(rows,'opt','completion_tokens'):.0f} |")
    L.append(f"| Токенов в промпте | {_avg(rows,'base','prompt_tokens'):.0f} | {_avg(rows,'opt','prompt_tokens'):.0f} |")
    if meta["repeat"] > 1:
        det = [r["opt"].get("identical") for r in rows if "error" not in r["opt"]]
        same = sum(1 for d in det if d)
        L.append(f"| Детерминизм «после» (одинаковый ответ на {meta['repeat']} прогонах) | — | {same}/{len(det)} |")
    L.append("")
    L.append("## Качество: ответы по вопросам")
    L.append("")
    for i, r in enumerate(rows, 1):
        L.append(f"### {i}. {r['question']}")
        for tag, key in (("До оптимизации", "base"), ("После оптимизации", "opt")):
            L.append("")
            L.append(f"**{tag}:**")
            if "error" in r[key]:
                L.append(f"> ошибка: {r[key]['error']}")
            else:
                m = r[key]
                L.append(f"> _{m['elapsed']:.1f} с · {m['tok_s']:.0f} ток/с · "
                         f"промпт {m['prompt_tokens']} + ответ {m['completion_tokens']:.0f} ток._")
                L.append("")
                for line in m["answer"].splitlines() or [""]:
                    L.append(f"> {line}")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Бенчмарк оптимизации локальной LLM (до/после)")
    ap.add_argument("--model", required=True, help="id модели в LM Studio")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--questions", help="файл с вопросами (по одному в строке)")
    ap.add_argument("--rag", action="store_true", help="подставлять контекст из data/rag_index.json")
    ap.add_argument("--isolate", action="store_true",
                    help="варьировать только параметры: RAG и промпт одинаковы в обоих режимах")
    ap.add_argument("--preset", default="legal", help="имя пресета для режима «после» (см. PRESETS)")
    ap.add_argument("--repeat", type=int, default=1, help="прогонов на вопрос (>1 — проверка детерминизма)")
    ap.add_argument("--out", default="docs/bench-legal.md", help="куда сохранить markdown-отчёт")
    args = ap.parse_args()

    if args.preset not in PRESETS:
        print(f"Неизвестный пресет «{args.preset}». Доступны: {', '.join(PRESETS)}")
        return 1

    client = OpenAI(base_url=args.url, api_key="lm-studio")
    questions = load_questions(args.questions)

    rag_index = None
    if args.rag:
        if not INDEX_PATH.exists():
            print("Нет data/rag_index.json — сначала проиндексируйте документы (/rag add в chat.py).")
            return 1
        # эмбеддинг-модель нужна только для поиска; берём первую подходящую
        embed_model = find_embedding_model(args.url, client)
        if not embed_model:
            print("Не найдена эмбеддинг-модель в LM Studio.")
            return 1
        rag_index = RagIndex(client, embed_model)
        print(f"RAG: {len(rag_index.entries)} чанков, эмбеддинги {embed_model}")

    opt_kwargs = preset_kwargs(args.model, args.preset)
    base_kwargs: dict = {}

    print(f"Модель: {args.model} · вопросов: {len(questions)} · повторов: {args.repeat}")
    print(f"«После»: пресет {args.preset} = {opt_kwargs}\n")

    rows: list[dict] = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q[:70]}...")
        context = retrieve(rag_index, q) if args.rag else ""
        # «до»: сырая модель без промпта/RAG; при --isolate — те же промпт и контекст, что «после»
        base = run_mode(client, args.model, q, system=(LEGAL_SYSTEM if args.isolate else None),
                        context=(context if args.isolate else ""), kwargs=base_kwargs, repeat=args.repeat)
        opt = run_mode(client, args.model, q, system=LEGAL_SYSTEM,
                       context=context, kwargs=opt_kwargs, repeat=args.repeat)
        if "error" in base or "error" in opt:
            print(f"    ошибка: {base.get('error') or opt.get('error')}")
        else:
            print(f"    до:    {base['elapsed']:.1f} с, {base['tok_s']:.0f} ток/с, "
                  f"ответ {base['completion_tokens']:.0f} ток")
            print(f"    после: {opt['elapsed']:.1f} с, {opt['tok_s']:.0f} ток/с, "
                  f"ответ {opt['completion_tokens']:.0f} ток")
        rows.append({"question": q, "base": base, "opt": opt})

    meta = {
        "model": args.model, "date": time.strftime("%Y-%m-%d %H:%M"),
        "rag": args.rag, "isolate": args.isolate, "repeat": args.repeat,
        "preset": args.preset, "preset_kwargs": opt_kwargs,
    }
    report = render_report(meta, rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out}\nСырые данные: {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
