"""CLI-чат с локальной LLM через LM Studio: память, RAG, автосжатие контекста."""

import argparse
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI, APIConnectionError, APIError, BadRequestError
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

from gen_params import PRESETS, SPEC, GenParams
from memory import KEEP_MESSAGES, Session, compress, list_sessions, session_preview
from rag import (
    ANSWER_MIN_SCORE, INDEX_PATH, RERANK_CANDIDATES, TOP_K, RagIndex,
    relevance_filter, rerank, rewrite_query, section_of,
)
from user_profile import MODES, Profile

DEFAULT_URL = "http://localhost:1234/v1"
DEFAULT_CONTEXT = 8192        # если LM Studio не сообщил размер контекста
COMPRESS_THRESHOLD = 0.8      # доля заполнения контекста, при которой жмём историю

HELP_TEXT = """\
Команды:
  /help              — эта справка
  /model             — сменить модель (история сохраняется)
  /system <текст>    — задать системный промпт; без текста — показать текущий
  /clear             — очистить историю и саммари текущей сессии
  /new               — начать новую сессию
  /sessions          — список сохранённых сессий
  /load <номер>      — продолжить сессию из списка /sessions
  /compress          — сжать историю вручную (последние 10 сообщений + саммари)
  /stats             — заполнение контекста, размер истории, профиль, RAG
  /profile           — факты о вас (подставляются в промпт только к месту):
                       add <факт> | edit <n> <текст> | del <n> | triggers <n>
                       hint <n> <вопрос> | mode smart|always|off
  /legal on | off    — юридический режим разом: систем-промпт юр-агента + параметры
                       генерации (preset legal) + ретривал (top_k, дедуп по статье, фильтр).
                       off — базовый вариант для сравнения «до/после». Главный демо-переключатель.
  /param             — параметры генерации для текущей модели:
                       <имя> <значение> — задать (temperature, max_tokens, context_window,
                       top_p, top_k, min_p, presence/frequency/repeat_penalty, seed)
                       preset legal — пресет юр-агента | preset off — снять (дефолты модели)
                       <имя> default — вернуть дефолт модели | reset — сбросить все
  /debug             — вкл/выкл строку статистики после каждого ответа
  /rag add <путь>    — проиндексировать файл или папку (txt, md, pdf, py, ...)
  /rag on | off      — включить/выключить подстановку контекста из базы
  /rag status        — что в индексе и какие улучшения ретривала включены
  /rag clear         — очистить индекс
  /rag rewrite on|off — переписывать запрос по истории перед поиском
  /rag rerank on|off  — LLM-переранжирование расширенной выборки
  /rag filter on|off  — отсекать слабые по близости фрагменты (без LLM)
  /exit, /quit       — выйти
Ctrl+C во время генерации — прервать ответ, не выходя из чата.\
"""

# команда -> (описание для меню, подкоманды)
COMMANDS: dict[str, tuple[str, list[str]]] = {
    "/help": ("справка по командам", []),
    "/model": ("сменить модель", []),
    "/system": ("системный промпт: показать или задать", []),
    "/clear": ("очистить историю и саммари", []),
    "/new": ("новая сессия", []),
    "/sessions": ("список сохранённых сессий", []),
    "/load": ("продолжить сессию по номеру", []),
    "/compress": ("сжать историю в саммари", []),
    "/stats": ("статистика: контекст, история, профиль, RAG", []),
    "/profile": ("факты о вас для промпта", ["add", "edit", "del", "triggers", "hint", "mode", "show"]),
    "/legal": ("юридический режим (параметры + ретривал) вкл/выкл", ["on", "off"]),
    "/param": ("параметры генерации модели", list(SPEC) + ["preset", "reset"]),
    "/rag": ("база знаний из ваших файлов", ["add", "on", "off", "status", "clear", "rewrite", "rerank", "filter"]),
    "/debug": ("вкл/выкл строку статистики", []),
    "/exit": ("выйти", []),
    "/quit": ("выйти", []),
}

# третий уровень: ("/команда", "подкоманда") -> варианты
THIRD_LEVEL = {("/profile", "mode"): ["smart", "always", "off"]}
THIRD_LEVEL.update({("/param", name): ["default"] for name in SPEC})
THIRD_LEVEL[("/param", "preset")] = list(PRESETS) + ["off"]
THIRD_LEVEL.update({("/rag", sub): ["on", "off"] for sub in ("rewrite", "rerank", "filter")})

# Профили ретривала для /legal. Два чётких режима, RAG включён в обоих:
#   ВКЛ  — ВСЕ оптимизации (rewrite follow-up'ов, rerank, фильтр, дедуп, широкий top_k, grounding-инструкция);
#   ВЫКЛ — голый RAG без единой оптимизации (просто чанки + вопрос, без инструкций и порога «не знаю»).
# grounding=True добавляет в контекст указание «отвечай только по фрагментам» + порог ANSWER_MIN_SCORE.
LEGAL_RETR = {"rewrite": True, "rerank": True, "filter": True, "top_k": 8, "dedup": True, "grounding": True}
BASE_RETR = {"rewrite": False, "rerank": False, "filter": False, "top_k": 3, "dedup": False, "grounding": False}

# Системный промпт юр-агента: guardrails, обязательные для юридического ассистента —
# факты и опора на нормы, точные ссылки, без гарантий исхода и без замены живого юриста.
LEGAL_SYSTEM = (
    "Ты — юридический ассистент по праву РФ. Соблюдай правила:\n"
    "1. Отвечай ТОЛЬКО на основе предоставленных фрагментов законодательства. Если их нет "
    "или в них нет ответа — прямо скажи «Не знаю» и попроси уточнить. Не выдумывай нормы, "
    "номера статей, даты и цитаты.\n"
    "2. Ссылайся на конкретные статьи (например, «согласно ст. 80 ТК РФ») и цитируй точно.\n"
    "3. Если фрагменты дают РАЗНЫЕ сроки/правила для разных категорий работников, "
    "по умолчанию отвечай ОБЩЕЙ нормой (для обычного работника), а особые случаи "
    "(испытательный срок, сезонные, срочный договор до 2 месяцев, руководитель, спортсмены) "
    "укажи отдельно как исключения с их статьями. Не выдавай частный случай за общее правило. "
    "Если вопрос явно про такую категорию — отвечай нормой для неё.\n"
    "4. Излагай факты и содержание нормы. Не давай оценок «выиграете/проиграете», не "
    "прогнозируй и не гарантируй исход дела.\n"
    "5. Не давай индивидуальных юридических советов и не заменяй консультацию юриста; при "
    "реальном споре рекомендуй обратиться к квалифицированному специалисту.\n"
    "6. Если норма могла измениться или вопрос выходит за пределы предоставленных документов "
    "— предупреди об этом.\n"
    "Пиши по-русски, кратко и по существу."
)


class SlashCompleter(Completer):
    """Подсказки команд при наборе «/»: команды, подкоманды, значения."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        parts = text.split(" ")
        if len(parts) == 1:
            for cmd, (desc, _) in COMMANDS.items():
                if cmd.startswith(parts[0]):
                    yield Completion(cmd, start_position=-len(parts[0]), display_meta=desc)
        elif len(parts) == 2 and parts[0] in COMMANDS:
            for sub in COMMANDS[parts[0]][1]:
                if sub.startswith(parts[1]):
                    meta = SPEC[sub][3] if parts[0] == "/param" and sub in SPEC else None
                    yield Completion(sub, start_position=-len(parts[1]), display_meta=meta)
        elif len(parts) == 3 and (parts[0], parts[1]) in THIRD_LEVEL:
            for value in THIRD_LEVEL[(parts[0], parts[1])]:
                if value.startswith(parts[2]):
                    yield Completion(value, start_position=-len(parts[2]))


def api_v0_models(base_url: str) -> list[dict] | None:
    """Расширенный список моделей LM Studio (/api/v0): тип, state, размер контекста."""
    api = base_url.rstrip("/").removesuffix("/v1") + "/api/v0/models"
    try:
        return httpx.get(api, timeout=5).json()["data"]
    except Exception:
        return None


def choose_model(client: OpenAI, base_url: str) -> tuple[str, int] | None:
    """Выбор модели. Возвращает (id, размер контекста) или None при ошибке."""
    infos = api_v0_models(base_url)
    if infos is not None:
        chat_models = [m for m in infos if m.get("type") in ("llm", "vlm")]
    else:  # /api/v0 недоступен — обычный список без пометок
        try:
            chat_models = [{"id": m.id} for m in client.models.list().data]
        except APIConnectionError:
            print("Не удалось подключиться к LM Studio. Проверьте, что сервер запущен (Developer -> Start Server).")
            return None
    if not chat_models:
        print("В LM Studio нет чат-моделей. Загрузите модель и повторите.")
        return None

    def limit_of(m: dict) -> int:
        return m.get("loaded_context_length") or m.get("max_context_length") or DEFAULT_CONTEXT

    if len(chat_models) == 1:
        m = chat_models[0]
        print(f"Модель: {m['id']}")
        return m["id"], limit_of(m)
    print("Доступные модели:")
    for i, m in enumerate(chat_models, 1):
        mark = "  [загружена]" if m.get("state") == "loaded" else ""
        print(f"  {i}. {m['id']}{mark}")
    while True:
        raw = input(f"Выберите модель [1-{len(chat_models)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(chat_models):
            m = chat_models[int(raw) - 1]
            return m["id"], limit_of(m)
        print("Введите номер из списка.")


def find_embedding_model(base_url: str, client: OpenAI) -> str | None:
    # reranker'ы LM Studio помечает type=embeddings, но для векторного поиска они
    # не годятся (дают relevance-скор пары, а не эмбеддинг) — исключаем по имени.
    def ok(mid: str) -> bool:
        return "rerank" not in mid.lower()

    infos = api_v0_models(base_url)
    if infos:
        embs = [m for m in infos if m.get("type") == "embeddings" and ok(m["id"])]
        embs.sort(key=lambda m: m.get("state") != "loaded")  # уже загруженные первыми
        if embs:
            return embs[0]["id"]
    try:
        for m in client.models.list().data:
            if "embed" in m.id.lower() and ok(m.id):
                return m.id
    except APIConnectionError:
        pass
    return None


def warm_up_embeddings(client: OpenAI, embed_model: str) -> bool:
    """Заранее подгружает эмбеддинг-модель в LM Studio (через JIT), чтобы RAG и
    профиль не тормозили на первом запросе. False — если JIT выключен/ошибка."""
    try:
        client.embeddings.create(model=embed_model, input=["прогрев"])
        return True
    except Exception:
        return False


def effective_context(params: GenParams, model: str, hw_limit: int) -> int:
    """Рабочий лимит контекста: override пользователя (/param context_window) или размер,
    сообщённый LM Studio. Управляет порогом автосжатия; в запрос к модели не отправляется."""
    override = params.for_model(model).get("context_window")
    return int(override) if override else hw_limit


def safe_top_k(requested: int, context_limit: int) -> int:
    """Страховка от обрыва ответа: на маленьком окне режем число чанков в промпте, чтобы
    reasoning-модели осталось место на размышления (она тратит тысячи токенов до ответа;
    иначе промпт+reasoning переполняют окно и ответ выходит пустым). Пороги эмпирические:
    на ≤8k места хватает только на ~4 чанка, на 12-16k — на ~6, дальше — сколько просят.
    Дедуп по статье компенсирует урезание (в 3-4 чанка попадают разные нормы)."""
    if context_limit >= 16000:
        return requested
    if context_limit >= 12000:
        return min(requested, 6)
    return min(requested, 4)


def stream_reply(client: OpenAI, model: str, messages: list[dict], gen_kwargs: dict):
    """Стримит ответ в stdout. Возвращает (текст | None, usage | None)."""
    kwargs = dict(model=model, messages=messages, stream=True, **gen_kwargs)
    try:
        try:
            stream = client.chat.completions.create(**kwargs, stream_options={"include_usage": True})
        except BadRequestError:  # старый LM Studio без include_usage
            stream = client.chat.completions.create(**kwargs)
        parts, usage, finish, reasoned = [], None, None, False
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            if getattr(choice.delta, "reasoning_content", None):
                reasoned = True  # модель «думала» (reasoning уходит в отдельное поле)
            if choice.delta.content:
                parts.append(choice.delta.content)
                print(choice.delta.content, end="", flush=True)
        print()
        if not parts:
            # различаем обрыв по окну (finish=length) и просто пустой ответ
            if finish == "length" or reasoned:
                print("[ответ не поместился в контекстное окно: reasoning-модель израсходовала его "
                      "на размышления (их у неё нельзя отключить). Поднимите loaded_context_length "
                      "модели в LM Studio (сейчас окно мало) — это надёжный фикс; либо /legal on "
                      "(temperature 0 даёт короткие стабильные размышления, чаще умещается)]")
            else:
                print("[пустой ответ от модели]")
        return "".join(parts) or None, usage
    except KeyboardInterrupt:
        print("\n[генерация прервана]")
        return ("".join(parts) or None) if parts else None, None
    except APIConnectionError:
        print("\nПотеряно соединение с LM Studio.")
        return None, None
    except APIError as e:
        # частый случай на reasoning-модели с малым окном: размышления переполняют контекст
        if "context" in str(e).lower():
            print("\n[переполнено контекстное окно: reasoning-модель не уместила размышления + ответ. "
                  "Поднимите loaded_context_length модели в LM Studio (сейчас окно мало) — надёжный фикс; "
                  "либо /legal on: temperature 0 даёт короткие стабильные размышления]")
        else:
            print(f"\nОшибка API: {e}")
        return None, None


def maybe_compress(session: Session, client: OpenAI, tokens_used: int, context_limit: int) -> None:
    """Если контекст заполнен на COMPRESS_THRESHOLD — сжимает историю."""
    if tokens_used <= context_limit * COMPRESS_THRESHOLD:
        return
    if len(session.messages) <= KEEP_MESSAGES:
        print(f"[контекст заполнен на {100 * tokens_used // context_limit}%, "
              f"но в истории уже только {len(session.messages)} сообщений — сжимать нечего]")
        return
    print(f"[контекст заполнен на {100 * tokens_used // context_limit}%, сжимаю историю...]")
    if compress(session, client):
        session.save()
        print(f"[история сжата: саммари + последние {KEEP_MESSAGES} сообщений]")
    else:
        print("[не удалось сжать историю, продолжаем как есть]")


def print_stats(session: Session, tokens_used: int, context_limit: int, rag_index, rag_on: bool,
                profile: Profile, params: GenParams) -> None:
    pct = 100 * tokens_used // context_limit if context_limit else 0
    print(f"Модель: {session.model}, контекст: {context_limit} токенов")
    print(f"Последний запрос: ~{tokens_used} токенов ({pct}%), порог сжатия — {int(COMPRESS_THRESHOLD * 100)}%")
    print(f"История: {len(session.messages)} сообщений, саммари: {'есть' if session.summary else 'нет'}")
    print(f"Сессия: {session.id} -> {session.path}")
    print(f"Профиль: {len(profile.facts)} фактов, режим {profile.mode}")
    overrides = params.for_model(session.model)
    shown = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "все — дефолты модели"
    print(f"Параметры генерации: {shown}")
    if rag_index and rag_index.entries:
        state = "включён" if rag_on else "выключен"
        print(f"RAG: {state}, {len(rag_index.entries)} чанков из {len(rag_index.sources())} файлов")
    else:
        print("RAG: индекс пуст (/rag add <путь>)")


RETR_LABELS = {"rewrite": "Переписывание запроса", "rerank": "LLM-переранжирование",
               "filter": "Фильтр релевантности"}


def handle_rag_command(arg: str, rag_index, rag_on: bool, retr: dict, client: OpenAI, base_url: str):
    """Обрабатывает /rag ... Возвращает (rag_index, rag_on); retr правит на месте."""
    sub, _, rest = arg.partition(" ")
    sub, rest = sub.lower(), rest.strip()
    if sub in RETR_LABELS:
        if rest in ("on", "off"):
            retr[sub] = rest == "on"
            print(f"{RETR_LABELS[sub]}: {'включено' if retr[sub] else 'выключено'}.")
        else:
            print(f"Использование: /rag {sub} on|off")
        return rag_index, rag_on
    if rag_index is None and sub in ("add", "on", "status", "clear"):
        embed_model = find_embedding_model(base_url, client)
        if not embed_model:
            print("В LM Studio не найдена эмбеддинг-модель. Скачайте, например, nomic-embed-text.")
            return None, False
        rag_index = RagIndex(client, embed_model)
        print(f"[эмбеддинги: {embed_model}]")
    if sub == "add":
        if not rest:
            print("Использование: /rag add <путь к файлу или папке>")
        else:
            try:
                files, chunks = rag_index.add_path(rest)
                print(f"Проиндексировано: {files} файлов, {chunks} чанков. RAG включён.")
                rag_on = True
            except FileNotFoundError:
                print(f"Не найдено: {rest}")
            except RuntimeError as e:
                print(str(e))
            except (APIConnectionError, APIError) as e:
                print(f"Ошибка эмбеддингов: {e}")
    elif sub == "on":
        if rag_index.entries:
            rag_on = True
            print("RAG включён.")
        else:
            print("Индекс пуст — сначала /rag add <путь>.")
    elif sub == "off":
        rag_on = False
        print("RAG выключен.")
    elif sub == "status":
        if rag_index.entries:
            print(f"RAG {'включён' if rag_on else 'выключен'}, {len(rag_index.entries)} чанков:")
            for src, n in rag_index.sources().items():
                print(f"  {src} — {n}")
        else:
            print("Индекс пуст.")
        enh = ", ".join(f"{name}={'вкл' if retr[key] else 'выкл'}"
                        for key, name in RETR_LABELS.items())
        print(f"Ретривал: {enh}, top_k={retr['top_k']}, дедуп по статье={'вкл' if retr['dedup'] else 'выкл'}, "
              f"grounding-инструкция={'вкл' if retr['grounding'] else 'выкл'}")
    elif sub == "clear":
        rag_index.clear()
        rag_on = False
        print("Индекс очищен, RAG выключен.")
    else:
        print("Подкоманды: /rag add <путь> | on | off | status | clear | rewrite/rerank/filter on|off")
    return rag_index, rag_on


def handle_profile_command(arg: str, profile: Profile, model: str) -> None:
    sub, _, rest = arg.partition(" ")
    sub, rest = sub.lower(), rest.strip()
    if not sub or sub == "show":
        if not profile.facts:
            print("Профиль пуст. /profile add <факт о вас> — добавить.")
            return
        print(f"Режим: {profile.mode} (smart — только релевантные факты, always — все, off — не использовать)")
        for i, f in enumerate(profile.facts, 1):
            print(f"  {i}. {f['text']} ({len(f['triggers'])} триггеров)")
    elif sub == "add":
        if not rest:
            print("Использование: /profile add <факт о вас>")
            return
        print("[генерирую триггеры — примеры вопросов, при которых факт пригодится...]")
        triggers = profile.add(rest, model)
        if triggers:
            print(f"Добавлено. Триггеры ({len(triggers)}), например: {'; '.join(triggers[:3])}")
        else:
            print("Добавлено, но триггеры сгенерировать не удалось — факт будет подставляться всегда.")
    elif sub == "edit":
        num, _, text = rest.partition(" ")
        text = text.strip()
        if num.isdigit() and 1 <= int(num) <= len(profile.facts) and text:
            print("[генерирую триггеры заново...]")
            profile.edit(int(num) - 1, text, model)
            print("Факт обновлён.")
        else:
            print("Использование: /profile edit <номер> <новый текст>")
    elif sub == "del":
        if rest.isdigit() and 1 <= int(rest) <= len(profile.facts):
            removed = profile.facts[int(rest) - 1]["text"]
            profile.delete(int(rest) - 1)
            print(f"Удалено: {removed}")
        else:
            print("Использование: /profile del <номер>")
    elif sub == "triggers":
        if rest.isdigit() and 1 <= int(rest) <= len(profile.facts):
            f = profile.facts[int(rest) - 1]
            print(f"Триггеры факта «{f['text']}»:")
            for t in f["triggers"] or ["(нет — факт подставляется всегда)"]:
                print(f"  - {t}")
        else:
            print("Использование: /profile triggers <номер>")
    elif sub == "hint":
        num, _, text = rest.partition(" ")
        text = text.strip()
        if num.isdigit() and 1 <= int(num) <= len(profile.facts) and text:
            if profile.add_trigger(int(num) - 1, text):
                print("Триггер добавлен.")
            else:
                print("Не удалось получить эмбеддинг — триггер не добавлен.")
        else:
            print("Использование: /profile hint <номер факта> <пример вопроса>")
    elif sub == "mode":
        if rest in MODES:
            profile.mode = rest
            profile.save()
            print(f"Режим профиля: {rest}")
        else:
            print("Режимы: smart (только релевантные факты) | always (все) | off (не использовать)")
    else:
        print("Подкоманды: /profile [show] | add <факт> | edit <n> <текст> | del <n> | "
              "triggers <n> | hint <n> <вопрос> | mode smart|always|off")


def handle_legal_command(arg: str, params: GenParams, model: str, retr: dict,
                         rag_index, rag_on: bool, session: Session, default_system: str) -> bool:
    """Зонтичный переключатель юр-режима: систем-промпт + генерация (preset legal) + ретривал.

    Правит params, retr и session.system_prompt на месте; возвращает новое rag_on. Один
    тумблер «до/после»: on — всё оптимизированное, off — базовый вариант модели.
    """
    arg = arg.strip().lower()
    if arg not in ("on", "off"):
        print("Использование: /legal on | off  (юр-режим: систем-промпт + параметры + ретривал)")
        return rag_on
    if arg == "on":
        params.apply_preset(model, "legal")
        retr.update(LEGAL_RETR)
        session.system_prompt = LEGAL_SYSTEM
        session.save()
        if rag_index and rag_index.entries:
            rag_on = True
        print("Юридический режим ВКЛ — все оптимизации (RAG включён):")
        print("  систем-промпт — юр-агент (факты, ссылки на статьи, без гарантий исхода, не заменяет юриста)")
        pk = ", ".join(f"{k}={v}" for k, v in PRESETS["legal"].items())
        print(f"  генерация — preset legal ({pk})")
        print(f"  ретривал  — top_k={retr['top_k']}, дедуп по статье, rewrite+rerank+фильтр, grounding-инструкция")
        print("  ⚠ rewrite и rerank — по одному LLM-вызову на сообщение (медленнее; при желании /rag rerank off)")
        if not (rag_index and rag_index.entries):
            print("  ⚠ RAG-индекс пуст — добавьте документ: /rag add <путь>")
    else:
        params.reset(model)
        retr.update(BASE_RETR)
        session.system_prompt = default_system
        session.save()
        print("Юридический режим ВЫКЛ — голый RAG без оптимизаций (RAG включён):")
        print(f"  систем-промпта нет{' (исходный пустой)' if not default_system else ' (восстановлен исходный)'}; "
              f"дефолты модели; ретривал top_k={retr['top_k']}, без дедупа/фильтра/rewrite/rerank и без grounding-инструкции")
    return rag_on


def handle_param_command(arg: str, params: GenParams, model: str) -> None:
    name, _, value = arg.partition(" ")
    name, value = name.lower(), value.strip()
    if not name:
        current = params.for_model(model)
        print(f"Параметры генерации для {model} (не заданные берутся из пресета модели в LM Studio):")
        for p, (_, _, _, desc) in SPEC.items():
            shown = current.get(p, "дефолт модели")
            print(f"  {p} = {shown}  — {desc}")
        print("Задать: /param <имя> <значение>; сбросить: /param <имя> default; всё: /param reset")
    elif name == "reset":
        params.reset(model)
        print(f"Все параметры для {model} сброшены на дефолты модели.")
    elif name == "preset":
        if not value or value.lower() in ("list", "?"):
            print("Пресеты параметров под задачу:")
            for pname, pvals in PRESETS.items():
                shown = ", ".join(f"{k}={v}" for k, v in pvals.items())
                print(f"  {pname} — {shown}")
            print("Применить: /param preset <имя>; снять (дефолты модели, для сравнения): /param preset off")
        elif value.lower() in ("off", "выкл", "baseline", "-"):
            params.reset(model)
            print(f"Пресет снят: параметры {model} — дефолты модели (базовый вариант для сравнения «до/после»).")
        else:
            try:
                applied = params.apply_preset(model, value.lower())
                shown = ", ".join(f"{k}={v}" for k, v in applied.items())
                print(f"Пресет «{value.lower()}» применён к {model}: {shown}")
            except ValueError as e:
                print(e)
    elif value.lower() in ("default", "дефолт", "-"):
        if params.unset(model, name):
            print(f"{name} сброшен на дефолт модели.")
        else:
            print(f"{name} и так не был задан (действует дефолт модели).")
    elif not value:
        print("Использование: /param <имя> <значение> | /param <имя> default | /param reset")
    else:
        try:
            print(f"{name} = {params.set(model, name, value)}")
        except ValueError as e:
            print(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Чат с локальной LLM через LM Studio")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"адрес API LM Studio (по умолчанию {DEFAULT_URL})")
    parser.add_argument("--model", help="id модели (по умолчанию — выбор из списка)")
    parser.add_argument("--system", default="", help="системный промпт для новой сессии")
    parser.add_argument("--temperature", type=float, default=None,
                        help="температура на эту сессию (по умолчанию — дефолт модели)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.url, api_key="lm-studio")

    hw_context = DEFAULT_CONTEXT  # размер окна из LM Studio (задаётся при загрузке модели)
    if args.model:
        model = args.model
        infos = api_v0_models(args.url) or []
        info = next((m for m in infos if m["id"] == model), None)
        if info:
            hw_context = info.get("loaded_context_length") or info.get("max_context_length") or DEFAULT_CONTEXT
    else:
        chosen = choose_model(client, args.url)
        if not chosen:
            return 1
        model, hw_context = chosen

    session = Session(system_prompt=args.system, model=model)
    embed_model = find_embedding_model(args.url, client)
    profile = Profile(client, embed_model)
    params = GenParams()
    if args.temperature is not None:  # флаг действует только на эту сессию
        params.set(model, "temperature", str(args.temperature), persist=False)
    # эффективный лимит: override /param context_window, иначе размер из LM Studio
    context_limit = effective_context(params, model, hw_context)
    rag_index = None
    rag_on = False
    # настройки ретривала (работают только при включённом RAG); top_k/dedup/grounding — под юр-режим
    retr = {"rewrite": True, "rerank": True, "filter": True, "top_k": TOP_K, "dedup": False, "grounding": True}
    debug = True   # строка статистики после каждого ответа
    tokens_used = 0  # ~размер последнего запроса по данным usage или оценке

    print(f"\nЧат с {model} (контекст {context_limit} токенов). /help — список команд, «/» покажет подсказки.")
    if list_sessions():
        print("Есть сохранённые сессии: /sessions — список, /load <номер> — продолжить.")

    # прогрев эмбеддинг-модели, если она понадобится (профиль smart с фактами или есть RAG-индекс)
    needs_embeddings = (profile.mode == "smart" and profile.facts) or INDEX_PATH.exists()
    if embed_model and needs_embeddings:
        print(f"Загружаю эмбеддинг-модель {embed_model}...", end="", flush=True)
        if warm_up_embeddings(client, embed_model):
            print(" готово.")
        else:
            print(" не удалось (включите JIT loading в LM Studio или загрузите её вручную).")
    print()

    # подсказки команд только в интерактивном терминале; при пайпе — обычный input
    prompt_session = PromptSession(completer=SlashCompleter(), complete_while_typing=True) \
        if sys.stdin.isatty() else None

    while True:
        try:
            if prompt_session:
                user_input = prompt_session.prompt("Вы: ").strip()
            else:
                # при пайпе из PowerShell в начало потока попадает BOM
                user_input = input("Вы: ").strip().strip("﻿").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПока!")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, arg = user_input.partition(" ")
            cmd, arg = cmd.lower(), arg.strip()
            if cmd in ("/exit", "/quit"):
                print("Пока!")
                return 0
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/clear":
                session.messages.clear()
                session.summary = ""
                session.save()
                print("История и саммари очищены.")
            elif cmd == "/new":
                session = Session(system_prompt=args.system, model=model)
                tokens_used = 0
                print("Новая сессия.")
            elif cmd == "/system":
                if arg:
                    session.system_prompt = arg
                    session.save()
                    print("Системный промпт задан.")
                else:
                    print(f"Текущий системный промпт: {session.system_prompt or '(не задан)'}")
            elif cmd == "/model":
                chosen = choose_model(client, args.url)
                if chosen:
                    model, hw_context = chosen
                    context_limit = effective_context(params, model, hw_context)
                    session.model = model
                    session.save()
                    print(f"Модель: {model} (контекст {context_limit}). История сохранена.")
            elif cmd == "/sessions":
                paths = list_sessions()
                if not paths:
                    print("Сохранённых сессий нет.")
                for i, p in enumerate(paths, 1):
                    print(f"  {i}. {session_preview(p)}")
            elif cmd == "/load":
                paths = list_sessions()
                if arg.isdigit() and 1 <= int(arg) <= len(paths):
                    session = Session.from_file(paths[int(arg) - 1])
                    if session.model != model and session.model:
                        print(f"[сессия велась с {session.model}, сейчас будет использоваться {model}]")
                    session.model = model
                    tokens_used = session.estimate_tokens()
                    print(f"Загружена сессия {session.id}: {len(session.messages)} сообщений"
                          f"{', есть саммари' if session.summary else ''}.")
                else:
                    print("Использование: /load <номер из /sessions>")
            elif cmd == "/compress":
                if compress(session, client):
                    session.save()
                    print(f"История сжата: саммари + последние {KEEP_MESSAGES} сообщений.")
                else:
                    print(f"Сжимать нечего: в истории не больше {KEEP_MESSAGES} сообщений.")
            elif cmd == "/stats":
                context_limit = effective_context(params, model, hw_context)
                print_stats(session, tokens_used, context_limit, rag_index, rag_on, profile, params)
            elif cmd == "/rag":
                rag_index, rag_on = handle_rag_command(arg, rag_index, rag_on, retr, client, args.url)
            elif cmd == "/profile":
                handle_profile_command(arg, profile, model)
            elif cmd == "/legal":
                rag_on = handle_legal_command(arg, params, model, retr, rag_index, rag_on,
                                              session, args.system)
            elif cmd == "/param":
                handle_param_command(arg, params, model)
            elif cmd == "/debug":
                debug = not debug
                print(f"Дебаг-строка {'включена' if debug else 'выключена'}.")
            else:
                print(f"Неизвестная команда: {cmd}. /help — список команд.")
            continue

        # рабочий лимит контекста мог измениться через /param context_window
        context_limit = effective_context(params, model, hw_context)

        # обычное сообщение: профиль + RAG -> запрос -> сохранение -> сжатие
        rag_context = ""
        rag_hits: list[dict] = []
        if rag_on and rag_index and rag_index.entries:
            try:
                # rewrite -> шире достаём -> фильтр релевантности -> rerank -> топ-K
                query = user_input
                if retr["rewrite"]:
                    query = rewrite_query(client, model, session.messages, user_input)
                    if debug and query != user_input:
                        print(f"[rewrite: {query[:80]}]")
                # страховка: на маленьком окне режем top_k, чтобы reasoning-модели
                # осталось место на размышления (иначе пустой ответ)
                eff_top_k = safe_top_k(retr["top_k"], context_limit)
                if debug and eff_top_k < retr["top_k"]:
                    print(f"[страховка: top_k {retr['top_k']}→{eff_top_k} — окно {context_limit} "
                          f"мало для reasoning-модели; для полного top_k поднимите контекст в LM Studio]")
                fetch = RERANK_CANDIDATES if retr["rerank"] else eff_top_k
                hits = rag_index.search(query, top_k=fetch, dedup=retr["dedup"])
                best_score = hits[0]["score"] if hits else 0.0
                if retr["filter"]:
                    hits = relevance_filter(hits)
                hits = rerank(client, model, query, hits) if retr["rerank"] else hits[:eff_top_k]
                # grounding-режим (юр-агент): порог уверенности + строгая инструкция «только по фрагментам»
                if retr["grounding"]:
                    if not hits or best_score < ANSWER_MIN_SCORE:
                        if debug:
                            print(f"[RAG: релевантность низкая (лучший score {best_score:.2f} "
                                  f"< {ANSWER_MIN_SCORE}) — отвечаю «не знаю»]")
                        print("LLM: Не знаю — в проиндексированных документах нет достаточно "
                              "релевантной информации по вашему вопросу. Уточните формулировку "
                              "или добавьте нужный документ через «/rag add».")
                        continue
                    rag_hits = hits
                    chunks = "\n---\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
                    rag_context = (
                        "Отвечай, опираясь ТОЛЬКО на пронумерованные фрагменты ниже. Если в них "
                        "нет ответа — напиши «Не знаю» и попроси уточнить, ничего не выдумывай.\n\n"
                        + chunks
                    )
                else:
                    # голый RAG (baseline): просто подставляем найденное, без порога и без инструкций
                    rag_hits = hits
                    rag_context = "\n---\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
                if debug:
                    print(f"[RAG: {len(rag_hits)} фрагментов, лучший score {best_score:.2f}]")
            except (APIConnectionError, APIError) as e:
                print(f"[RAG не сработал: {e}]")

        profile_facts = profile.relevant(user_input)
        profile_block = "\n".join(f"- {f}" for f in profile_facts)
        if profile_facts and debug:
            print(f"[профиль: {'; '.join(f[:40] for f in profile_facts)}]")

        session.messages.append({"role": "user", "content": user_input})
        messages = session.build_messages(rag_context, profile_block)

        print("LLM: ", end="", flush=True)
        started = time.perf_counter()
        reply, usage = stream_reply(client, model, messages, params.request_kwargs(model))
        elapsed = time.perf_counter() - started
        if not reply:
            session.messages.pop()  # не сохраняем вопрос без ответа
            continue
        session.messages.append({"role": "assistant", "content": reply})
        session.save()

        # структура ответа: источники и цитаты собираем из самих чанков (без LLM)
        if rag_hits:
            print("\nИсточники:")
            for i, h in enumerate(rag_hits, 1):
                print(f"  [{i}] {Path(h['source']).name} · {section_of(h['text'])}")
            print("Цитаты:")
            for i, h in enumerate(rag_hits, 1):
                frag = " ".join(h["text"].split())[:220]
                print(f"  [{i}] {frag}…")

        tokens_used = usage.total_tokens if usage else session.estimate_tokens()
        if debug:
            pct = 100 * tokens_used // context_limit
            if usage:
                tps = usage.completion_tokens / elapsed if elapsed > 0 else 0
                print(f"[{elapsed:.1f} с | промпт {usage.prompt_tokens} + ответ {usage.completion_tokens} "
                      f"= {usage.total_tokens} ток. | {tps:.0f} ток/с | контекст {pct}% из {context_limit}]")
            else:
                print(f"[{elapsed:.1f} с | ~{tokens_used} ток. (оценка) | контекст {pct}% из {context_limit}]")
        maybe_compress(session, client, tokens_used, context_limit)


if __name__ == "__main__":
    sys.exit(main())
