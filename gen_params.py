"""Параметры генерации: пер-модельные переопределения поверх дефолтов LM Studio.

Принцип: параметр, который пользователь не задавал, в запрос не попадает вообще —
тогда LM Studio применяет настройки из пресета самой модели. Поэтому «сбросить на
дефолт модели» = просто удалить переопределение. Переопределения хранятся отдельно
для каждой модели в data/params.json.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
PARAMS_PATH = DATA_DIR / "params.json"

# имя: (тип, минимум, максимум, описание); границы None = не проверяем
SPEC = {
    "temperature": (float, 0.0, 2.0, "случайность: 0 — детерминированно, выше — разнообразнее"),
    "max_tokens": (int, 1, None, "лимит длины ответа (у reasoning-моделей включает размышления!)"),
    "context_window": (int, 512, None, "рабочий лимит контекста для порога автосжатия; в API не шлётся, default — брать размер из LM Studio"),
    "top_p": (float, 0.0, 1.0, "nucleus sampling: доля вероятностной массы"),
    "top_k": (int, 1, None, "выбор из k самых вероятных токенов"),
    "min_p": (float, 0.0, 1.0, "отсечка маловероятных токенов"),
    "presence_penalty": (float, -2.0, 2.0, "штраф за сам факт повтора токена"),
    "frequency_penalty": (float, -2.0, 2.0, "штраф, растущий с числом повторов"),
    "repeat_penalty": (float, 0.0, None, "множительный штраф повторов (llama.cpp)"),
    "seed": (int, None, None, "зерно генерации для воспроизводимости"),
}

# параметры OpenAI API — идут аргументами SDK; остальные (LM Studio) — через extra_body
NATIVE = {"temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty", "seed"}

# клиентские параметры: влияют на поведение самого CLI (порог сжатия), в API НЕ отправляются
CLIENT_SIDE = {"context_window"}

# готовые наборы параметров под конкретную задачу. /param preset <имя> применяет разом,
# /param preset off (= /param reset) снимает всё — возврат к дефолтам модели для сравнения «до/после».
PRESETS = {
    # локальный юридический агент: максимальная детерминированность и опора на источники,
    # без «фантазии».
    # max_tokens СПЕЦИАЛЬНО не задаём: локальная reasoning-модель (qwen3.5) тратит на
    # размышления тысячи токенов до ответа (замерено ~5-7к), и любой маленький лимит
    # обрезает её ПОСРЕДИ размышлений -> пустой ответ. Реальный предел длины задаёт
    # контекстное окно (см. /param context_window и раздел про сжатие в ARCHITECTURE.md).
    "legal": {
        "temperature": 0.0,     # закон не терпит креатива: один вопрос -> один и тот же ответ
        "top_p": 0.9,           # страховка, если у пресета модели temperature > 0
        "top_k": 20,            # узкая выборка токенов -> меньше отсебятины
        "min_p": 0.05,          # отсечь совсем маловероятные токены
        "repeat_penalty": 1.1,  # мягкий штраф за повтор формулировок
    },
}


class GenParams:
    """Хранилище переопределений: {model_id: {имя: значение}}."""

    def __init__(self):
        self.by_model: dict[str, dict] = {}
        if PARAMS_PATH.exists():
            self.by_model = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PARAMS_PATH.write_text(json.dumps(self.by_model, ensure_ascii=False, indent=2), encoding="utf-8")

    def for_model(self, model: str) -> dict:
        return self.by_model.get(model, {})

    def set(self, model: str, name: str, raw: str, persist: bool = True) -> object:
        """Валидирует и ставит параметр. Возвращает значение, кидает ValueError."""
        if name not in SPEC:
            raise ValueError(f"Неизвестный параметр «{name}». Доступны: {', '.join(SPEC)}")
        typ, lo, hi, _ = SPEC[name]
        try:
            value = typ(raw.replace(",", "."))  # 0,5 тоже принимаем
        except ValueError:
            raise ValueError(f"{name} должен быть числом ({typ.__name__})")
        if lo is not None and value < lo or hi is not None and value > hi:
            raise ValueError(f"{name}: допустимо от {lo} до {'∞' if hi is None else hi}")
        self.by_model.setdefault(model, {})[name] = value
        if persist:
            self.save()
        return value

    def unset(self, model: str, name: str) -> bool:
        """Убирает переопределение (возврат к дефолту модели). False — и не было."""
        if name in self.by_model.get(model, {}):
            del self.by_model[model][name]
            if not self.by_model[model]:
                del self.by_model[model]
            self.save()
            return True
        return False

    def reset(self, model: str) -> None:
        self.by_model.pop(model, None)
        self.save()

    def apply_preset(self, model: str, name: str) -> dict:
        """Применяет именованный пресет (замена ВСЕХ переопределений модели). Возвращает набор."""
        if name not in PRESETS:
            raise ValueError(f"Неизвестный пресет «{name}». Доступны: {', '.join(PRESETS)}")
        self.by_model[model] = dict(PRESETS[name])
        self.save()
        return self.by_model[model]

    def request_kwargs(self, model: str) -> dict:
        """kwargs для client.chat.completions.create: нативные + extra_body.

        Клиентские параметры (CLIENT_SIDE, напр. context_window) в API не отправляются.
        """
        current = {k: v for k, v in self.for_model(model).items() if k not in CLIENT_SIDE}
        kwargs = {k: v for k, v in current.items() if k in NATIVE}
        extra = {k: v for k, v in current.items() if k not in NATIVE}
        if extra:
            kwargs["extra_body"] = extra
        return kwargs
