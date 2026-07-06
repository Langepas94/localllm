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

    def request_kwargs(self, model: str) -> dict:
        """kwargs для client.chat.completions.create: нативные + extra_body."""
        current = self.for_model(model)
        kwargs = {k: v for k, v in current.items() if k in NATIVE}
        extra = {k: v for k, v in current.items() if k not in NATIVE}
        if extra:
            kwargs["extra_body"] = extra
        return kwargs
