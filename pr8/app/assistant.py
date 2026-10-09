"""Цикл модель → інструменти → модель із журналом викликів."""

import os
import time
from dataclasses import dataclass, field
import json

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()
MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def _add_usage(total: dict, usage: dict | None) -> None:
    if not usage:
        return
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def answer(question: str, customer_id: str) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer("Будь ласка, напишіть запитання.", stopped="input_error")
    if MAX_ROUNDS < 1:
        return Answer("Помічник тимчасово неправильно налаштований: ліміт звертань має бути додатним.", stopped="config_error")

    messages = llm.build_messages(question)
    tool_specs = tools.specs()
    ctx = tools.Context(customer_id=customer_id)
    calls: list[ToolTrace] = []
    usage: dict = {}
    model_elapsed = 0.0
    tools_elapsed = 0.0
    model_name = None

    for round_no in range(1, MAX_ROUNDS + 1):
        try:
            response = llm.chat(messages, tool_specs, tool_choice="auto")
        except llm.LLMError as exc:
            return Answer(str(exc), calls=calls, rounds=round_no - 1, stopped="model_error", model=model_name,
                          elapsed={"model": model_elapsed, "tools": tools_elapsed}, usage=usage or None)

        model_elapsed += response.get("elapsed", 0.0)
        model_name = response.get("model") or model_name
        _add_usage(usage, response.get("usage"))
        message = response["message"]
        messages.append(message)  # повертаємо повідомлення моделі як є

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            text = message.get("content") or "Не вдалося отримати текстову відповідь від моделі."
            return Answer(text, calls=calls, rounds=round_no, stopped="answer", model=model_name,
                          elapsed={"model": model_elapsed, "tools": tools_elapsed}, usage=usage or None)

        # Усі запропоновані виклики одного раунду виконуються. Кожен результат
        # має свій tool_call_id; небезпечні/невалідні виклики лише відхиляються.
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            name = function.get("name", "")
            raw_args = function.get("arguments", "")
            started = time.perf_counter()
            result = tools.call(name, raw_args, ctx)
            elapsed = time.perf_counter() - started
            tools_elapsed += elapsed
            calls.append(ToolTrace(round=round_no, name=name, arguments=raw_args,
                                   status=result.status, reason=result.reason,
                                   result=result.content, elapsed=elapsed))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result.content, ensure_ascii=False),
            })

        # Якщо це останній раунд і модель так і не дала текст — контрольований результат.
        if round_no == MAX_ROUNDS:
            return Answer(
                "Не вдалося завершити відповідь у встановлений ліміт звертань. Спробуйте уточнити запит або повторити його.",
                calls=calls, rounds=round_no, stopped="limit", model=model_name,
                elapsed={"model": model_elapsed, "tools": tools_elapsed}, usage=usage or None,
            )

    return Answer("Не вдалося завершити відповідь.", calls=calls, rounds=MAX_ROUNDS, stopped="limit",
                  model=model_name, elapsed={"model": model_elapsed, "tools": tools_elapsed}, usage=usage or None)
