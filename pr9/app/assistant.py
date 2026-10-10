"""Агентний підхід із ПР8 — те, з чим порівнюється workflow.

Модель сама обирає інструменти й порядок; код перевіряє кожен виклик
(`tools.call`) і обмежує кількість звертань до моделі (`TOOL_MAX_ROUNDS`).
Клієнта визначає `customer_id` з сеансу. Свідомо без підтвердження
оператора: у ПР8 помічник міг створити заявку одразу, і саме це
порівнюється.
"""

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))
LIMIT_TEXT = ("Не вдалося завершити обробку звернення автоматично за відведену кількість "
              "кроків. Передаємо його оператору.")
ERROR_TEXT = "Зараз не вдається обробити звернення. Спробуйте, будь ласка, пізніше."


@dataclass
class ToolTrace:
    """Запис про один виклик інструмента (`round` — з 1)."""

    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    """Результат циклу: `text`, журнал `calls`, `rounds`, `stopped`
    (`answer` | `limit` | `llm_error`), `elapsed` за видами, `usage`."""

    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(question: str, customer_id: str) -> Answer:
    ctx = tools.Context(customer_id)
    messages = llm.build_messages(question)
    specs = tools.specs()
    out = Answer(text="")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t_model = t_tools = 0.0
    try:
        for rnd in range(1, MAX_ROUNDS + 1):
            # На останньому звертанні інструменти вимкнено: потрібна відповідь текстом.
            choice = "none" if rnd == MAX_ROUNDS else "auto"
            reply = llm.chat(messages, specs, choice)
            out.rounds = rnd
            out.model = reply["model"]
            t_model += reply["elapsed"]
            for k in usage:
                usage[k] += reply["usage"].get(k, 0)
            messages.append(reply["message"])
            if not reply["tool_calls"] or choice == "none":
                out.text = reply["content"].strip()
                break
            for c in reply["tool_calls"]:
                res = tools.call(c["name"], c["arguments"], ctx)
                t_tools += res.elapsed
                out.calls.append(ToolTrace(rnd, c["name"], c["arguments"], res.status,
                                           res.reason, res.content, res.elapsed))
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(
                                     res.content if res.content is not None
                                     else {"error": res.reason}, ensure_ascii=False)})
        if not out.text:
            out.text, out.stopped = LIMIT_TEXT, "limit"
    except llm.LLMError as exc:
        out.text, out.stopped = ERROR_TEXT, f"llm_error:{exc.kind}"
    except Exception as exc:  # noqa: BLE001 — не 500
        out.text, out.stopped = ERROR_TEXT, f"error:{type(exc).__name__}"
    out.usage = usage
    out.elapsed = {"model": round(t_model, 4), "tools": round(t_tools, 4)}
    return out
