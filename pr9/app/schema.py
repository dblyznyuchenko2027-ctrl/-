"""Контракт кроку розбору: що модель має витягти зі звернення й як це
перевіряється.

Рішення:

* `intent` — вид: `return`, `cancel`, `status`, `other`, `unclear`
  («не розумію»), `multiple` (дві й більше різні дії). «Не розумію» і
  «інше» — різні значення: перше веде до уточнення, друге — до
  оператора;
* `order_id` — рядок або `null` («не названо»). Перевірка змісту: номер
  має дослівно міститися в тексті звернення, інакше це вигадка;
* `item_mention` — товар словами клієнта (або `null`); артикул
  обирається окремим кроком із позицій замовлення;
* `reason_category` — пропозиція моделі, `unknown` дозволено; але вона
  обовʼязково спирається на `reason_quote` — дослівну цитату з тексту
  (перевіряється кодом). Без цитати категорія не приймається: так
  `defect` не з’являється там, де клієнт про дефект не писав;
* `quantity` — кількість, якщо названа;
* схема не просить нічого, чого не використовує наступний крок.
"""

import json
import re

from .jsoncheck import SchemaError, check

INTENTS = ["return", "cancel", "status", "other", "unclear", "multiple"]
REASONS = ["not_suitable", "defect", "wrong_item", "damaged", "unknown"]


def request_schema() -> dict:
    """JSON Schema результату розбору (для `response_format` і для перевірки)."""
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": INTENTS},
            "order_id": {"type": ["string", "null"]},
            "item_mention": {"type": ["string", "null"]},
            "quantity": {"type": ["integer", "null"]},
            "reason_category": {"type": "string", "enum": REASONS},
            "reason_quote": {"type": ["string", "null"]},
        },
        "required": ["intent", "order_id", "item_mention", "quantity",
                     "reason_category", "reason_quote"],
        "additionalProperties": False,
    }


def item_choice_schema(skus: list[str]) -> dict:
    """Схема вибору позиції: `enum` складено з артикулів замовлення, які
    повернув інструмент, плюс `none`. Модель не може обрати те, чого в
    замовленні немає."""
    return {
        "type": "object",
        "properties": {"sku": {"type": "string", "enum": list(skus) + ["none"]}},
        "required": ["sku"],
        "additionalProperties": False,
    }


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    return m.group(1) if m else raw


def validate(raw: str, text: str | None = None) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані для наступних
    кроків. `text` — оригінал звернення, для перевірки змісту. Якщо щось
    не так — `SchemaError` з поясненням (його ж можна показати моделі
    при повторі)."""
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise SchemaError(f"відповідь не є JSON: {exc}") from exc
    check(data, request_schema())

    for key in ("order_id", "item_mention", "reason_quote"):
        if isinstance(data[key], str):
            data[key] = data[key].strip() or None
    if data["order_id"] is not None:
        if not re.fullmatch(r"\d{3,8}", data["order_id"]):
            raise SchemaError(f"order_id {data['order_id']!r} не схожий на номер замовлення")
        if text is not None and data["order_id"] not in re.sub(r"[\s\-]", "", text) \
                and data["order_id"] not in text:
            raise SchemaError(f"номера {data['order_id']} у тексті звернення немає; "
                              "якщо клієнт його не називав, поверни null")
    if data["quantity"] is not None and not 1 <= data["quantity"] <= 99:
        raise SchemaError("quantity поза межами 1..99")
    if data["reason_category"] in (None, "unknown"):
        data["reason_category"] = "unknown"
    elif not data["reason_quote"]:
        raise SchemaError("для reason_category потрібна reason_quote — цитата з тексту; "
                          "якщо причини в тексті немає, постав unknown")
    if data["reason_quote"] is not None and text is not None \
            and _norm(data["reason_quote"]) not in _norm(text):
        raise SchemaError("reason_quote не є дослівною цитатою з тексту звернення")
    return data
