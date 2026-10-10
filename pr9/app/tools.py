"""Інструменти: єдине місце, яке знає про сервіс магазину (`shop/service.py`).

Ті самі інструменти викликає і агент (за рішенням моделі), і workflow
(за рішенням коду). В обох випадках виклик іде через `call` і дає
`ToolResult`; перевірка схеми, власника й відбір полів відбуваються
тут, а не в тих, хто викликає.

Рішення:

* клієнта визначає `Context` із сеансу; моделі й аргументам його не
  віддано, а чуже замовлення відповідає так само, як неіснуюче;
* моделі й workflow повертаються лише потрібні поля: без службової
  примітки, даних картки й довільного тексту постачальника з картки
  товару (опис — найлегший шлях для чужої «інструкції» потрапити в
  контекст);
* кожен виклик може бути обмежений переліком `allowed` — набором
  інструментів кроку; перевіряє код, а не інструкція;
* читання при `unavailable` повторюються (3 спроби); операції зі
  змінами — ні: невідомо, чи вони відбулися, тому звіряється той, хто
  викликає (workflow через `list_returns` / стан замовлення);
* жоден виняток не виходить назовні.
"""

import json
import re
import time
from dataclasses import dataclass

from shop import service

from .jsoncheck import SchemaError, check


@dataclass
class Context:
    """Сеанс: клієнт, який увійшов. Модель цього поля не бачить."""

    customer_id: str


@dataclass
class ToolResult:
    """Підсумок виклику.

    `status`: `ok` | `rejected` (не пройшло перевірок, сервіс не
    викликано) | `error` (сервіс відмовив або не відповів). `code` —
    `not_found`, `policy`, `invalid_request`, `unavailable`,
    `not_allowed`, `unknown_tool`, `bad_arguments`, `internal`.
    `content` — що піде моделі/кодові; `reason` — пояснення для журналу.
    """

    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None
    code: str | None = None
    attempts: int = 1
    elapsed: float = 0.0


READ_RETRIES = 3
RETURN_REASONS = ["not_suitable", "defect", "wrong_item", "damaged"]
_ID = {"type": "string", "pattern": r"^\d{3,8}$"}
_SKU = {"type": "string", "pattern": r"^[A-Za-z0-9\-]{2,20}$"}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


# Назва -> (опис для моделі, схема, чи змінює дані)
_TOOLS: dict[str, tuple[str, dict, bool]] = {
    "list_orders": ("Перелік замовлень поточного клієнта: номер, дата, статус, сума.",
                    _obj({}, []), False),
    "get_order": ("Стан замовлення поточного клієнта: позиції, статус, доставка, номер "
                  "відправлення.", _obj({"order_id": _ID}, ["order_id"]), False),
    "get_product": ("Картка товару: назва, категорія, чи повертається товар належної "
                    "якості, гарантія.", _obj({"sku": _SKU}, ["sku"]), False),
    "get_stock": ("Наявність товару на складі.", _obj({"sku": _SKU}, ["sku"]), False),
    "search_products": ("Пошук товарів у каталозі за словами.",
                        _obj({"query": {"type": "string", "maxLength": 100}}, ["query"]), False),
    "list_returns": ("Заявки на повернення за замовленням поточного клієнта.",
                     _obj({"order_id": _ID}, ["order_id"]), False),
    "create_return": ("Створити заявку на повернення товару із замовлення поточного клієнта.",
                      _obj({"order_id": _ID, "sku": _SKU,
                            "reason": {"type": "string", "enum": RETURN_REASONS},
                            "quantity": {"type": "integer", "minimum": 1, "maximum": 99},
                            "comment": {"type": "string", "maxLength": 300}},
                           ["order_id", "sku", "reason"]), True),
    "cancel_order": ("Скасувати замовлення поточного клієнта (лише у статусі «Готується»).",
                     _obj({"order_id": _ID}, ["order_id"]), True),
}

MUTATING = {name for name, (_, _, m) in _TOOLS.items() if m}


def specs(names: set[str] | None = None) -> list[dict]:
    """Описи інструментів для API моделі (можна обмежити переліком)."""
    return [{"type": "function",
             "function": {"name": n, "description": d, "parameters": s}}
            for n, (d, s, _) in _TOOLS.items() if names is None or n in names]


# ------------------------------------------------------------ відбір полів

def _order_view(o: dict) -> dict:
    d = o.get("delivery", {})
    return {
        "order_id": o["order_id"], "created_at": o["created_at"],
        "status": o["status"], "status_label": o["status_label"], "total": o["total"],
        "items": [{"sku": i["sku"], "name": i["name"], "quantity": i["quantity"]}
                  for i in o["items"]],
        "delivery": {"method": d.get("method"), "city": d.get("city"),
                     "point": d.get("point"), "tracking": d.get("tracking"),
                     "delivered_at": d.get("delivered_at")},
    }


def _product_view(p: dict) -> dict:
    return {"sku": p["sku"], "name": p["name"], "category": p["category"],
            "returnable": p["returnable"], "warranty_months": p["warranty_months"]}


def _return_view(r: dict) -> dict:
    return {k: r[k] for k in ("return_id", "order_id", "sku", "quantity", "reason",
                              "comment", "created_at", "status", "next_steps")}


def _own_order(ctx: Context, order_id: str) -> dict:
    """Замовлення клієнта з сеансу; чуже — як неіснуюче."""
    order = service.get_order(order_id)
    if order["customer_id"] != ctx.customer_id:
        raise service.NotFound("замовлення не знайдено")
    return order


def _run(name: str, args: dict, ctx: Context):
    if name == "list_orders":
        return [{k: o[k] for k in ("order_id", "created_at", "status", "status_label",
                                   "total", "items_count")}
                for o in service.list_orders(ctx.customer_id)]
    if name == "get_order":
        return _order_view(_own_order(ctx, args["order_id"]))
    if name == "get_product":
        return _product_view(service.get_product(args["sku"]))
    if name == "get_stock":
        return service.get_stock(args["sku"])
    if name == "search_products":
        return service.search_products(args["query"], limit=5)
    if name == "list_returns":
        _own_order(ctx, args["order_id"])
        return [_return_view(r) for r in service.list_returns(args["order_id"])]
    if name == "create_return":
        order = _own_order(ctx, args["order_id"])
        return _return_view(service.create_return(
            order["order_id"], args["sku"], args["reason"], args.get("quantity", 1),
            args.get("comment", "")))
    if name == "cancel_order":
        _own_order(ctx, args["order_id"])
        return _order_view(service.cancel_order(args["order_id"]))
    raise AssertionError(name)


# ---------------------------------------------------------------- виклик

def call(name: str, raw_arguments: str, ctx: Context,
         allowed: set[str] | None = None) -> ToolResult:
    """Виконати виклик. Ні назві, ні аргументам не вірити наперед.

    `allowed` — інструменти, дозволені кроку, що викликає; `None` — усі.
    """
    started = time.perf_counter()
    args = None

    def done(res: ToolResult) -> ToolResult:
        res.elapsed = round(time.perf_counter() - started, 4)
        return res

    try:
        if name not in _TOOLS:
            return done(ToolResult("rejected", None, f"невідомий інструмент {name!r}",
                                   code="unknown_tool"))
        if allowed is not None and name not in allowed:
            return done(ToolResult("rejected", None, f"інструмент {name} недоступний цьому кроку",
                                   code="not_allowed"))
        try:
            args = json.loads(raw_arguments) if raw_arguments and raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return done(ToolResult("rejected", None, f"аргументи не є JSON: {exc}",
                                   code="bad_arguments"))
        try:
            check(args, _TOOLS[name][1])
        except SchemaError as exc:
            return done(ToolResult("rejected", None, f"аргументи не пройшли схему: {exc}",
                                   arguments=args if isinstance(args, dict) else None,
                                   code="bad_arguments"))
        mutating = _TOOLS[name][2]
        attempts = 0
        while True:
            attempts += 1
            try:
                return done(ToolResult("ok", _run(name, args, ctx), arguments=args,
                                       attempts=attempts))
            except service.Unavailable as exc:
                if mutating or attempts >= READ_RETRIES:
                    return done(ToolResult("error", {"error": "unavailable"}, str(exc),
                                           args, "unavailable", attempts))
                time.sleep(0.05 * attempts)
    except service.ShopError as exc:
        text = re.sub(r"\s+", " ", str(exc))
        return done(ToolResult("error", {"error": exc.code, "message": text}, text,
                               args if isinstance(args, dict) else None, exc.code))
    except Exception as exc:  # noqa: BLE001 — назовні не виходить нічого
        return done(ToolResult("error", {"error": "internal"}, f"{type(exc).__name__}: {exc}",
                               None, "internal"))


# --------------------------------------------- допоміжне (не для моделі)

def customer_exists(customer_id: str) -> bool:
    """Чи є такий клієнт. Для перевірки входу; не звертається до
    «ненадійної» частини сервісу, тож не дає хибного «невідомий клієнт»
    при збої."""
    return any(c["customer_id"] == customer_id for c in service.list_customers())


def today() -> str:
    """Сьогоднішня дата сервісу, ISO. Потрібна правилам строків."""
    return service.today().isoformat()
