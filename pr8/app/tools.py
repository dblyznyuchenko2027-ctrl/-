"""Безпечний шар tool calling між моделлю та внутрішнім сервісом магазину."""

from dataclasses import dataclass
import json
import re
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from shop import service


@dataclass
class Context:
    customer_id: str


@dataclass
class ToolResult:
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


# ВАЖЛИВО: customer_id навмисно відсутній з усіх схем. Його джерело — ctx.
_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "Показати список замовлень увійденого клієнта. Використовуйте, коли клієнт питає про свої замовлення без конкретного номера. Не приймає ідентифікатор клієнта.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Показати стан конкретного замовлення. Використовуйте, коли клієнт назвав номер замовлення. Номер належить до сеансу клієнта: код перевірить власника до виконання.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "pattern": "^[0-9][0-9 ]{3,9}$", "description": "Номер замовлення, лише цифри; пробіли всередині допускаються і будуть нормалізовані."}
                },
                "required": ["order_id"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Знайти товари за назвою або словами з опису. Використовуйте, коли товар названо словами і потрібен його SKU для наступних операцій. Не вигадуйте SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "category": {"type": ["string", "null"]},
                    "max_price": {"type": ["string", "number", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Отримати картку конкретного товару за SKU. Використовуйте після пошуку, коли потрібні опис, ціна, вага або правила повернення.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string", "minLength": 1, "maxLength": 40}},
                "required": ["sku"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Перевірити наявність товару за SKU і дату очікуваного поповнення, якщо його немає.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string", "minLength": 1, "maxLength": 40}},
                "required": ["sku"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delivery_quote",
            "description": "Порахувати вартість і строк доставки товарів по Україні. Використовуйте лише після того, як відомі SKU товарів і кількість. method: branch — відділення, courier — кур'єр, pickup — самовивіз.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "minLength": 1},
                    "method": {"type": "string", "enum": ["branch", "courier", "pickup"]},
                    "items": {
                        "type": "array", "minItems": 1, "maxItems": 20,
                        "items": {"type": "object", "properties": {
                            "sku": {"type": "string", "minLength": 1},
                            "quantity": {"type": "integer", "minimum": 1, "maximum": 50},
                        }, "required": ["sku", "quantity"], "additionalProperties": False},
                    },
                },
                "required": ["city", "method", "items"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Створити заявку на повернення товару. Це операція зі зміною даних: викликайте лише коли клієнт прямо попросив оформити повернення і назвав замовлення, товар та фактичну причину. Не вигадуйте причину. Перед виконанням код перевірить власника замовлення і правила магазину.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "pattern": "^[0-9][0-9 ]{3,9}$"},
                    "sku": {"type": "string", "minLength": 1, "maxLength": 40},
                    "reason": {"type": "string", "enum": ["not_suitable", "defect", "wrong_item", "damaged"]},
                    "quantity": {"type": "integer", "minimum": 1, "maximum": 50},
                    "comment": {"type": "string", "maxLength": 500},
                },
                "required": ["order_id", "sku", "reason"], "additionalProperties": False,
            },
        },
    },
]


def specs() -> list[dict]:
    return _SPECS


def _normalize_order_id(value: str) -> str:
    # Допускаємо типове написання «№ 10 458» на рівні діалогу; модель зазвичай
    # уже передасть лише цифри, а пробіли не впливають на ідентифікатор.
    return re.sub(r"\s+", "", str(value)).strip()


def _safe_order(order: dict) -> dict:
    """Тільки поля, потрібні клієнтові; internal_note/payment не потрапляють у LLM."""
    delivery = order.get("delivery", {})
    return {
        "order_id": order["order_id"],
        "created_at": order["created_at"],
        "status": order["status"],
        "status_label": order.get("status_label"),
        "items": [
            {"sku": i["sku"], "name": i["name"], "quantity": i["quantity"], "price": i["price"]}
            for i in order.get("items", [])
        ],
        "total": order["total"],
        "delivery": {
            "method": delivery.get("method"), "carrier": delivery.get("carrier"),
            "city": delivery.get("city"), "point": delivery.get("point"),
            "tracking": delivery.get("tracking"), "delivered_at": delivery.get("delivered_at"),
            "cost": delivery.get("cost"),
        },
    }


def _safe_product(product: dict) -> dict:
    return {
        "sku": product["sku"], "name": product["name"], "category": product["category"],
        "price": product["price"], "weight_kg": product["weight_kg"],
        "warranty_months": product["warranty_months"], "returnable": product["returnable"],
        "bulky": product["bulky"], "description": product["description"],
    }


def _safe_return(record: dict) -> dict:
    return {k: record[k] for k in ("return_id", "order_id", "sku", "quantity", "reason", "created_at", "status", "next_steps") if k in record}


def _safe_result(name: str, value: Any) -> Any:
    if name == "get_order":
        return _safe_order(value)
    if name == "get_product":
        return _safe_product(value)
    if name == "search_products":
        return [{k: item[k] for k in ("sku", "name", "category", "price")} for item in value]
    if name == "get_stock":
        return {k: value.get(k) for k in ("sku", "available", "expected_restock")}
    if name == "delivery_quote":
        return {k: value.get(k) for k in ("city", "method", "order_total", "weight_kg", "cost", "days", "note")}
    if name == "list_orders":
        return [{k: item[k] for k in ("order_id", "created_at", "status", "status_label", "total", "items_count")} for item in value]
    if name == "create_return":
        return _safe_return(value)
    return value


def _schema_for(name: str) -> dict | None:
    for spec in _SPECS:
        if spec["function"]["name"] == name:
            return spec["function"]["parameters"]
    return None


def _owned_order(ctx: Context, order_id: str) -> dict:
    order = service.get_order(order_id)
    if str(order.get("customer_id")) != str(ctx.customer_id):
        raise PermissionError("замовлення не належить увійденому клієнтові")
    return order


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    try:
        schema = _schema_for(name)
        if schema is None:
            return ToolResult("rejected", reason="невідомий інструмент", content={"error": "Невідомий інструмент; виклик не виконано."})
        try:
            args = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            return ToolResult("rejected", reason="аргументи не є коректним JSON", content={"error": "Аргументи інструмента мають бути коректним JSON."})
        if not isinstance(args, dict):
            return ToolResult("rejected", reason="аргументи мають бути JSON-об'єктом", content={"error": "Аргументи інструмента мають бути JSON-об'єктом."})
        try:
            Draft202012Validator(schema).validate(args)
        except ValidationError as exc:
            return ToolResult("rejected", reason=f"аргументи не відповідають схемі: {exc.message}", content={"error": f"Аргументи не відповідають схемі: {exc.message}"})

        # Нормалізація номера після структурної перевірки.
        if name in {"get_order", "create_return"}:
            args["order_id"] = _normalize_order_id(args["order_id"])

        # Власник перевіряється кодом, не моделлю.
        if name == "list_orders":
            value = service.list_orders(ctx.customer_id)
        elif name == "get_order":
            value = _owned_order(ctx, args["order_id"])
        elif name == "search_products":
            value = service.search_products(**args)
        elif name == "get_product":
            value = service.get_product(args["sku"])
        elif name == "get_stock":
            value = service.get_stock(args["sku"])
        elif name == "delivery_quote":
            value = service.delivery_quote(**args)
        elif name == "create_return":
            _owned_order(ctx, args["order_id"])
            value = service.create_return(**args)
        else:
            return ToolResult("rejected", reason="інструмент не дозволений", content={"error": "Ця операція недоступна помічнику."})
        return ToolResult("ok", content=_safe_result(name, value), arguments=args)
    except PermissionError as exc:
        return ToolResult("rejected", reason=str(exc), content={"error": str(exc)})
    except service.ShopError as exc:
        return ToolResult("error", reason=str(exc), content={"error": {"code": exc.code, "message": str(exc)}})
    except Exception:
        # Внутрішні деталі/traceback не віддаємо моделі.
        return ToolResult("error", reason="внутрішня помилка інструмента", content={"error": {"code": "internal_error", "message": "Внутрішня помилка сервісу."}})
