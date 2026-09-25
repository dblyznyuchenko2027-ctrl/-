"""Контракт відповіді моделі: що саме вона має повернути і як це
перевіряється.

Окремий модуль, як у ПР4, і з тієї самої причини: схема — домовленість
між моделлю та рештою застосунку, і вона не залежить від провайдера.
`app/rag.py` працює лише з тим, що пройшло перевірку тут.

Що потрібно решті застосунку від відповіді (розділ 2 практичної роботи):

* текст відповіді для клієнта;
* на які фрагменти контексту відповідь спирається — номери з позначок,
  які ви поставили в `retrieval.build_context`;
* чи знайдено відповідь у наданих фрагментах, чи їх для відповіді не
  вистачило.

Як назвати поля, які з них обовʼязкові, чи потрібні ще якісь — ваше
рішення. Як і те, чим описати схему: JSON Schema вручну або
pydantic-модель. Реалізацію з ПР4 можна взяти за основу.

Сторінка каркаса бере текст відповіді з поля `answer` результату
`rag.answer`, а не з відповіді моделі напряму — тож назви полів у схемі
ні до чого не привʼязані, крім вашого коду.
"""


import json

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Текст відповіді для клієнта українською мовою.",
        },
        "found": {
            "type": "boolean",
            "description": "Чи знайдено відповідь у наданих фрагментах контексту.",
        },
        "sources": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": "Номери фрагментів контексту (у квадратних дужках), на які спирається відповідь.",
        },
    },
    "required": ["answer", "found", "sources"],
    "additionalProperties": False,
}


class SchemaError(ValueError):
    """Відповідь моделі не є JSON або не відповідає схемі."""


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі.

    Ця сама схема передається моделі як опис очікуваного результату
    (текстом в інструкції — `app/llm.py` підставляє її в системний
    промпт, а `response_format={"type": "json_object"}` вимагає від
    провайдера валідний JSON) і використовується тут-таки для перевірки
    того, що повернулося.
    """
    return _SCHEMA


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна
    довіряти структурно.

    Якщо текст не є JSON або не проходить схему — підняти `SchemaError` з
    поясненням, що саме не так; далі це ловить `app/llm.py` й вирішує,
    повторювати запит чи здатися.

    Схема перевіряє лише форму (типи полів, обовʼязковість). Чи існують
    названі моделлю номери джерел серед тих, що їй показали, і чи не
    «знайшла» вона відповідь, не пославшись ні на що, — перевірка змісту,
    і її місце в `app/rag.py`, а не тут: тут ми ще не знаємо, скільки
    фрагментів було в контексті.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"відповідь моделі не є коректним JSON: {exc}") from exc

    try:
        import jsonschema

        jsonschema.validate(data, _SCHEMA)
    except ImportError:
        if not isinstance(data, dict):
            raise SchemaError("відповідь моделі — не обʼєкт JSON")
        for field_name in ("answer", "found", "sources"):
            if field_name not in data:
                raise SchemaError(f"у відповіді немає обовʼязкового поля '{field_name}'")
        if not isinstance(data["answer"], str):
            raise SchemaError("поле 'answer' має бути рядком")
        if not isinstance(data["found"], bool):
            raise SchemaError("поле 'found' має бути булевим")
        if not isinstance(data["sources"], list) or not all(
            isinstance(x, int) for x in data["sources"]
        ):
            raise SchemaError("поле 'sources' має бути списком цілих чисел")
    except Exception as exc:
        raise SchemaError(f"відповідь не відповідає схемі: {exc}") from exc

    return data
