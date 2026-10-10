"""Мінімальна перевірка за підмножиною JSON Schema: type (у т.ч. список),
enum, required, properties, additionalProperties=false, minimum/maximum,
maxLength/minLength, pattern, items. Без зовнішніх залежностей, щоб
поведінка не залежала від версії бібліотеки. Користуються `tools.py`
(аргументи інструментів) і `schema.py` (результат розбору)."""

import re


class SchemaError(ValueError):
    """Значення не відповідає схемі; текст каже, де саме."""


_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


def check(value, schema: dict, path: str = "$") -> None:
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else types
        if not any(_TYPES[t](value) for t in types):
            raise SchemaError(f"{path}: очікується {'/'.join(types)}, отримано {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} не входить до {schema['enum']}")
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaError(f"{path}: довше за {schema['maxLength']} символів")
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaError(f"{path}: коротше за {schema['minLength']} символів")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise SchemaError(f"{path}: {value!r} не відповідає формату {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaError(f"{path}: менше за {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaError(f"{path}: більше за {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(f"{path}: немає обовʼязкового поля {key!r}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                raise SchemaError(f"{path}: зайві поля {sorted(extra)}")
        for key, sub in props.items():
            if key in value:
                check(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            check(item, schema["items"], f"{path}[{i}]")
