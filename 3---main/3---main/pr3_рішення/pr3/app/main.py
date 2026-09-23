

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР3")
logger = logging.getLogger("app.main")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"

# Захист від абсурдно довгого вводу: не техобмеження моделі, а здоровий глузд —
# щоб один клієнт не міг випадково чи навмисно роздути токени й вартість запиту.
MAX_QUESTION_LENGTH = 2000


class Question(BaseModel):
    """Звернення користувача."""

    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        # Порожнє чи пробіли-only звернення відсікаємо на веб-рівні: це не
        # збій LLM, а некоректний ввід, і йому не місце в модулі llm.py.
        if not value or not value.strip():
            raise ValueError("Звернення не може бути порожнім.")
        if len(value) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Звернення задовге (максимум {MAX_QUESTION_LENGTH} символів). "
                "Скоротіть, будь ласка, текст."
            )
        return value


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель.

    Контекст — це дані застосунку, а не знання моделі. Він живе окремим
    файлом і передається в запит разом зі зверненням.
    """
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку зі зверненням."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """Повернути відповідь помічника у форматі JSON.

    Порожнє звернення відсікається валідацією Pydantic (422 — стандартна
    поведінка FastAPI для некоректного тіла запиту, тож окремо її тут не
    ловимо). Далі — збої самого виклику моделі, кожен зі своїм статусом:

    - LLMTimeoutError            -> 504 Gateway Timeout
    - LLMRateLimitError          -> 429 Too Many Requests
    - LLMServiceUnavailableError -> 503 Service Unavailable
    - LLMAuthError               -> 500 Internal Server Error (це проблема
      конфігурації застосунку, а не клієнта, тому 5xx, а не 4xx)

    Клієнт бачить лише коротке, зрозуміле повідомлення (`user_message`).
    Технічний текст провайдера (`technical_detail`) іде тільки в журнал —
    там можуть бути деталі помилки, яких зовнішньому користувачеві
    показувати не варто.
    """
    context = load_context()
    try:
        return llm.ask(payload.question, context)
    except llm.LLMTimeoutError as exc:
        logger.warning("Таймаут виклику моделі: %s", exc.technical_detail)
        raise HTTPException(status_code=504, detail=exc.user_message) from exc
    except llm.LLMRateLimitError as exc:
        logger.warning("Перевищено ліміт запитів: %s", exc.technical_detail)
        raise HTTPException(status_code=429, detail=exc.user_message) from exc
    except llm.LLMServiceUnavailableError as exc:
        logger.error("Сервіс моделі недоступний: %s", exc.technical_detail)
        raise HTTPException(status_code=503, detail=exc.user_message) from exc
    except llm.LLMAuthError as exc:
        logger.error("Помилка автентифікації до моделі: %s", exc.technical_detail)
        raise HTTPException(status_code=500, detail=exc.user_message) from exc
    except llm.LLMError as exc:
        logger.error("Неочікувана помилка моделі: %s", exc.technical_detail)
        raise HTTPException(status_code=500, detail=exc.user_message) from exc
