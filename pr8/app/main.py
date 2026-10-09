"""Веб-рівень застосунку ПР8."""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from shop import service

from . import assistant, llm, tools

app = FastAPI(title="Помічник клієнта — ПР8")
INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


def answer_to_dict(result: assistant.Answer) -> dict:
    return {
        "answer": result.text,
        "calls": [asdict(call) for call in result.calls],
        "rounds": result.rounds,
        "stopped": result.stopped,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    return service.list_customers()


@app.get("/api/tools")
def api_tools() -> list[dict]:
    return tools.specs()


@app.post("/api/reset")
def api_reset() -> dict:
    service.reset()
    return {"ok": True}


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    question = payload.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Порожнє питання."})
    customer_ids = {c["customer_id"] for c in service.list_customers()}
    if payload.customer_id not in customer_ids:
        return JSONResponse(status_code=400, content={"error": "Невідомий клієнт."})
    try:
        result = assistant.answer(question, payload.customer_id)
        return answer_to_dict(result)
    except Exception:
        # Не віддаємо трасування, ключі або внутрішні винятки клієнтові.
        return JSONResponse(status_code=500, content={"error": "Не вдалося обробити запит. Спробуйте ще раз."})
