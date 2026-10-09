"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти.

Цей файл не знає ані як шукаються фрагменти, ані як збирається контекст,
ані якою моделлю й за якою інструкцією отримано відповідь — усе це
лишається в `app/retrieval.py`, `app/llm.py`, `app/schema.py` і
поєднується в `app/rag.py`. Тут вирішується інше: що застосунок приймає
від сторінки, що віддає їй і з яким HTTP-статусом.

Індекс будується заздалегідь командою `python ingest.py` (з папки pr6),
а тут лише читається при старті — як у ПР5.

Запуск із папки pr6:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import index, keyword, llm, rag
from . import retrieval as retrieval_module
from .schema import SchemaError

app = FastAPI(title="Помічник за базою знань — ПР6")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Те, що надсилає сторінка.

    `filters` — умови на метадані, які обрав користувач, наприклад
    `{"product": "Вега S"}`; порожній словник — без фільтрів. Те, що
    користувач обирати не має (аудиторія документа), сюди не входить —
    це рішення коду, а не сторінки.
    """

    question: str
    filters: dict = {}


def hit_to_dict(hit: index.Hit) -> dict:
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


def answer_to_dict(result: rag.Answer) -> dict:
    """Перетворити результат конвеєра на те, що піде на сторінку."""
    return {
        "answer": result.text,
        "found": result.found,
        "sources": [
            {"ref": s.ref, "score": s.score, "source": s.chunk.source,
             "metadata": s.chunk.metadata, "text": s.chunk.text}
            for s in result.sources
        ],
        "retrieved": [hit_to_dict(h) for h in result.retrieved],
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Прочитати збудований індекс і зібрати індекс за словами з тих
    самих фрагментів — як у ПР5.

    Без індексу застосунок усе одно стартує: сторінка має відкритися й
    пояснити, що робити.
    """
    app.state.index = None
    app.state.keyword_index = None
    try:
        app.state.index = index.load()
    except Exception as exc:
        print(f"Індекс не завантажено: {type(exc).__name__}: {exc}")
        return
    app.state.keyword_index = keyword.build(app.state.index.chunks)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку помічника."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Стан індексу: чи збудовано, скільки фрагментів, якою моделлю."""
    idx = app.state.index
    if idx is None:
        return {"ready": False, "hint": "індекс не збудовано — виконайте python ingest.py"}
    sources = {chunk.source for chunk in idx.chunks}
    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    """Відповісти на питання й повернути відповідь із джерелами.

    Сторінка очікує обʼєкт із полями `answer`, `found`, `sources`,
    `retrieved`, `model`, `elapsed`, `usage` — див. `answer_to_dict`.

    Збої, оброблені тут явно, з відповідним HTTP-статусом:

    * порожнє питання — 400, до звернення до індексу чи моделі;
    * індекс не збудовано (`app.state.index is None`) — 503, з підказкою
      виконати `python ingest.py`;
    * невідомий фільтр (усе, крім `product`, зокрема спроба передати
      `audience`/`status` напряму) — `retrieval.FilterError` → 400;
    * збої мовної моделі (тайм-аут, ліміт, невірний ключ, недоступність,
      відповідь, що не пройшла перевірку схемою навіть після повтору) —
      `llm.LLMError` → 502 (застосунок працює, зовнішній сервіс — ні);
    * будь-що інше неочікуване — 500, але з поясненням типу помилки, а не
      мовчазним падінням.
    """
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Питання не може бути порожнім.")

    if app.state.index is None:
        raise HTTPException(
            status_code=503,
            detail="Індекс не збудовано. Виконайте `python ingest.py` з папки pr6 і перезапустіть застосунок.",
        )

    try:
        result = rag.answer(
            question,
            app.state.index,
            app.state.keyword_index,
            filters=payload.filters or None,
        )
    except retrieval_module.FilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (llm.LLMError, SchemaError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Неочікувана помилка: {type(exc).__name__}: {exc}"
        ) from exc

    return answer_to_dict(result)
