"""Прогін набору питань через увесь конвеєр (без веб-рівня).

Запуск із папки pr6:

    python eval/run_eval.py                       # прогін questions.json
    python eval/run_eval.py --out eval/run2.json   # інша назва файлу результатів
    python eval/run_eval.py --label "3 фрагменти замість 4"

Скрипт не оцінює правильність сам — оцінка фактів і ґрунтованості потребує
людини (або окремого судді-моделі на більшому наборі). Він виконує механічну
частину: проганяє кожне питання через `rag.answer`, зберігає повну «памʼять»
кожного прогону (що бачила модель, що вона відповіла, скільки це коштувало) і
рахує ту частину перевірки, яку можна порахувати автоматично — чи очікуваний
документ потрапив у контекст, чи збігається «очікується_відповідь» із тим, що
сталося насправді, чи всі номери джерел у відповіді дійсні.

Двічі проганяти весь набір вручну через сторінку — дорого й повільно (кожне
питання — виклик моделі); цей скрипт заощаджує і те, і те, і дає однаковий,
відтворюваний журнал для порівняння "до" і "після" однієї зміни (крок 4 в
eval/README.md).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import index as index_module
from app import keyword
from app import rag
from app.retrieval import FilterError

QUESTIONS_PATH = Path(__file__).parent / "questions.json"


def load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["питання"]


def run_one(question_item: dict, search_index, keyword_index) -> dict:
    question = question_item["питання"]
    filters = question_item.get("фільтр")
    started = time.perf_counter()
    try:
        result = rag.answer(question, search_index, keyword_index, filters=filters)
        error = None
    except FilterError as exc:
        result = None
        error = f"FilterError: {exc}"
    except Exception as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    total_elapsed = time.perf_counter() - started

    expected_docs = set(question_item.get("очікувані_документи", []))
    expected_answer = bool(question_item.get("очікується_відповідь", True))

    if result is None:
        return {
            "вид": question_item.get("вид"),
            "питання": question,
            "помилка": error,
            "очікується_відповідь": expected_answer,
            "фактично_відповіло": None,
        }

    retrieved_sources = {h.chunk.source for h in result.retrieved}
    cited_sources = {s.chunk.source for s in result.sources}
    expected_in_context = sorted(expected_docs & retrieved_sources)
    expected_missing = sorted(expected_docs - retrieved_sources)
    unexpected_in_context = sorted(retrieved_sources - expected_docs)

    return {
        "вид": question_item.get("вид"),
        "питання": question,
        "фільтр": filters,
        "очікувані_документи": sorted(expected_docs),
        "очікується_відповідь": expected_answer,
        "фактично_відповіло": result.found,
        "відповідність_очікуванню": result.found == expected_answer,
        "відповідь": result.text,
        "джерела_відповіді": sorted(cited_sources),
        "документи_в_контексті": sorted(retrieved_sources),
        "очікувані_документи_в_контексті": expected_in_context,
        "очікувані_документи_відсутні_в_контексті": expected_missing,
        "неочікувані_документи_в_контексті": unexpected_in_context,
        "фрагменти_в_контексті": [
            {
                "source": h.chunk.source,
                "heading": h.chunk.metadata.get("heading"),
                "score": h.score,
                "text": h.chunk.text,
            }
            for h in result.retrieved
        ],
        "модель": result.model,
        "час_пошуку_с": result.elapsed.get("retrieval"),
        "час_генерації_с": result.elapsed.get("generation"),
        "токени": result.usage,
        "час_прогону_питання_с": total_elapsed,
    }


def summarize(results: list[dict]) -> dict:
    completed = [r for r in results if r.get("помилка") is None]
    n = len(completed)
    if n == 0:
        return {"питань": len(results), "завершено": 0, "примітка": "жодне питання не завершилося без збою"}

    context_hit = sum(
        1
        for r in completed
        if not r["очікувані_документи"] or not r["очікувані_документи_відсутні_в_контексті"]
    )
    expectation_match = sum(1 for r in completed if r["відповідність_очікуванню"])
    false_positive_refusal = sum(
        1 for r in completed if r["очікується_відповідь"] and not r["фактично_відповіло"]
    )
    false_positive_answer = sum(
        1 for r in completed if not r["очікується_відповідь"] and r["фактично_відповіло"]
    )
    total_tokens = sum(
        (r["токени"] or {}).get("total_tokens") or 0 for r in completed if r.get("токени")
    )
    avg_gen_time = sum(r["час_генерації_с"] or 0 for r in completed) / n

    return {
        "питань_у_наборі": len(results),
        "завершено_без_збою": n,
        "збоїв": len(results) - n,
        "очікуваний_документ_у_контексті__частка": round(context_hit / n, 2),
        "відповідність_очікуваній_поведінці__частка": round(expectation_match / n, 2),
        "відповів_дарма__мало_бути_відмова": false_positive_answer,
        "відмовився_дарма__мала_бути_відповідь": false_positive_refusal,
        "токенів_разом": total_tokens,
        "середній_час_генерації_с": round(avg_gen_time, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(QUESTIONS_PATH), help="шлях до questions.json")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results.json"), help="куди зберегти результати")
    parser.add_argument("--label", default="", help="підпис прогону (наприклад, яку саме одну зміну зроблено)")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Немає файлу {questions_path}. Скопіюйте questions.example.json у questions.json (або вкажіть --questions).")
        return 1

    try:
        search_index = index_module.load()
    except FileNotFoundError as exc:
        print(f"Індекс не знайдено: {exc}")
        return 1

    keyword_index = keyword.build(search_index.chunks)
    questions = load_questions(questions_path)

    print(f"Прогін {len(questions)} питань{f' ({args.label})' if args.label else ''}…")
    results = []
    for i, item in enumerate(questions, start=1):
        print(f"  [{i}/{len(questions)}] {item['питання'][:60]}…")
        results.append(run_one(item, search_index, keyword_index))

    summary = summarize(results)
    payload = {"label": args.label, "summary": summary, "results": results}

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nПідсумок:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nПовні результати (з контекстом, який бачила модель) — у {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
