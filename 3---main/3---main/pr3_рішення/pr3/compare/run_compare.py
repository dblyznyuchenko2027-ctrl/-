"""Прогнати набір звернень із requests.json через застосунок і зберегти результат.

Використання:
    1. Запустіть застосунок в одному терміналі:  uvicorn app.main:app --reload
    2. Скопіюйте requests.example.json у requests.json і допишіть свої звернення.
    3. В іншому терміналі (з папки pr3) запустіть перший прогін:

        python compare/run_compare.py --label baseline

    4. Зупиніть застосунок, змініть ОДНЕ налаштування в .env (наприклад,
       LLM_TEMPERATURE з 0.2 на 0.9), перезапустіть застосунок і зробіть
       другий прогін під іншою міткою:

        python compare/run_compare.py --label high-temp

    Кожен прогін дає файл compare/results_<label>.json із відповіддю,
    часом виконання й кількістю спроб для кожного звернення — саме на
    цих файлах пишуться висновки у findings.md.

    Порівняти стабільність можна прогнавши той самий набір під тією самою
    міткою двічі й порівнявши, наскільки різняться відповіді на однакове
    звернення.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REQUESTS_FILE = Path(__file__).parent / "requests.json"
APP_URL = "http://127.0.0.1:8000/api/ask"


def load_requests() -> list[dict]:
    if not REQUESTS_FILE.exists():
        raise SystemExit(
            f"Не знайдено {REQUESTS_FILE}. Скопіюйте requests.example.json "
            "у requests.json і допишіть свої звернення."
        )
    data = json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))
    return data["звернення"]


def call_app(question: str) -> dict:
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        APP_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return {"status": resp.status, "http_elapsed": round(time.perf_counter() - started, 3), **payload}
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return {
            "status": exc.code,
            "http_elapsed": round(time.perf_counter() - started, 3),
            "error": payload.get("detail", str(exc)),
        }
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Не вдалося з'єднатися з {APP_URL}: {exc}. "
            "Застосунок запущено? (uvicorn app.main:app --reload)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Мітка конфігурації, напр. baseline")
    args = parser.parse_args()

    requests_list = load_requests()
    results = []
    print(f"Прогін «{args.label}»: {len(requests_list)} звернень\n")

    for item in requests_list:
        kind = item.get("вид", "?")
        text = item["текст"]
        print(f"[{kind}] {text[:60]}...")
        result = call_app(text)
        results.append({"вид": kind, "текст": text, **result})
        status = result.get("status")
        if status == 200:
            print(f"    -> {result.get('elapsed', '?')} с, {len(result.get('answer', ''))} символів\n")
        else:
            print(f"    -> помилка {status}: {result.get('error')}\n")

    out_file = Path(__file__).parent / f"results_{args.label}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено: {out_file}")


if __name__ == "__main__":
    main()
