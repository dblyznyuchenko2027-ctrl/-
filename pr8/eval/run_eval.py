"""Прогін eval/questions.json з журналами.
Запуск: python eval/run_eval.py
Потрібен робочий .env і доступ до моделі.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.assistant import answer
from shop import service

questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
out = []
for q in questions:
    service.reset()
    result = answer(q["question"], q["customer"])
    out.append({
        "id": q["id"], "kind": q["kind"], "customer": q["customer"],
        "question": q["question"], "expect_tools": q["expect_tools"],
        "answer": result.text, "rounds": result.rounds, "stopped": result.stopped,
        "calls": [c.__dict__ for c in result.calls], "model": result.model,
        "elapsed": result.elapsed, "usage": result.usage,
    })

path = ROOT / "eval" / "results.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Збережено {len(out)} результатів у {path}")
