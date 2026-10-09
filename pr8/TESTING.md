# Перевірка ПР8

## Виконано в середовищі підготовки

- `python -m compileall -q app shop eval/run_eval.py` — OK.
- 16 локальних тестів tool-layer — **16/16 PASS**.
- Перевірено: власник замовлення, відмова для чужого замовлення, нормалізація номера, невалідний JSON, зайві поля, відсутність адмін-інструментів, пошук/товар/склад, доставка, повернення, політика повернення, відсутність витоку `internal_note` та `card_last4`.
- Імітований цикл `модель → tool → модель` — OK.

## Що треба виконати локально з реальним API

```powershell
cd pr8
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# вписати LLM_API_KEY
python check_env.py
uvicorn app.main:app --reload
```

Потім відкрити `http://127.0.0.1:8000` і прогнати `eval/run_eval.py`.

Для експерименту змінити лише:

```text
TOOL_MAX_ROUNDS=2
```

Повернути `TOOL_MAX_ROUNDS=3` і окремо перевірити:

```text
SHOP_FAILURE_RATE=0.3
```

Перед кожним eval-прогоном виконується `service.reset()`.
