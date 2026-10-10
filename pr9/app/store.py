"""Сховище стану звернень: зберегти, прочитати, перелічити.

Один JSON-файл на звернення в `RUNS_DIR`. Запис атомарний: дані
пишуться у тимчасовий файл поруч, скидаються на диск (`fsync`) і
підмінюють цільовий через `os.replace` — після збою лишається або
стара, або нова версія, але не напівзаписана. Модуль не знає ні про
модель, ні про магазин, ні про кроки: зберігає `Run` і повертає таким,
яким зберіг.

Одночасні зміни того самого звернення — справа workflow (блокування за
`run_id` і перевірка статусу), а не сховища: воно лише гарантує
цілісність файлу.
"""

import json
import os
import re
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))
_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_write_lock = threading.Lock()


def _path(run_id: str) -> Path:
    if not _ID.match(run_id or ""):
        raise ValueError(f"некоректний номер звернення {run_id!r}")
    return RUNS_DIR / f"{run_id}.json"


def _from_dict(d: dict) -> Run:
    d = dict(d)
    d["steps"] = [Step(**s) for s in d.get("steps", [])]
    allowed = Run.__dataclass_fields__.keys()
    return Run(**{k: v for k, v in d.items() if k in allowed})


def save(run: Run) -> None:
    """Зберегти стан, замінивши попередній (атомарно)."""
    path = _path(run.run_id)
    payload = json.dumps(asdict(run), ensure_ascii=False, indent=2)
    with _write_lock:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=RUNS_DIR, prefix=f".{run.run_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def load(run_id: str) -> Run | None:
    """Прочитати стан; `None`, якщо такого немає (або файл нечитний)."""
    try:
        path = _path(run_id)
    except ValueError:
        return None
    try:
        return _from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def list_runs(status: str | None = None) -> list[Run]:
    """Усі звернення (за потреби лише з певним статусом); нові першими."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in RUNS_DIR.glob("*.json"):
        run = load(p.stem)
        if run is not None and (status is None or run.status == status):
            runs.append(run)
    runs.sort(key=lambda r: (r.created_at or "", r.run_id), reverse=True)
    return runs
