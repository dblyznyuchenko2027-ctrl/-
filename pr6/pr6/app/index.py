"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих.

Цей модуль — той самий, що в ПР5, і тут він лишений заготовкою навмисно.
Замініть файл своєю реалізацією з `pr5/app/` цілком: у ПР6 вона не
змінюється, а лише використовується. Якщо в ПР5 ви перейменували
функції або поля — узгодьте з ними виклики в нових модулях.

Індекс — це вектори всіх фрагментів, самі фрагменти з метаданими й назва
моделі, якою вектори отримано. Він будується окремою командою
(`ingest.py`) і зберігається на диску, а застосунок при старті лише
читає його: перераховувати ембедінги колекції на кожен запуск — марна
трата часу, а на кожен запит — тим паче.

Веб-рівень (`app/main.py`) звертається сюди з вектором запиту й отримує
список влучень із оцінкою схожості. Звідки береться вектор — не справа
індексу; що показувати клієнтові — не його справа теж.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яку міру схожості взяти й що зробити з векторами перед порівнянням;
* як шукати: перебір усіх векторів (для сотень фрагментів — цілком
  доречно) чи бібліотека наближеного пошуку;
* у якому вигляді зберігати індекс на диску і що обовʼязково покласти
  поруч із векторами, щоб потім не переплутати, чиї вони;
* де застосовувати фільтри за метаданими — до ранжування чи після — і що
  станеться з top-k у кожному з варіантів;
* що робити з кількома фрагментами одного документа у видачі;
* чи потрібен поріг схожості й звідки взяти його значення.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті.

    `vectors` — масив (кількість фрагментів × розмірність); рядок i
    відповідає `chunks[i]`. `model_name` — модель, якою отримано вектори:
    без неї індекс, збудований однією моделлю, мовчки шукатиме векторами
    іншої.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів.

    Перевіряє, що векторів стільки ж, скільки фрагментів (інакше рядок i
    вектора мовчки відповідав би не тому фрагменту), і нормалізує вектори
    — схожість рахуємо косинусну, а на нормалізованих векторах вона
    зводиться до звичайного скалярного добутку.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"кількість фрагментів ({len(chunks)}) не збігається з кількістю "
            f"векторів ({vectors.shape[0]})"
        )
    return SearchIndex(chunks=chunks, vectors=_normalize(vectors), model_name=model_name)


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск: вектори — `numpy` (`vectors.npy`),
    фрагменти з метаданими і назва моделі — JSON (`chunks.json`). Назва
    моделі зберігається обовʼязково: без неї індекс, збудований однією
    моделлю, мовчки шукав би векторами іншої.
    """
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", index.vectors)
    payload = {
        "model_name": index.model_name,
        "chunks": [
            {"text": c.text, "source": c.source, "metadata": c.metadata}
            for c in index.chunks
        ],
    }
    (path / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска.

    Якщо індексу немає — підняти зрозумілу помилку: веб-рівень має
    сказати користувачеві «індекс не збудовано», а не впасти з
    `FileNotFoundError` десь усередині.
    """
    vectors_path = path / "vectors.npy"
    chunks_path = path / "chunks.json"
    if not vectors_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"індекс не знайдено у {path} — виконайте `python ingest.py` з папки pr6"
        )
    vectors = np.load(vectors_path)
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [
        Chunk(text=c["text"], source=c["source"], metadata=c["metadata"])
        for c in payload["chunks"]
    ]
    if len(chunks) != vectors.shape[0]:
        raise ValueError("індекс пошкоджено: кількість фрагментів і векторів не збігається")
    return SearchIndex(chunks=chunks, vectors=vectors, model_name=payload["model_name"])


def _matches(metadata: dict, filters: dict) -> bool:
    """Перевірити, чи фрагмент задовольняє всі умови фільтра.

    Точна рівність значення поля; поле, якого в метаданих немає, не
    збігається ні з чим (а не пропускається мовчки).
    """
    return all(metadata.get(key) == value for key, value in filters.items())


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти фрагменти, найближчі до вектора запиту.

    Схожість — косинусна (скалярний добуток нормалізованих векторів).
    Ранжування рахується для всієї колекції одразу — перебором: для
    кількох сотень фрагментів це швидше й простіше, ніж бібліотека
    наближеного пошуку, і не вимагає компромісу в точності.

    Поріг відсікає найслабші влучення до фільтрів за метаданими (слабке
    влучення не варте перевірки полів); фільтри звужують те, що
    лишилося; `top_k` обрізає вже відфільтроване й відсортоване. Такий
    порядок означає, що зменшення видачі фільтром ніколи не «підмішує»
    гірші влучення на місце кращих — top_k завжди береться з чесно
    відсортованого списку.
    """
    if len(index) == 0:
        return []
    query_vector = np.asarray(query_vector, dtype=np.float32)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    scores = index.vectors @ query_vector
    order = np.argsort(-scores)

    hits: list[Hit] = []
    for i in order:
        score = float(scores[i])
        if threshold is not None and score < threshold:
            break
        chunk = index.chunks[i]
        if filters and not _matches(chunk.metadata, filters):
            continue
        hits.append(Hit(chunk=chunk, score=score))
        if len(hits) >= top_k:
            break
    return hits
