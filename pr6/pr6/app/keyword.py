"""Пошук за ключовими словами — точка порівняння для семантичного.

Цей модуль — той самий, що в ПР5, і тут він лишений заготовкою навмисно.
Замініть файл своєю реалізацією з `pr5/app/` цілком: у ПР6 вона не
змінюється, а лише використовується. Якщо в ПР5 ви перейменували
функції або поля — узгодьте з ними виклики в нових модулях.

Той самий набір фрагментів, той самий формат влучень (`Hit`), ті самі
фільтри — інший спосіб ранжувати. Без цього модуля не буде з чим
порівнювати семантичний пошук, а без порівняння — не буде відповіді на
питання, чи він узагалі потрібен для цієї колекції.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення:

* як розбивати текст на слова: що робити з регістром, розділовими
  знаками, апострофом у «звʼязок», числами й артикулами на кшталт
  `OR-X2-BLK`;
* чи зводити слова до основи — і чим, якщо так; без цього «повернути»
  й «повернення» для пошуку різні слова;
* чим ранжувати: кількістю збігів, TF-IDF, BM25 (пакет `rank_bm25`
  вже в залежностях);
* у яких одиницях оцінка й чи можна її порівнювати з оцінкою
  семантичного пошуку.
"""

import re
from dataclasses import dataclass, field

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit

_TOKEN_RE = re.compile(r"[a-zа-яіїєґ0-9]+(?:['\-][a-zа-яіїєґ0-9]+)*", re.IGNORECASE)


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами: самі фрагменти й готовий BM25-індекс
    (`extra["bm25"]`) поверх їхніх токенів."""

    chunks: list[Chunk]
    extra: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Одна й та сама функція для обох: запит і фрагмент мають розбиватися
    однаково, інакше збігів не буде. Основу слів не виділяємо (без
    стемера для української це окрема задача) — тому «повернення» і
    «повернути» для цього пошуку різні токени; це свідомий компроміс:
    точний пошук за словами як точка порівняння з семантичним, а не
    заміна йому.
    """
    return _TOKEN_RE.findall(text.lower())


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний.

    Ранжування — BM25 (`rank_bm25.BM25Okapi`): ураховує і частоту слова
    у фрагменті, і те, наскільки слово рідкісне в усій колекції, тому
    рідкісні терміни («Альтаїр», «сіріус-r10») важать більше за часті
    («доставка», «замовлення»).
    """
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return KeywordIndex(chunks=chunks, extra={"bm25": bm25})


def _matches(metadata: dict, filters: dict) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту.

    Формат результату — той самий `Hit`, що й у векторного пошуку, з тими
    самими фільтрами за метаданими. Оцінка тут — бал BM25, який росте без
    верхньої межі й **не порівнюється** з косинусною схожістю
    семантичного пошуку (0…1) — це різні шкали; звідси в `retrieval.py`
    видачі двох пошуків не змішуються за оцінкою, а розглядаються окремо.
    Фрагменти з нульовим балом (жодного спільного слова із запитом) не
    повертаються.
    """
    bm25 = index.extra.get("bm25")
    if bm25 is None or not index.chunks:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    hits: list[Hit] = []
    for i in order:
        score = float(scores[i])
        if score <= 0:
            break
        chunk = index.chunks[i]
        if filters and not _matches(chunk.metadata, filters):
            continue
        hits.append(Hit(chunk=chunk, score=score))
        if len(hits) >= top_k:
            break
    return hits
