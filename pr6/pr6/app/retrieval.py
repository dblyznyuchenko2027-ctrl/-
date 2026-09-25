"""Відбір фрагментів для моделі та збирання контексту.

Між пошуком із ПР5 і мовною моделлю стоїть шар, якого в ПР5 не було: він
вирішує, *що саме* модель побачить. Пошук повертає влучення з оцінками —
стільки, скільки попросили; модель потребує тексту — обмеженого за
розміром, упорядкованого, з позначками джерел, на які вона зможе
послатися. Цей шар — тут.

Сам пошук сюди імпортується, а не переписується: `index.search` і
`keyword.search` — ваші модулі з ПР5. Веб-рівень сюди не звертається;
цей модуль викликає `app/rag.py`.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яким пошуком добирати фрагменти — семантичним, за словами, обома
  (і як тоді злити дві видачі в одну);
* які фільтри накладає код **завжди**, незалежно від того, що просив
  користувач: аудиторія, статус документа;
* скільки фрагментів віддавати моделі, чи згортати кілька фрагментів
  одного документа й чи потрібен поріг «нижче — не віддавати»;
* у якому порядку ставити фрагменти в контекст і як їх позначати, щоб
  модель могла послатися на конкретний;
* як умістити контекст у бюджет і що відкидати, коли він не вміщується.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from . import embeddings
from . import keyword as keyword_module
from .documents import Chunk
from .index import Hit, SearchIndex
from .index import search as semantic_search
from .keyword import KeywordIndex

load_dotenv()

CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))

_SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "8"))

MANDATORY_FILTERS = {"audience": "клієнти", "status": "чинний"}

ALLOWED_USER_FILTERS = {"product"}

MAX_PER_SOURCE = 2


class FilterError(ValueError):
    """Фільтр, якого сторінці не дозволено передавати: невідоме поле або
    поле, яке керує внутрішнім запобіжником (audience, status)."""


@dataclass
class Source:
    """Джерело, показане моделі й користувачеві.

    `ref` — номер, під яким фрагмент стоїть у контексті й на який модель
    посилається у відповіді. `chunk` — сам фрагмент із метаданими;
    `score` — оцінка пошуку, за якою його відібрано.
    """

    ref: int
    chunk: Chunk
    score: float


def _validate_filters(filters: dict) -> None:
    unknown = set(filters) - ALLOWED_USER_FILTERS
    if unknown:
        raise FilterError(
            "невідомий або заборонений фільтр: " + ", ".join(sorted(unknown))
        )


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти, які варто показати моделі.

    Кроки:

    1. Перевірити, що `filters` від сторінки містить лише дозволене поле
       (`product`) — інакше `FilterError` до будь-якого звернення до
       пошуку чи моделі (порожнє питання перевіряє `app/rag.py`/веб-рівень
       раніше за цю функцію).
    2. Додати `MANDATORY_FILTERS` — вони йдуть **після** `filters` у
       злитті словників, тож навіть якби користувацький фільтр містив
       ключ `audience` чи `status`, він однаково не пройшов би крок 1 і
       не дійшов би сюди; тут просто немає жодного шляху, яким запит зі
       сторінки міг би їх перевизначити.
    3. Семантичний пошук (`index.search`) — основний спосіб відбору:
       колекція багатомовна, і модель ембедінгів впорається з
       перифразами й синонімами краще, ніж збіг слів. Пошук за словами
       (`keyword.search`) додає лише те, що семантичний пропустив і чого
       немає в його видачі — оцінки цих двох пошуків не в одних одиницях
       (косинусна схожість проти бала BM25), тому їх не змішують, а
       просто доповнюють список: спершу все, що дав семантичний пошук
       (він і визначає порядок), потім рештки з видачі за словами як
       додаткові кандидати нижче за пріоритетом.
    4. Не більш ніж `MAX_PER_SOURCE` фрагментів одного документа —
       інакше один довгий документ міг би зайняти весь контекст коштом
       іншого потрібного джерела (питання "гроші й бонуси" з двох
       документів — саме такий випадок).
    5. Обрізати до `CONTEXT_CHUNKS`: пошук міг дивитися ширше
       (`SEARCH_TOP_K`), моделі йде менше і лише найкраще.

    Порожній список — коректний результат: він означає «у базі знань про
    це немає» (або поріг відсік усе), і що робити далі, вирішує
    `app/rag.py` — саме там перевіряють порожню видачу до виклику моделі.
    """
    if query is None or not query.strip():
        raise ValueError("порожнє питання")

    filters = dict(filters or {})
    _validate_filters(filters)
    combined_filters = {**filters, **MANDATORY_FILTERS}

    query_vector = embeddings.embed_query(query)
    hits = list(semantic_search(index, query_vector, top_k=_SEARCH_TOP_K, filters=combined_filters))

    if keyword_index is not None:
        seen_ids = {id(h.chunk) for h in hits}
        keyword_hits = keyword_module.search(
            keyword_index, query, top_k=_SEARCH_TOP_K, filters=combined_filters
        )
        for h in keyword_hits:
            if id(h.chunk) not in seen_ids:
                hits.append(h)
                seen_ids.add(id(h.chunk))

    per_source_count: dict[str, int] = {}
    collapsed: list[Hit] = []
    for hit in hits:
        source = hit.chunk.source
        per_source_count[source] = per_source_count.get(source, 0) + 1
        if per_source_count[source] <= MAX_PER_SOURCE:
            collapsed.append(hit)

    return collapsed[:CONTEXT_CHUNKS]


def _estimate_tokens(text: str) -> int:
    """Та сама груба оцінка, що в ПР4: приблизно 4 символи змішаного
    українсько-англійського тексту на один токен. Із запасом — оцінка
    свідомо не рахує токенізатор моделі, а лише відсікає явний перебір."""
    return max(1, len(text) // 4)


def build_context(hits: list[Hit], budget: int = CONTEXT_BUDGET) -> tuple[str, list[Source]]:
    """Зібрати з влучень текст контексту для моделі та перелік джерел.

    Порядок фрагментів у контексті — той самий, що в `hits` (за спаданням
    релевантності), тобто найкраще влучення йде **першим**. Моделі за
    даними досліджень ("lost in the middle") краще вдаються початок і
    кінець довгого контексту, ніж середина; ставити найкраще влучення
    першим означає, що навіть якщо контекст обрізано бюджетом, найважливіше
    вже точно потрапило в нього.

    Позначка кожного фрагмента: `[номер] Назва документа · Заголовок
    розділу · редакція від ДАТА` — номер потрібен моделі, щоб послатися на
    конкретний фрагмент, а код (`app/rag.py`) — щоб перевірити посилання;
    назва й розділ дають те, чого немає в самому уривку; дата дозволяє
    відрізнити чинне від застарілого, якщо обидва раптом опиняться в
    контексті. Текст фрагмента йде як є, без переказу.

    Фрагменти додаються, доки вміщуються в `budget` (токени рахуються
    грубою оцінкою символів). Щойно черговий фрагмент не влазить —
    решту відкидаємо; ми вже поклали найкраще влучення першим, тож
    найважливіше не втрачається навіть при обрізанні.
    """
    parts: list[str] = []
    sources: list[Source] = []
    used = 0
    for hit in hits:
        chunk = hit.chunk
        meta = chunk.metadata
        ref = len(sources) + 1
        label_bits = [f"[{ref}]", meta.get("title") or chunk.source]
        if meta.get("heading"):
            label_bits.append(meta["heading"])
        if meta.get("updated"):
            label_bits.append(f"редакція від {meta['updated']}")
        label = " · ".join(label_bits)
        piece = f"{label}\n{chunk.text}"
        piece_tokens = _estimate_tokens(piece)

        if parts and used + piece_tokens > budget:
            break
        parts.append(piece)
        used += piece_tokens
        sources.append(Source(ref=ref, chunk=chunk, score=hit.score))

    context_text = "\n\n".join(parts)
    return context_text, sources
