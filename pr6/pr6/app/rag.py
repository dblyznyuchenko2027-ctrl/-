"""Конвеєр відповіді за базою знань: від питання до відповіді з джерелами.

Це єдине місце, яке знає всі кроки послідовно: знайти фрагменти
(`retrieval.retrieve`), зібрати з них контекст (`retrieval.build_context`),
спитати модель (`llm.ask`), перевірити те, що вона повернула, і скласти
результат для веб-рівня. Модулі-учасники одне про одного не знають:
пошук не знає про модель, модель — про індекс, веб-рівень — ні про що з
цього.

Функція `answer` — заготовка. Порядок кроків очевидний; рішення — ні:

* що робити, коли пошук не повернув нічого придатного: не викликати
  модель і відповісти наперед заданим текстом, чи викликати все одно —
  і чому;
* чи довіряти полю «відповідь знайдено», яке заповнила сама модель, і
  чим його перевірити: наприклад, чи існують названі нею номери джерел
  серед показаних, чи не порожній їх перелік;
* що показувати як джерела: лише ті фрагменти, на які модель послалася,
  чи всі, які їй показали;
* що потрапляє в `retrieved` — те, що бачила модель, — і чи віддавати
  це користувачеві чи лише вам для налагодження;
* що рахувати окремо: час пошуку і час генерації — це різні витрати, і
  на сторінці вони показуються окремо.
"""

import time
from dataclasses import dataclass, field

from . import llm
from . import retrieval as retrieval_module
from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source


@dataclass
class Answer:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `text` — відповідь для клієнта. `found` — чи відповідь спирається на
    базу знань (після вашої перевірки, а не зі слів моделі). `sources` —
    джерела, які показуються під відповіддю. `retrieved` — фрагменти,
    які бачила модель, для налагодження: за ними видно, чия це помилка —
    пошуку чи генерації. `elapsed` — час за етапами, наприклад
    `{"retrieval": 0.04, "generation": 1.9}`. `usage` — токени запиту й
    відповіді, якщо модель викликалась.
    """

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


NO_INFO_TEXT = "У базі знань немає інформації для відповіді на це питання."
UNVERIFIED_TEXT = (
    "Не можу впевнено відповісти на це питання: модель послалася на "
    "джерело, якого їй не показували. Зверніться, будь ласка, до "
    "оператора підтримки."
)


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Відповісти на питання за базою знань.

    Конвеєр: знайти (`retrieval.retrieve`) → зібрати контекст
    (`retrieval.build_context`) → спитати модель (`llm.ask`) → перевірити
    зміст того, що вона повернула → скласти `Answer`.

    Рішення, ухвалені тут:

    * Якщо після відбору не лишилося жодного фрагмента (поріг відсік усе
      або нічого не підійшло під фільтри) — модель НЕ викликається:
      відповідь наперед відома, детермінована й безкоштовна. Це ловить
      лише частину випадків "у базі немає" (питання про ціну навушників
      пройде поріг, бо інструкція до навушників схожа, — ціни в ній
      просто немає), тому другий рубіж — нижче.
    * Полю `found`, яке заповнила сама модель, не довіряємо напряму
      (як і в ПР4): відповідь визнається `found: true`, лише якщо модель
      сама сказала `found: true` **і** послалася щонайменше на один
      номер джерела, **і** серед її посилань немає жодного неіснуючого
      номера. Будь-яке з трьох порушень — відповідь не показується як
      факт із бази знань:
        - `found: true` без жодного джерела — підозріло (могла
          відповісти "з голови");
        - є посилання на номер, якого не було в контексті, — це збій
          (посилання не пройшло перевірку кодом), і показувати текст як
          підтверджений не можна;
        - інакше модель сама визнала, що відповіді немає — довіряємо.
    * Як джерела клієнту показуються лише ті фрагменти, на які модель
      реально послалася (`sources` у відповіді), а не всі, що їй
      показали, — "ось на чому ґрунтується відповідь", а не "ось що
      бачила модель". Друге — то `retrieved`, і воно завжди повне (всі
      фрагменти, передані в контекст), незалежно від того, чи відповідь
      визнано ґрунтованою: без нього для кожної невдачі довелося б
      вгадувати, чия вона — пошуку чи генерації.
    * Час пошуку й час генерації рахуються окремо: це різні витрати
      (локальна модель ембедінгів проти мережевого виклику), і на
      сторінці вони показуються окремо.
    """
    t0 = time.perf_counter()
    hits = retrieval_module.retrieve(question, index, keyword_index, filters=filters)
    t_retrieval = time.perf_counter() - t0

    if not hits:
        return Answer(
            text=NO_INFO_TEXT,
            found=False,
            sources=[],
            retrieved=[],
            model=None,
            elapsed={"retrieval": t_retrieval, "generation": 0.0},
            usage=None,
        )

    context, sources = retrieval_module.build_context(hits)
    sources_by_ref = {s.ref: s for s in sources}

    result = llm.ask(question, context)
    data = result["data"]

    cited_refs = list(dict.fromkeys(data.get("sources") or []))
    valid_refs = [ref for ref in cited_refs if ref in sources_by_ref]
    has_invalid_ref = any(ref not in sources_by_ref for ref in cited_refs)
    model_says_found = bool(data.get("found"))

    if model_says_found and valid_refs and not has_invalid_ref:
        text = data.get("answer") or NO_INFO_TEXT
        found = True
        shown_sources = [sources_by_ref[ref] for ref in valid_refs]
    elif has_invalid_ref:
        text = UNVERIFIED_TEXT
        found = False
        shown_sources = []
    else:
        text = NO_INFO_TEXT
        found = False
        shown_sources = []

    return Answer(
        text=text,
        found=found,
        sources=shown_sources,
        retrieved=hits,
        model=result["model"],
        elapsed={"retrieval": t_retrieval, "generation": result["elapsed"]},
        usage=result["usage"],
    )
