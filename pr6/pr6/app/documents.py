"""Колекція документів: читання, метадані, поділ на фрагменти.

Цей модуль — той самий, що в ПР5, і тут він лишений заготовкою навмисно.
Замініть файл своєю реалізацією з `pr5/app/` цілком: у ПР6 вона не
змінюється, а лише використовується. Якщо в ПР5 ви перейменували
функції або поля — узгодьте з ними виклики в нових модулях.

Це єдине місце, яке знає, як влаштовані файли в `docs/`: де в них
метадані, як розмічено текст, за якими межами його ділити. Решта
застосунку працює з готовими фрагментами (`Chunk`) і не читає файлів.

Розбір блоку метаданих реалізовано: це формат файлів, а не предмет
роботи. Поділ на фрагменти — заготовка: розмір, межі, перекриття і те,
що саме потрапляє в кожен фрагмент, — рішення з розділу 2 практичної
роботи.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс те, що ви вирішите додати самі:
    заголовок розділу, порядковий номер фрагмента, позицію в документі.
    Фільтри пошуку працюють саме з цим словником.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку
    файлу. Повертає словник метаданих і решту тексту. Порожні значення
    (`product:` без нічого) стають порожнім рядком. Якщо блоку немає —
    порожній словник і текст як є.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


_H1_RE = re.compile(r"^#\s+.*$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def _strip_h1(text: str) -> str:
    """Прибрати рядок заголовка першого рівня (`# Назва`) з початку тексту.

    Назва документа вже є в метаданих (`title`); дублювати її як окремий
    "розділ" немає сенсу — вона додається до кожного фрагмента нижче.
    """
    lines = text.split("\n")
    while lines and (not lines[0].strip() or _H1_RE.match(lines[0].strip())):
        if lines[0].strip() and not _H1_RE.match(lines[0].strip()):
            break
        lines.pop(0)
    return "\n".join(lines)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Поділити тіло документа на розділи за заголовками другого рівня.

    Заголовки `## ...` — природна межа фрагмента: під одним заголовком
    зазвичай стоїть одна закінчена думка ("Строк зберігання", "Скидання
    до заводських налаштувань"), і саме тому ділимо саме за ними, а не за
    довільною кількістю абзаців. Повертає список (заголовок, текст
    розділу); текст перед першим заголовком (якщо є) іде з порожнім
    заголовком.
    """
    body = _strip_h1(text)
    matches = list(_H2_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [("", stripped)] if stripped else []

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = body[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        if content:
            sections.append((heading, content))
    return sections


def _split_by_budget(content: str, size: int, overlap: int) -> list[str]:
    """Поділити текст розділу на шматки не довші за `size` символів, із
    перекриттям `overlap` між сусідніми.

    Більшість розділів у цій колекції коротші за `size` і виходять одним
    шматком без жодного поділу — до нього доходить лише те, що справді
    задовге. Ріжемо по межі речення чи абзацу, коли це можливо, щоб не
    розрізати умову й число посередині ("Посилка зберігається... 7
    днів"), а коли ні — по межі слова.
    """
    content = content.strip()
    if len(content) <= size:
        return [content] if content else []

    step = max(size - overlap, 1)
    pieces: list[str] = []
    start = 0
    n = len(content)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = content[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut == -1:
                cut = window.rfind(" ")
            if cut > size * 0.5:
                end = start + cut + 1
        piece = content[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Поділити текст документа на фрагменти.

    Межі — заголовки другого рівня (`_split_into_sections`); задовгий
    розділ додатково ріжеться за розміром символів із перекриттям
    (`_split_by_budget`, параметри `CHUNK_SIZE`/`CHUNK_OVERLAP`).

    Кожен фрагмент отримує на початку назву документа й заголовок
    розділу ("Навушники Оріон X2 — Скидання до заводських налаштувань"),
    щоб уривок був зрозумілий сам по собі — без цього "Утримуйте кнопку
    10 секунд" не каже, про який пристрій ідеться і в якому режимі. Це
    той самий текст, що йде в індекс, у показ користувачеві й у контекст
    моделі — переказу тут немає.

    У метадані фрагмента, понад метадані документа, додається `heading`
    (заголовок розділу) і `chunk_index` (порядковий номер фрагмента в
    документі) — для позначки джерела і для діагностики.
    """
    title = (metadata.get("title") or source).strip()
    sections = _split_into_sections(text)
    chunks: list[Chunk] = []
    chunk_index = 0
    for heading, content in sections:
        for piece in _split_by_budget(content, CHUNK_SIZE, CHUNK_OVERLAP):
            chunk_index += 1
            prefix = f"{title} — {heading}" if heading else title
            full_text = f"{prefix}\n\n{piece}"
            chunk_metadata = dict(metadata)
            chunk_metadata["heading"] = heading
            chunk_metadata["chunk_index"] = chunk_index
            chunks.append(Chunk(text=full_text, source=source, metadata=chunk_metadata))
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
