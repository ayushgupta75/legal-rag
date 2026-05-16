"""
Legal-aware chunker.

Strategy: recursive structural splitting — split on natural legal boundaries
(article → section → paragraph → sentence) and only go deeper when a chunk
exceeds MAX_CHARS.  Every chunk carries full citation metadata and a parent_id
so callers can do parent-doc retrieval.
"""
import re
import uuid
import hashlib

from dataclasses import dataclass, field
from typing import Iterator

MAX_CHARS = 2000   # target ceiling per chunk
OVERLAP   = 200    # character overlap between consecutive chunks at the same level

# Ordered from coarsest to finest — we try each level until chunks fit
SEPARATOR_LEVELS = [
    re.compile(r"(?m)^(?=(?:ARTICLE|AMENDMENT|TITLE|PART|CHAPTER)\s+[IVXLC\d]+)", re.I),
    re.compile(r"(?m)^(?=(?:Section|Sec\.?|§)\s*[\d\w.–-]+)", re.I),
    re.compile(r"\n\n+"),
    re.compile(r"(?<=[.!?])\s+"),
]


@dataclass
class LegalChunkData:
    id: str
    source: str
    title: str
    section: str
    jurisdiction: str
    citation: str
    text: str
    char_count: int
    effective_date: str | None
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def version_hash(self) -> str:
        return hashlib.md5(self.text.encode()).hexdigest()



def _split(text: str, pattern: re.Pattern) -> list[str]:
    """Split text at pattern boundaries, keeping each header attached to its body."""
    parts = pattern.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _add_overlap(parts: list[str], overlap: int) -> list[str]:
    """Append a tail of the previous chunk to the start of the next."""
    if len(parts) <= 1:
        return parts
    result = [parts[0]]
    for i in range(1, len(parts)):
        tail = result[i - 1][-overlap:] if len(result[i - 1]) > overlap else result[i - 1]
        result.append(tail + "\n" + parts[i])
    return result


def _recursive_split(text: str, level: int = 0) -> list[str]:
    """
    Split text using SEPARATOR_LEVELS[level].
    Always tries structural splits first; only falls back to deeper levels
    when a chunk exceeds MAX_CHARS or the current pattern finds no boundaries.
    """
    if level >= len(SEPARATOR_LEVELS):
        return [text]

    parts = _split(text, SEPARATOR_LEVELS[level])

    # Pattern found no boundaries — go deeper only if text is oversized
    if len(parts) <= 1:
        if len(text) > MAX_CHARS:
            return _recursive_split(text, level + 1)
        return [text]

    # Split happened — recurse into any oversized children
    result = []
    for part in parts:
        if len(part) > MAX_CHARS:
            result.extend(_recursive_split(part, level + 1))
        else:
            result.append(part)

    # Only add overlap at paragraph/sentence level (not structural boundaries)
    # — structural headers must remain the first line for citation extraction
    if level >= 2:
        return _add_overlap(result, OVERLAP)
    return result


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def chunk_document(
    raw_text: str,
    *,
    source: str,
    title: str,
    jurisdiction: str,
    effective_date: str | None = None,
    citation_prefix: str = "",
    extra_metadata: dict | None = None,
) -> Iterator[LegalChunkData]:
    """
    Generic structural chunker for any legal document.

    Splits recursively on legal boundaries (article → section → paragraph →
    sentence) until each chunk is under MAX_CHARS.  Parent chunks are yielded
    first; child chunks reference them via parent_id.
    """
    top_chunks = _recursive_split(raw_text)

    for top_text in top_chunks:
        top_id = str(uuid.uuid4())
        top_header = _first_line(top_text)
        top_citation = f"{citation_prefix} {top_header}".strip()

        # Yield the top-level chunk (acts as the parent for retrieval expansion)
        yield LegalChunkData(
            id=top_id,
            source=source,
            title=title,
            section=top_header,
            jurisdiction=jurisdiction,
            citation=top_citation,
            text=top_text.strip(),
            char_count=len(top_text.strip()),
            effective_date=effective_date,
            parent_id=None,
            metadata=extra_metadata or {},
        )

        # If the top chunk was itself sub-split, yield children with parent_id
        children = _recursive_split(top_text, level=1)
        if len(children) > 1:
            for child_text in children:
                if len(child_text.strip()) < 50:
                    continue
                child_header = _first_line(child_text)
                yield LegalChunkData(
                    id=str(uuid.uuid4()),
                    source=source,
                    title=title,
                    section=child_header or top_header,
                    jurisdiction=jurisdiction,
                    citation=f"{top_citation}, {child_header}".strip(", "),
                    text=child_text.strip(),
                    char_count=len(child_text.strip()),
                    effective_date=effective_date,
                    parent_id=top_id,
                    metadata=extra_metadata or {},
                )


# ---------------------------------------------------------------------------
# Convenience wrappers — thin shims over chunk_document
# ---------------------------------------------------------------------------

def chunk_constitution(raw_text: str) -> Iterator[LegalChunkData]:
    yield from chunk_document(
        raw_text,
        source="constitution",
        title="United States Constitution",
        jurisdiction="federal",
        effective_date="1788-06-21",
        citation_prefix="U.S. Const.",
    )


def chunk_uscode(raw_text: str, title_num: str, title_name: str) -> Iterator[LegalChunkData]:
    yield from chunk_document(
        raw_text,
        source="uscode",
        title=f"Title {title_num} - {title_name}",
        jurisdiction="federal",
        citation_prefix=f"{title_num} U.S.C.",
        extra_metadata={"title_num": title_num},
    )


def chunk_cfr(raw_text: str, title_num: str, part: str) -> Iterator[LegalChunkData]:
    yield from chunk_document(
        raw_text,
        source="cfr",
        title=f"CFR Title {title_num}, Part {part}",
        jurisdiction="federal",
        citation_prefix=f"{title_num} C.F.R. pt. {part}",
        extra_metadata={"title_num": title_num, "part": part},
    )
