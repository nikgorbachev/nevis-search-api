"""Conservative text normalization for dirty OCR/email indexing and clean summaries.

Raw document content is never mutated in storage; normalized text is used only for
embeddings, full-text vectors, trigram/Levenshtein matching, and clean summaries.
"""

import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_LINE_HYPHEN = re.compile(r"([A-Za-z])-\s*\r?\n\s*([A-Za-z])")
_SPACED_LETTERS = re.compile(r"\b([A-Za-z])(?: ([A-Za-z])){3,}\b")
_QUOTE_PREFIX = re.compile(r"(?m)^\s*(?:>\s*)+")
_DIVIDER_RUN = re.compile(r"(?im)^\s*(?:[-=_*~]{4,}|-{2,}\s*(?:original|forwarded)\s+message\s*-{2,})\s*$")
_PCT_SPACE = re.compile(r"(\d)\s+%")
_CURRENCY_GLUED = re.compile(r"\b(EUR|USD|GBP|CHF)(\d)")
_FORM_CODE = re.compile(r"\b([A-Za-z]{2,10})[-\s](\d[\w-]*)\b")
_EMAIL_HEADER = re.compile(
    r"^(?:from|to|cc|bcc|sent|date|subject|reply-to)\s*:|^-{2,}\s*(?:original|forwarded)\s+message",
    re.I,
)
_VOCAB_WORD = re.compile(r"[a-z]{4,32}")


def normalize_text(text: str) -> str:
    """Clean OCR/PDF/email formatting noise while preserving semantics, numbers, and IDs."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = _ZERO_WIDTH.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _LINE_HYPHEN.sub(r"\1\2", s)
    s = _QUOTE_PREFIX.sub("", s)
    s = _DIVIDER_RUN.sub("", s)
    s = _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), s)
    s = _PCT_SPACE.sub(r"\1%", s)
    s = _CURRENCY_GLUED.sub(r"\1 \2", s)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n")]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fold_accents(text: str) -> str:
    """Accent-folded lowercase string matching PostgreSQL lower(unaccent(...))."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").lower()


def normalize_for_index(text: str) -> str:
    """Normalized, accent-folded text with compact form-code aliases for indexing."""
    clean = normalize_text(text)
    folded = fold_accents(clean)
    compact_codes = sorted({f"{a}{b}" for a, b in _FORM_CODE.findall(folded)})
    if compact_codes:
        return f"{folded} {' '.join(compact_codes)}"
    return folded


def extract_vocab_words(*texts: str) -> set[str]:
    """Extract lowercase alphabetic corpus words (len 4..32) for fuzzy typo lookup."""
    words: set[str] = set()
    for text in texts:
        if not text:
            continue
        folded = fold_accents(normalize_text(text))
        words.update(_VOCAB_WORD.findall(folded))
    return words


def summarize_document(title: str, content: str, max_chars: int = 220) -> str:
    """Deterministic, readable 1-2 sentence extractive summary from cleaned document text."""
    clean = normalize_text(content)
    substantive_lines = [
        line for line in clean.splitlines()
        if line and not _EMAIL_HEADER.match(line)
    ]
    body = " ".join(substantive_lines) if substantive_lines else normalize_text(title)
    if not body:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", body)
    summary = sentences[0]
    if len(sentences) > 1 and len(summary) + 1 + len(sentences[1]) <= max_chars:
        summary = f"{summary} {sentences[1]}"
    if len(summary) <= max_chars:
        return summary
    cutoff = summary.rfind(" ", 0, max_chars - 1)
    if cutoff < max_chars // 2:
        cutoff = max_chars - 1
    return summary[:cutoff].rstrip(" ,;:-") + "…"
