"""Text cleaning, sentence splitting and lexical similarity helpers."""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_WORD = re.compile(r"[a-z0-9']+")


def clean(text: str) -> str:
    """Unicode-normalise, re-join hyphenated line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                # soft hyphen
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return _WS.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = clean(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def normalise_key(text: str) -> str:
    """Key used for exact-duplicate detection (case/punctuation/digits insensitive)."""
    return " ".join(re.sub(r"\d+", "#", t) for t in tokens(text))


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = tokens(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: str, b: str, n: int = 3) -> float:
    """Word n-gram Jaccard. Used to label matches verbatim vs paraphrase."""
    sa, sb = shingles(a, n), shingles(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalised character-level similarity (difflib ratio, a Levenshtein-like measure)."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower(), autojunk=False).ratio()
