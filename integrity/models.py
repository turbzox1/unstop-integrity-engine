"""Plain dataclasses shared across pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Segment:
    """A sentence-sized unit of text with its location in the source file."""
    text: str
    page: Optional[int] = None          # 1-based page (PDF) or section (DOCX)
    start: Optional[float] = None       # seconds (video)
    end: Optional[float] = None


@dataclass
class Document:
    doc_id: str
    path: str
    modality: str                       # pdf | docx | video | text
    segments: list[Segment] = field(default_factory=list)
    status: str = "ok"                  # ok | empty | corrupted | no_audio | silent | error
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def usable(self) -> bool:
        return self.status == "ok" and bool(self.segments)


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    modality: str
    text: str
    sentence_idx: list[int]             # indices into the document's segment list
    page: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def location(self) -> dict:
        if self.start is not None:
            return {"timestamp": f"{fmt_ts(self.start)} - {fmt_ts(self.end or self.start)}"}
        return {"page": self.page}


def fmt_ts(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
