"""Format-specific extractors: PDF, DOCX, MP4 (audio -> ASR) and plain text.

Every extractor returns a ``Document``; it never raises. Failures are recorded in
``Document.status`` / ``Document.warnings`` so one bad file cannot stop a batch.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import wave
from collections import Counter
from pathlib import Path

from .config import Config
from .models import Document, Segment
from .textnorm import clean, split_sentences

log = logging.getLogger(__name__)

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".mp4": "video", ".mov": "video",
             ".mkv": "video", ".txt": "text", ".md": "text"}
_PAGE_NUM = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)
DOCX_WORDS_PER_PAGE = 450   # estimate used when a DOCX has no explicit page breaks


def extract(path: Path, cfg: Config) -> Document:
    modality = SUPPORTED.get(path.suffix.lower())
    doc = Document(doc_id=path.name, path=str(path), modality=modality or "unknown")
    if modality is None:
        return _fail(doc, "unsupported", f"Unsupported file type '{path.suffix}'")
    if not path.exists() or path.stat().st_size == 0:
        return _fail(doc, "empty", "File is empty (0 bytes); skipped")
    try:
        {"pdf": _pdf, "docx": _docx, "video": _video, "text": _text}[modality](path, doc, cfg)
    except Exception as exc:  # noqa: BLE001 - any parser failure must be contained
        return _fail(doc, "corrupted", f"Could not parse file: {type(exc).__name__}: {exc}")
    if doc.status == "ok" and not doc.segments:
        _fail(doc, "empty", "No extractable text (scanned image PDF or blank document?)")
    return doc


def _fail(doc: Document, status: str, msg: str) -> Document:
    log.warning("%s: %s", doc.doc_id, msg)
    doc.status = status
    doc.warnings.append(msg)
    return doc


# ---------------------------------------------------------------------------------- PDF
def _pdf(path: Path, doc: Document, cfg: Config) -> None:
    import pymupdf as fitz

    with fitz.open(str(path)) as pdf:
        pages = []
        for page in pdf:
            h = page.rect.height
            blocks = []
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
                text = text.strip()
                if not text:
                    continue
                # Header/footer band: short blocks fully inside the top/bottom 7% margin.
                in_margin = y1 < 0.07 * h or y0 > 0.93 * h
                if in_margin and len(text.split()) <= 15:
                    doc.meta.setdefault("stripped_margin_lines", 0)
                    doc.meta["stripped_margin_lines"] += 1
                    continue
                blocks.append(text)
            pages.append(blocks)
    doc.meta["pages"] = len(pages)

    # Lines repeated on most pages (running headers that sit outside the margin band).
    if len(pages) >= 3:
        counts: Counter = Counter()
        for p in pages:
            for key in {re.sub(r"\d+", "#", b) for b in p}:
                counts[key] += 1
        repeated = {k for k, c in counts.items() if c >= 0.6 * len(pages)}
    else:
        repeated = set()

    for page_no, blocks in enumerate(pages, start=1):
        kept = [b for b in blocks if re.sub(r"\d+", "#", b) not in repeated
                and not _PAGE_NUM.match(b.strip())]
        # Split per text block so titles/headings do not fuse with the following sentence.
        for block in kept:
            for sent in split_sentences(block):
                doc.segments.append(Segment(sent, page=page_no))


# --------------------------------------------------------------------------------- DOCX
def _docx(path: Path, doc: Document, cfg: Config) -> None:
    import docx
    from docx.oxml.ns import qn

    d = docx.Document(str(path))
    page, words_on_page, explicit_breaks = 1, 0, False
    for para in d.paragraphs:
        brk = any(br.get(qn("w:type")) == "page" for br in para._p.iter(qn("w:br")))
        if brk:
            explicit_breaks = True
        text = clean(para.text)
        if text and not para.style.name.lower().startswith(("header", "footer")):
            for sent in split_sentences(text):
                doc.segments.append(Segment(sent, page=page))
                words_on_page += len(sent.split())
        if brk:
            page, words_on_page = page + 1, 0
        elif not explicit_breaks and words_on_page >= DOCX_WORDS_PER_PAGE:
            page, words_on_page = page + 1, 0
    # Tables often carry real content in executive summaries.
    for table in d.tables:
        for row in table.rows:
            row_text = clean(" | ".join(c.text for c in row.cells))
            if len(row_text.split()) >= 4:
                doc.segments.append(Segment(row_text, page=page))
    doc.meta["page_numbers"] = "explicit" if explicit_breaks else f"estimated ({DOCX_WORDS_PER_PAGE} words/page)"


# --------------------------------------------------------------------------------- text
def _text(path: Path, doc: Document, cfg: Config) -> None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    for sent in split_sentences(raw):
        doc.segments.append(Segment(sent, page=1))


# -------------------------------------------------------------------------------- video
def _video(path: Path, doc: Document, cfg: Config) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        _fail(doc, "error", "ffmpeg/ffprobe not found on PATH; cannot process video")
        return
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of",
                            "json", str(path)], capture_output=True, text=True)
    if probe.returncode != 0:
        _fail(doc, "corrupted", f"ffprobe failed: {probe.stderr.strip()[:200]}")
        return
    info = json.loads(probe.stdout or "{}")
    doc.meta["duration_s"] = round(float(info.get("format", {}).get("duration", 0) or 0), 2)
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        _fail(doc, "no_audio", "Video has no audio track; nothing to transcribe (flagged for manual review)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                        "-ar", "16000", "-sample_fmt", "s16", str(wav)], check=True)
        vol = subprocess.run(["ffmpeg", "-i", str(wav), "-af", "volumedetect", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        m = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", vol)
        max_db = float("-inf") if not m or m.group(1) == "-inf" else float(m.group(1))
        doc.meta["max_volume_db"] = max_db
        if max_db < cfg.silence_max_volume_db:
            _fail(doc, "silent", f"Audio track is silent (max volume {max_db} dB); flagged for manual review")
            return
        segments, backend = transcribe(wav, cfg)
    doc.meta["asr_backend"] = backend
    if backend == "none":
        _fail(doc, "error", "No ASR backend available (install openai-whisper or pocketsphinx)")
        return
    for text, start, end in segments:
        text = clean(text)
        if text:
            doc.segments.append(Segment(text, start=start, end=end))


def transcribe(wav: Path, cfg: Config) -> tuple[list[tuple[str, float, float]], str]:
    """Return [(text, start_s, end_s)] and the backend name used."""
    order = ["whisper", "pocketsphinx"] if cfg.asr_backend == "auto" else [cfg.asr_backend]
    for backend in order:
        try:
            if backend == "whisper":
                return _whisper(wav, cfg), f"whisper-{cfg.whisper_model}"
            if backend == "pocketsphinx":
                return _pocketsphinx(wav), "pocketsphinx"
        except Exception as exc:  # noqa: BLE001
            log.info("ASR backend %s unavailable: %s", backend, exc)
    return [], "none"


_WHISPER_CACHE: dict = {}


def _whisper(wav: Path, cfg: Config) -> list[tuple[str, float, float]]:
    import whisper

    model = _WHISPER_CACHE.get(cfg.whisper_model)
    if model is None:
        model = _WHISPER_CACHE[cfg.whisper_model] = whisper.load_model(cfg.whisper_model)
    result = model.transcribe(str(wav), language="en", fp16=False)
    return [(s["text"], float(s["start"]), float(s["end"])) for s in result["segments"]]


def _pocketsphinx(wav: Path, words_per_segment: int = 12) -> list[tuple[str, float, float]]:
    """Offline CMU PocketSphinx fallback. Lower accuracy than Whisper, no punctuation, so
    we group recognised words into fixed-size pseudo-sentences that keep timestamps."""
    from pocketsphinx import Decoder

    with wave.open(str(wav), "rb") as w:
        audio = w.readframes(w.getnframes())
    decoder = Decoder(samprate=16000, loglevel="FATAL")
    decoder.start_utt()
    decoder.process_raw(audio, full_utt=True)
    decoder.end_utt()
    frate = 100.0
    words = [(s.word.split("(")[0], s.start_frame / frate, s.end_frame / frate)
             for s in decoder.seg() if not s.word.startswith(("<", "[")) and s.word != "<sil>"]
    out = []
    for i in range(0, len(words), words_per_segment):
        group = words[i:i + words_per_segment]
        out.append((" ".join(w for w, _, _ in group) + ".", group[0][1], group[-1][2]))
    return out
