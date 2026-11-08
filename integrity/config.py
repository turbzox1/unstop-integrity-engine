"""Central configuration. Every threshold used by the pipeline lives here so it can be tuned
and cited in EVALUATION.md."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- chunking -------------------------------------------------------------------
    # The brief suggests 200-400 tokens. Our synthetic submissions are 1-2 pages, so a
    # 300-token chunk would swallow a whole document and destroy localisation. We use
    # ~60-word chunks (~80 tokens) with a one-sentence overlap; see README "Assumptions".
    chunk_target_words: int = 60
    video_chunk_target_words: int = 60   # ~10-12 s of speech -> tighter timestamps
    chunk_overlap_sentences: int = 1
    min_chunk_words: int = 8           # very short chunks (titles, sign-offs) are ignored

    # ---- embeddings -----------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "auto"    # auto | sbert | tfidf

    # ---- similarity thresholds (cosine) -----------------------------------------------
    # Separate thresholds per backend because TF-IDF cosines live on a different scale.
    peer_threshold: dict = field(default_factory=lambda: {"sbert": 0.80, "tfidf": 0.45})
    external_threshold: dict = field(default_factory=lambda: {"sbert": 0.80, "tfidf": 0.45})
    top_k: int = 5

    # ---- boilerplate / template detection ---------------------------------------------
    # A sentence that appears (near-)verbatim in at least this many distinct documents is
    # treated as shared institutional text (case prompt, instructions) and discounted.
    boilerplate_min_docs: int = 3
    boilerplate_min_fraction: float = 0.0   # e.g. 0.02 at 10k scale
    boilerplate_sim: float = 0.92           # cosine for "near-verbatim" sentences (sbert)
    boilerplate_jaccard: float = 0.80       # lexical fallback for near-verbatim

    # ---- ASR ----------------------------------------------------------------------------
    asr_backend: str = "auto"          # auto | whisper | pocketsphinx
    whisper_model: str = "base"
    silence_max_volume_db: float = -50.0

    # ---- risk banding -------------------------------------------------------------------
    high_similarity_pct: float = 40.0
    medium_similarity_pct: float = 15.0
    high_ai_pct: float = 80.0
    medium_ai_pct: float = 60.0


DEFAULT = Config()
