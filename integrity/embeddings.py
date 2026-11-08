"""Embedding backends + a tiny vector index.

* ``sbert``  - Sentence-Transformers (default all-MiniLM-L6-v2, 384-d, L2-normalised).
* ``tfidf``  - offline fallback (word 1-2 grams + char 3-5 grams TF-IDF). Used automatically
               when the neural model cannot be loaded (no internet / no torch), so the pipeline
               always runs. It is lexical, so it is weaker on paraphrase (see EVALUATION.md).

Vectors are L2-normalised, so inner product == cosine similarity.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.backend = None
        self._model = None
        self._tfidf = None
        if backend in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name, device="cpu")
                self.backend = "sbert"
            except Exception as exc:  # noqa: BLE001
                if backend == "sbert":
                    raise
                log.warning("Sentence-Transformers unavailable (%s); falling back to TF-IDF", exc)
        if self.backend is None:
            self.backend = "tfidf"

    @property
    def name(self) -> str:
        return self.model_name if self.backend == "sbert" else "tfidf(word1-2+char3-5)"

    def fit(self, corpus: list[str]) -> "Embedder":
        """TF-IDF needs a vocabulary; fit it on every text that will be embedded."""
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import FeatureUnion
            self._tfidf = FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
            ]).fit(corpus)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        if self.backend == "sbert":
            vecs = self._model.encode(texts, batch_size=64, normalize_embeddings=True,
                                      show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        vecs = self._tfidf.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)


class VectorIndex:
    """Exact inner-product search. Uses FAISS IndexFlatIP when installed, else NumPy.
    Flat (exact) search is the right choice at prototype scale (<1e5 vectors); see README
    for the HNSW / IVF-PQ plan at 10k submissions."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self._faiss = None
        try:
            import faiss
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except Exception:  # noqa: BLE001
            self._faiss = None

    @property
    def kind(self) -> str:
        return "faiss.IndexFlatIP" if self._faiss is not None else "numpy-cosine"

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.vectors))
        queries = np.ascontiguousarray(queries, dtype="float32")
        if self._faiss is not None:
            return self._faiss.search(queries, k)
        sims = queries @ self.vectors.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx
