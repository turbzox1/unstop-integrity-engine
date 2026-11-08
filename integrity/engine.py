"""Core matching engine: peer-to-peer, cross-modal and external similarity."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ai_detect import AIDetector
from .chunking import chunk_document, detect_boilerplate
from .config import Config
from .embeddings import Embedder, VectorIndex
from .extractors import SUPPORTED, extract
from .models import Chunk, Document
from .textnorm import jaccard, levenshtein_ratio

log = logging.getLogger(__name__)


@dataclass
class Match:
    query: Chunk
    source: Chunk
    score: float


def load_corpus(input_dir: Path, cfg: Config) -> list[Document]:
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("."))
    skipped = sorted(p.name for p in input_dir.iterdir()
                     if p.is_file() and p.suffix.lower() not in SUPPORTED and p.suffix != ".json")
    for name in skipped:
        log.warning("%s: unsupported extension, ignored", name)
    docs = []
    for p in files:
        log.info("Extracting %s", p.name)
        docs.append(extract(p, cfg))
    return docs


def load_reference(ref_dir: Path | None, cfg: Config) -> list[Document]:
    if ref_dir is None or not ref_dir.exists():
        return []
    refs = []
    for p in sorted(ref_dir.glob("*")):
        if p.suffix.lower() not in SUPPORTED:
            continue
        doc = extract(p, cfg)
        if doc.modality == "text" and doc.segments and doc.segments[0].text.startswith("SOURCE_URL:"):
            first = doc.segments.pop(0).text
            doc.meta["url"] = first.replace("SOURCE_URL:", "").strip()
        if doc.usable:
            refs.append(doc)
    return refs


class IntegrityEngine:
    def __init__(self, cfg: Config, ai_mode: str = "heuristic"):
        self.cfg = cfg
        self.embedder = Embedder(cfg.embedding_backend, cfg.embedding_model)
        self.ai = AIDetector(ai_mode)

    # ------------------------------------------------------------------------------
    def run(self, docs: list[Document], refs: list[Document]) -> dict:
        cfg = self.cfg
        usable = [d for d in docs if d.usable]
        boiler = detect_boilerplate(usable, cfg)

        chunks: list[Chunk] = []
        for d in usable:
            chunks += chunk_document(d, boiler.get(d.doc_id, set()), cfg, start_id=len(chunks))
        ref_chunks: list[Chunk] = []
        for r in refs:
            ref_chunks += chunk_document(r, set(), cfg, start_id=len(ref_chunks))

        all_texts = [c.text for c in chunks] + [c.text for c in ref_chunks]
        self.embedder.fit(all_texts or ["empty"])
        thr_peer = cfg.peer_threshold[self.embedder.backend]
        thr_ext = cfg.external_threshold[self.embedder.backend]

        peer_matches = self._match(chunks, chunks, thr_peer, exclude_same_doc=True)
        ext_matches = self._match(chunks, ref_chunks, thr_ext, exclude_same_doc=False)

        doc_by_id = {d.doc_id: d for d in docs}
        ref_by_id = {r.doc_id: r for r in refs}
        results = {}
        for d in docs:
            results[d.doc_id] = self._doc_result(d, boiler.get(d.doc_id, set()), chunks,
                                                 peer_matches, ext_matches, doc_by_id, ref_by_id)
        matrix = self._pair_matrix(usable, boiler, peer_matches)
        return {
            "documents": results,
            "pair_matrix": matrix,
            "run_info": {
                "embedding_backend": self.embedder.backend,
                "embedding_model": self.embedder.name,
                "vector_index": VectorIndex(np.zeros((1, 4), dtype="float32")).kind,
                "peer_threshold": thr_peer,
                "external_threshold": thr_ext,
                "n_documents": len(docs),
                "n_usable": len(usable),
                "n_chunks": len(chunks),
                "n_reference_docs": len(refs),
                "n_reference_chunks": len(ref_chunks),
                "ai_detector": self.ai.mode,
            },
        }

    # ------------------------------------------------------------------------------
    def _match(self, queries: list[Chunk], corpus: list[Chunk], thr: float,
               exclude_same_doc: bool) -> dict[int, list[Match]]:
        """For each query chunk: best-scoring chunk per *other* document above thr."""
        out: dict[int, list[Match]] = defaultdict(list)
        if not queries or not corpus:
            return out
        q_vecs = self.embedder.encode([c.text for c in queries])
        c_vecs = q_vecs if corpus is queries else self.embedder.encode([c.text for c in corpus])
        index = VectorIndex(c_vecs)
        # ask for enough neighbours that self-doc chunks cannot crowd out other documents
        k = min(len(corpus), self.cfg.top_k + 40)
        sims, idx = index.search(q_vecs, k)
        for qi, q in enumerate(queries):
            best: dict[str, Match] = {}
            for s, j in zip(sims[qi], idx[qi]):
                if j < 0:
                    continue
                src = corpus[j]
                if exclude_same_doc and src.doc_id == q.doc_id:   # self-match exclusion
                    continue
                if s < thr:
                    break                                           # results are sorted
                if src.doc_id not in best or s > best[src.doc_id].score:
                    best[src.doc_id] = Match(q, src, float(s))
            out[q.chunk_id] = sorted(best.values(), key=lambda m: -m.score)
        return out

    # ------------------------------------------------------------------------------
    @staticmethod
    def _coverage(doc: Document, boiler: set[int], doc_chunks: list[Chunk],
                  matches: dict[int, list[Match]], only_source: str | None = None) -> float:
        """% of the document's non-boilerplate words that sit inside a matched chunk."""
        countable = [i for i in range(len(doc.segments)) if i not in boiler]
        total = sum(len(doc.segments[i].text.split()) for i in countable)
        if total == 0:
            return 0.0
        covered: set[int] = set()
        for c in doc_chunks:
            ms = matches.get(c.chunk_id, [])
            if only_source is not None:
                ms = [m for m in ms if m.source.doc_id == only_source]
            if ms:
                covered.update(c.sentence_idx)
        words = sum(len(doc.segments[i].text.split()) for i in covered if i not in boiler)
        return round(100.0 * words / total, 1)

    def _pair_matrix(self, usable, boiler, peer_matches) -> dict:
        by_doc = defaultdict(list)
        for ms in peer_matches.values():
            for m in ms:
                by_doc[m.query.doc_id].append(m.query)
        matrix = {}
        for d in usable:
            row = {}
            chunks_d = list({c.chunk_id: c for c in by_doc[d.doc_id]}.values())
            for other in usable:
                if other.doc_id == d.doc_id:
                    row[other.doc_id] = None
                    continue
                row[other.doc_id] = self._coverage(d, boiler.get(d.doc_id, set()), chunks_d,
                                                   peer_matches, only_source=other.doc_id)
            matrix[d.doc_id] = row
        return matrix

    # ------------------------------------------------------------------------------
    def _doc_result(self, d, boiler, chunks, peer_matches, ext_matches, doc_by_id, ref_by_id):
        cfg = self.cfg
        base = {"document_id": d.doc_id, "modality": d.modality, "status": d.status,
                "warnings": d.warnings, "extraction_meta": d.meta}
        if not d.usable:
            base.update({
                "total_integrity_risk": "NEEDS_MANUAL_REVIEW",
                "scores": {"peer_similarity_percentage": None, "external_similarity_percentage": None,
                           "ai_generated_probability": None},
                "peer_matches": [], "multimodal_matches": [], "external_matches": [],
                "boilerplate_excluded": [],
                "explanations": f"Not scored: {'; '.join(d.warnings)}. The file was skipped without "
                                "affecting the rest of the batch; a reviewer should request a valid re-upload.",
            })
            return base

        doc_chunks = [c for c in chunks if c.doc_id == d.doc_id]
        peer_pct = self._coverage(d, boiler, doc_chunks, peer_matches)
        ext_pct = self._coverage(d, boiler, doc_chunks, ext_matches)
        ai = self.ai.score(" ".join(s.text for i, s in enumerate(d.segments) if i not in boiler),
                           d.modality)

        peer_list, mm_list = [], []
        for m in self._dedupe([m for c in doc_chunks for m in peer_matches.get(c.chunk_id, [])]):
            entry = self._explain(m, doc_by_id[m.source.doc_id], d)
            (mm_list if m.source.modality != d.modality and "video" in (m.source.modality, d.modality)
             else peer_list).append(entry)
        ext_list = [self._explain(m, ref_by_id[m.source.doc_id], d, external=True)
                    for m in self._dedupe([m for c in doc_chunks for m in ext_matches.get(c.chunk_id, [])])]

        risk = self._risk(peer_pct, ext_pct, ai.get("probability"))
        boiler_list = [{"text": d.segments[i].text, **self._seg_loc(d.segments[i])} for i in sorted(boiler)]
        base.update({
            "total_integrity_risk": risk,
            "scores": {"peer_similarity_percentage": peer_pct,
                       "external_similarity_percentage": ext_pct,
                       "ai_generated_probability": ai.get("probability")},
            "peer_matches": peer_list,
            "multimodal_matches": mm_list,
            "external_matches": ext_list,
            "ai_analysis": ai,
            "boilerplate_excluded": boiler_list,
        })
        base["explanations"] = self._narrative(base)
        return base

    @staticmethod
    def _dedupe(matches: list[Match]) -> list[Match]:
        """Keep the strongest match per region: drop a match if its query sentences overlap
        >50% with an already-kept match against the same source document."""
        kept: list[Match] = []
        for m in sorted(matches, key=lambda x: -x.score):
            qs = set(m.query.sentence_idx)
            if any(k.source.doc_id == m.source.doc_id and
                   len(qs & set(k.query.sentence_idx)) > 0.5 * len(qs) for k in kept):
                continue
            kept.append(m)
        return sorted(kept, key=lambda x: (x.source.doc_id, x.query.sentence_idx[0]))

    @staticmethod
    def _seg_loc(seg) -> dict:
        from .models import fmt_ts
        if seg.start is not None:
            return {"timestamp": f"{fmt_ts(seg.start)} - {fmt_ts(seg.end or seg.start)}"}
        return {"page": seg.page}

    def _align(self, m: Match, q_doc: Document, src_doc: Document, top: int = 3) -> list[dict]:
        """Sentence-level alignment inside a matched chunk pair: for each query sentence the
        most similar source sentence, so reviewers see the exact offending line and its own
        page / timestamp rather than a whole chunk."""
        qs = [q_doc.segments[i] for i in m.query.sentence_idx]
        ss = [src_doc.segments[i] for i in m.source.sentence_idx]
        if not qs or not ss:
            return []
        sims = self.embedder.encode([x.text for x in qs]) @ self.embedder.encode([x.text for x in ss]).T
        pairs = []
        for qi in range(len(qs)):
            si = int(np.argmax(sims[qi]))
            pairs.append((float(sims[qi, si]), qs[qi], ss[si]))
        pairs.sort(key=lambda x: -x[0])
        out = []
        for score, q, src in pairs[:top]:
            ql, sl = self._seg_loc(q), self._seg_loc(src)
            out.append({"similarity": round(score, 3), "query_sentence": q.text, "matched_sentence": src.text,
                        **{f"query_{k}": v for k, v in ql.items()}, **{f"matched_{k}": v for k, v in sl.items()}})
        return out

    def _explain(self, m: Match, src_doc: Document, q_doc: Document, external: bool = False) -> dict:
        jac = jaccard(m.query.text, m.source.text)
        lev = levenshtein_ratio(m.query.text, m.source.text)
        if jac >= 0.5:
            kind = "verbatim copy"
        elif jac >= 0.15:
            kind = "near-verbatim (light edits)"
        else:
            kind = "semantic paraphrase (low word overlap)"
        q_loc, s_loc = m.query.location(), m.source.location()
        location = {f"query_{k}": v for k, v in q_loc.items()}
        location.update({f"matched_{k}": v for k, v in s_loc.items()})
        entry = {
            ("matched_source" if external else "matched_file"): m.source.doc_id,
            "similarity_score": round(m.score, 3),
            "match_type": kind,
            "lexical_overlap": {"trigram_jaccard": round(jac, 3), "char_ratio": round(lev, 3)},
            "query_passage": m.query.text,
            "matched_passage": m.source.text,
            "location": location,
            "aligned_sentences": self._align(m, q_doc, src_doc),
        }
        if external:
            entry["source_url"] = src_doc.meta.get("url", "local reference corpus")
        return entry

    def _risk(self, peer, ext, ai) -> str:
        cfg = self.cfg
        sim = max(peer, ext)
        if sim >= cfg.high_similarity_pct or (ai is not None and ai >= cfg.high_ai_pct):
            return "HIGH"
        if sim >= cfg.medium_similarity_pct or (ai is not None and ai >= cfg.medium_ai_pct):
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _narrative(r: dict) -> str:
        s = r["scores"]
        parts = []
        sources = defaultdict(list)
        for m in r["peer_matches"] + r["multimodal_matches"]:
            loc = m["location"]
            sources[m["matched_file"]].append(loc.get("query_page") or loc.get("query_timestamp"))
        if sources:
            desc = "; ".join(f"{f} (query at {', '.join(str(x) for x in sorted(set(map(str, locs))))})"
                             for f, locs in sources.items())
            parts.append(f"{s['peer_similarity_percentage']}% of original (non-template) text overlaps "
                         f"with other submissions: {desc}.")
            kinds = {m["match_type"] for m in r["peer_matches"] + r["multimodal_matches"]}
            if any("paraphrase" in k for k in kinds):
                parts.append("At least one overlap is a semantic paraphrase: wording differs but the "
                             "arguments match, so exact-string tools would miss it.")
            if r["multimodal_matches"]:
                parts.append("Cross-modal overlap found between a video transcript and a written "
                             "submission; check the timestamps against the page reference.")
            parts.append("Note: similarity is symmetric, so the report cannot tell who copied whom; "
                         "use submission timestamps to decide.")
        else:
            parts.append("No significant overlap with other submissions.")
        if r["external_matches"]:
            srcs = sorted({m["matched_source"] for m in r["external_matches"]})
            parts.append(f"{s['external_similarity_percentage']}% matches external reference sources "
                         f"({', '.join(srcs)}) without being original analysis.")
        ai = s["ai_generated_probability"]
        if ai is not None:
            level = "high" if ai >= 80 else "elevated" if ai >= 60 else "low"
            parts.append(f"AI-generation indicator is {level} ({ai}%); this is a stylometric signal, "
                         "not proof, and should only prompt a closer read.")
        if r["boilerplate_excluded"]:
            parts.append(f"{len(r['boilerplate_excluded'])} sentences were recognised as shared "
                         "template/prompt text (present in 3+ submissions) and excluded from scoring.")
        return " ".join(parts)
