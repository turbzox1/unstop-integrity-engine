"""Sentence-aware chunking with overlap and boilerplate (template) detection."""
from __future__ import annotations

from collections import defaultdict

from .config import Config
from .models import Chunk, Document
from .textnorm import jaccard, normalise_key


def detect_boilerplate(docs: list[Document], cfg: Config) -> dict[str, set[int]]:
    """Return {doc_id: {segment indices}} for sentences shared (near-)verbatim by at least
    ``boilerplate_min_docs`` distinct documents - e.g. a case prompt pasted by every team.

    Exact matches use a normalised key; near-verbatim matches use word-trigram Jaccard.
    O(n^2) over sentences is fine for a prototype; at 10k docs this becomes MinHash-LSH.
    """
    usable = [d for d in docs if d.usable]
    min_docs = max(cfg.boilerplate_min_docs, int(cfg.boilerplate_min_fraction * len(usable)))
    sents = [(d.doc_id, i, s.text) for d in usable for i, s in enumerate(d.segments)
             if len(s.text.split()) >= 5]
    # union-find over near-identical sentences
    parent = list(range(len(sents)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_key = defaultdict(list)
    for n, (_, _, text) in enumerate(sents):
        by_key[normalise_key(text)].append(n)
    for group in by_key.values():
        for n in group[1:]:
            parent[find(n)] = find(group[0])
    for a in range(len(sents)):
        for b in range(a + 1, len(sents)):
            if sents[a][0] != sents[b][0] and find(a) != find(b) \
                    and jaccard(sents[a][2], sents[b][2]) >= cfg.boilerplate_jaccard:
                parent[find(b)] = find(a)

    clusters = defaultdict(list)
    for n in range(len(sents)):
        clusters[find(n)].append(n)
    result: dict[str, set[int]] = defaultdict(set)
    for members in clusters.values():
        if len({sents[m][0] for m in members}) >= min_docs:
            for m in members:
                result[sents[m][0]].add(sents[m][1])
    return dict(result)


def chunk_document(doc: Document, skip: set[int], cfg: Config, start_id: int = 0) -> list[Chunk]:
    """Greedy sentence packing to ~chunk_target_words with N-sentence overlap.
    Chunks never cross a page boundary so every match has an exact page reference;
    video chunks carry [start, end] timestamps of their first/last sentence."""
    chunks: list[Chunk] = []
    target = cfg.video_chunk_target_words if doc.modality == "video" else cfg.chunk_target_words
    idxs = [i for i in range(len(doc.segments)) if i not in skip]

    # split into runs that stay on one page and are contiguous in the source
    runs, cur = [], []
    for i in idxs:
        if cur and (doc.segments[i].page != doc.segments[cur[-1]].page or i != cur[-1] + 1):
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)

    for run in runs:
        pos = 0
        while pos < len(run):
            words, end = 0, pos
            while end < len(run) and (words < target or end == pos):
                words += len(doc.segments[run[end]].text.split())
                end += 1
            members = run[pos:end]
            segs = [doc.segments[i] for i in members]
            text = " ".join(s.text for s in segs)
            if len(text.split()) >= cfg.min_chunk_words:
                chunks.append(Chunk(
                    chunk_id=start_id + len(chunks), doc_id=doc.doc_id, modality=doc.modality,
                    text=text, sentence_idx=members, page=segs[0].page,
                    start=segs[0].start, end=segs[-1].end))
            if end >= len(run):
                break
            pos = max(end - cfg.chunk_overlap_sentences, pos + 1)
    return chunks
