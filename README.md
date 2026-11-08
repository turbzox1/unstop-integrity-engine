# Multimodal Submission Integrity & Plagiarism Detection Engine

A CLI pipeline that ingests PDF, DOCX and MP4 submissions, turns them into page- or
timestamp-tagged text chunks, indexes them as vectors, and produces an **explainable audit
report** for each file. Each report has three scores (internal peer similarity, external/web
match, AI-generation likelihood) and the **exact matching passages** with their locations, so
a reviewer can check every claim in seconds.

```
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.9+
pip install -r requirements.txt                          # ffmpeg must also be on PATH
python scripts/download_models.py                        # optional: pre-cache MiniLM + Whisper
./run_all.sh                                             # pipeline + evaluation, single command
python -m pytest -q                                      # unit tests (offline, ~1 s)
```

Outputs in `reports/`:

* `<file>.json` follows the schema in the brief, plus `external_matches`, `aligned_sentences`,
  `ai_analysis`, `boilerplate_excluded`, `warnings`.
* `<file>.md` is the human-readable report with side-by-side passages and locations.
* `summary.md` / `summary.json` hold the batch table plus the pairwise coverage matrix.
* `evaluation_metrics.json` contains the scores against ground truth (also written into `EVALUATION.md`).

Useful flags: `--embedding_backend {auto,sbert,tfidf}`, `--asr_backend {auto,whisper,pocketsphinx}`,
`--whisper_model tiny|base|small`, `--ai_detector {heuristic,roberta}`, `--reference_dir`, `-v`.

## Architecture

```
input_dir/ ──► extractors ──► Document(segments: text + page | [start,end]s)
  .pdf  PyMuPDF blocks, header/footer margin strip, repeated-line removal, per-page sentences
  .docx python-docx paragraphs + tables, explicit page breaks (else ~450 words/page estimate)
  .mp4  ffprobe (audio stream?) → ffmpeg 16 kHz mono → volumedetect (silent?) → Whisper
        (fallback PocketSphinx) → timestamped segments
        │
        ▼
boilerplate detector ── sentences shared (near-)verbatim by ≥3 docs → excluded from scoring
        │
        ▼
chunker ── ~60-word sentence-packed chunks, 1-sentence overlap, never cross a page
        │
        ▼
embedder (MiniLM-L6-v2, L2-normalised │ TF-IDF fallback) ──► FAISS IndexFlatIP
        │
        ├─ peer search: top-k per chunk, **same-document hits skipped**, best hit per other doc ≥ τ
        ├─ external search: same, against reference_corpus/ (simulated web knowledge base)
        └─ AI indicator: stylometric heuristic (optional RoBERTa detector)
        │
        ▼
scoring ── coverage % = non-template words inside matched chunks / non-template words
explainer ── per match: passages, cosine, trigram Jaccard, char ratio, match type,
             page/timestamp for both sides, top-3 aligned sentence pairs
reporter ── JSON + Markdown per file, batch summary + N×N coverage matrix
```

Code layout: `integrity/extractors.py`, `chunking.py`, `embeddings.py` (embedder + index),
`engine.py` (matching, scoring, explanations), `ai_detect.py`, `reporter.py`, `config.py`
(every threshold in one place), `textnorm.py`.

## Methodology

**Similarity score.** A chunk counts as "matched" when its best cosine against a chunk from a
*different* document reaches τ (0.80 for MiniLM, as in the brief; 0.45 for the TF-IDF
fallback, whose cosines sit on a lower scale). Peer % is the share of the document's
non-template words that fall inside matched chunks. Measuring word coverage, not average
cosine, keeps one copied paragraph from being diluted and keeps shared vocabulary from
inflating the score. Overlapping matches against the same source are de-duplicated.

**Self-match exclusion.** All chunks share one index. Each query requests `top_k + 40`
neighbours and drops any hit from the same `doc_id` before taking the best hit per source
document, so a document's own chunks cannot crowd out real matches.

**Cross-modal.** Video transcripts go into the same index as written text. A match where
exactly one side is a video is reported under `multimodal_matches`, with `query_timestamp` /
`matched_page` (or the reverse).

**Template discount (Scenario 4).** Sentences are clustered by a normalised exact key and by
word-trigram Jaccard ≥ 0.8. Any cluster that spans at least 3 distinct submissions is treated as
institutional boilerplate (case prompt, instructions, citation lines). It is removed before
chunking and listed in the report under "excluded", so nothing is hidden.

**External.** `reference_corpus/` holds simulated "web" articles (synthetic, with a fake URL
header). They are chunked and indexed like submissions. In production this would be a
search-API retrieval step with this same local matcher behind it.

**AI likelihood.** A deterministic, documented heuristic: LLM discourse-marker density,
burstiness (sentence-length variation), specificity (numbers per 100 words), and long-word
ratio. Each is z-scored against fixed human-prose reference values and passed through a
logistic function. The per-feature contributions are printed in every report. It needs at
least 60 words, otherwise it returns `null` with a note. It is an indicator only (see
EVALUATION.md, failure 5).

**Risk band.** HIGH if max(peer, external) ≥ 40% or AI ≥ 80%. MEDIUM if ≥ 15% or AI ≥ 60%.
Otherwise LOW. Unreadable files get `NEEDS_MANUAL_REVIEW` with null scores.

**Robustness (Scenario 3).** Each extractor catches its own errors and returns a status
(`empty`, `corrupted`, `no_audio`, `silent`, `unsupported`) plus a warning. The batch always
finishes, and each skipped file still gets a report.

## Sample data

`sample_submissions/` contains 14 synthetic files (6 PDF, 5 DOCX, 3 MP4 of 10 to 26 s) plus
`dataset_manifest.json` with the ground truth. All of it is generated by
`scripts/build_sample_data.py` (reportlab, python-docx, espeak-ng + MBROLA TTS, ffmpeg). No
real or copyrighted student data is used. `sub_07_ai_generated.docx` was written by an LLM
(Claude) on purpose.

## Results

See `EVALUATION.md` for the confusion matrix, per-scenario pass/fail, and failure analysis.
With the offline fallbacks (TF-IDF + PocketSphinx), the exact copy, the cross-modal copy and
the web copy are detected with zero false positives across 66 pairs; the template documents
are correctly cleared. The paraphrase pair needs the neural embedding model, which is the
default whenever it can be loaded.

## Assumptions

1. **Chunk size.** About 60 words (about 80 tokens) instead of 200 to 400 tokens, because the
   sample submissions are 1 to 2 pages. Configurable in `config.py`.
2. **Peer % definition.** The share of a submission's own non-template words matched in any
   other submission. A per-source breakdown is in the coverage matrix.
3. **"Web" search** is simulated by a local reference corpus, which the brief permits.
4. **DOCX page numbers** come from explicit page breaks when present, otherwise they are
   estimated at 450 words per page (stated in `extraction_meta`). DOCX has no fixed layout.
5. **Boilerplate** means text shared by at least 3 submissions. Two documents sharing text
   is treated as possible copying.
6. **Direction of copying** is not inferred. Both sides of a pair are flagged, and the
   report says so.
7. **Silent video** means no audio stream, or a max volume below −50 dB. Such files get
   `NEEDS_MANUAL_REVIEW`, not a clean bill.
8. **Fallbacks.** If MiniLM or Whisper cannot be downloaded, the pipeline falls back to
   TF-IDF / PocketSphinx and records this in every report instead of failing.

## Scaling to 10,000 submissions

**Workload.** About 6k documents plus 4k videos of up to 60 s ≈ 70 hours of audio, about
500k chunks, and 500k × 384-d vectors.

**1. Ingestion as an asynchronous job queue.** Uploads land in object storage (S3/GCS), and
each one enqueues a job (SQS / Redis Streams + Celery, or Kafka for very high volume). There
are separate worker pools per stage so the slow stage never blocks the fast ones:
* *Parse workers (CPU, autoscaled):* PDF/DOCX extraction is about 50 to 200 ms per document,
  so 6k documents take a few minutes on 8 vCPUs. Corrupt files go to a dead-letter queue
  with the error, and the batch continues.
* *ASR workers (GPU):* Whisper-base via `faster-whisper` (CTranslate2, int8/fp16) runs at
  about 50 to 100x real-time on one T4/L4, so 70 h of audio takes about 1 to 1.5 GPU-hours
  (about $1 to 2 on spot). CPU-only would be about 70 CPU-hours: still feasible across 32
  cores in about 2 h, but GPU is cheaper per hour of audio. Batch by duration and use VAD to
  skip silence.
* *Embedding workers (GPU, or CPU with ONNX):* MiniLM on one T4 does about 5 to 10k chunks/s,
  so 500k chunks take about 1 to 2 minutes. This is not a bottleneck.

Every stage is idempotent and keyed by a content hash (SHA-256), so retries and duplicate
uploads are free, and exact duplicate files are caught before any ML runs.

**2. Vector store and index.**
* Up to about 1M vectors, FAISS **HNSW** (`M=32, efSearch≈128`) in a service, or a managed
  equivalent (Qdrant, Milvus, pgvector-HNSW), gives >95% recall@10 at about 1 ms per query.
  Memory is about 0.8 GB of vectors plus about 0.25 GB of graph, which fits on one node.
* Flat search stays useful as the ground-truth baseline for measuring HNSW recall, and for
  small per-challenge indexes (<50k vectors), where exact search is already under 10 ms.
* Beyond about 10M vectors (many challenges and years of history), IVF-PQ (`nlist≈4096`,
  PQ64) cuts memory about 16x. Use GPU FAISS for bulk all-pairs jobs.
* Partition by challenge ID (peer comparison only makes sense within a challenge) and keep a
  separate shared index for the external/reference corpus and for templates.
* Metadata (doc, page, timestamp, text) lives in Postgres, keyed by vector ID. The vector DB
  stores only IDs and vectors.

**3. Matching cost.** Naive all-pairs is 500k² = 2.5×10¹¹ similarity calls. With ANN, each
chunk queries top-k=10, so 5M candidate pairs. A coarse **MinHash-LSH** pass first removes
verbatim duplicates and template text cheaply, and an optional cross-encoder re-ranks only
the pairs above about 0.7, typically under 1% of candidates. New submissions are matched
incrementally on arrival (query, then insert), so cost grows linearly, not quadratically.

**4. Boilerplate at scale.** Replace the O(n²) sentence comparison with MinHash-LSH clusters,
and treat any cluster that appears in more than about 2% of a challenge's submissions, or in
an organiser-uploaded template, as boilerplate.

**5. Bottlenecks and costs.**

| Stage | Throughput (1 node) | 10k-batch wall time | Main risk |
|---|---|---|---|
| PDF/DOCX parse | ~50 docs/s on 8 vCPU | ~2 min | scanned PDFs need OCR (10x slower) |
| ASR | 50-100x real-time / T4 | ~1 h | the dominant cost and latency |
| Embedding | ~5k chunks/s / T4 | ~2 min | none |
| ANN search | ~1k QPS / node | ~10 min | recall loss if efSearch is too low |
| Re-rank + reports | ~200 pairs/s / GPU | ~10 min | only if the cross-encoder is enabled |

End-to-end, a 10k batch finishes in about 1.5 h with 1 to 2 GPUs and a few CPU nodes, for
about $5 to 15 of compute. Latency for one new submission is a few seconds, or 10 to 30 s for
a video. Scale-out is horizontal: more ASR workers for the queue that backs up, and index
shards per challenge. Monitor queue depth, per-stage p95 latency, failed-file rate, and
weekly HNSW recall against a flat-search sample.

## AI assistant usage disclosure

I built this project with help from an AI coding assistant (Claude). It helped with
boilerplate code, the synthetic dataset text (and it wrote the deliberately AI-generated
sample), and drafting documentation. I reviewed, ran and tested the pipeline end to end, and I
can explain every design decision and line of code.
