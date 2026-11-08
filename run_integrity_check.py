#!/usr/bin/env python3
"""Multimodal submission integrity checker.

    python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

from integrity.config import Config
from integrity.engine import IntegrityEngine, load_corpus, load_reference
from integrity.reporter import write_reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=Path("sample_submissions"))
    ap.add_argument("--output_dir", type=Path, default=Path("reports"))
    ap.add_argument("--reference_dir", type=Path, default=Path("reference_corpus"),
                    help="Local 'web' knowledge base for external similarity")
    ap.add_argument("--embedding_backend", choices=["auto", "sbert", "tfidf"], default="auto")
    ap.add_argument("--asr_backend", choices=["auto", "whisper", "pocketsphinx"], default="auto")
    ap.add_argument("--whisper_model", default="base", help="tiny | base | small ...")
    ap.add_argument("--ai_detector", choices=["heuristic", "roberta"], default="heuristic")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore")
    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    cfg = Config(embedding_backend=args.embedding_backend, asr_backend=args.asr_backend,
                 whisper_model=args.whisper_model)
    t0 = time.time()
    docs = load_corpus(args.input_dir, cfg)
    refs = load_reference(args.reference_dir, cfg)
    engine = IntegrityEngine(cfg, ai_mode=args.ai_detector)
    result = engine.run(docs, refs)
    result["run_info"]["asr_backends"] = sorted({d.meta.get("asr_backend") for d in docs
                                                 if d.meta.get("asr_backend")})
    result["run_info"]["runtime_s"] = round(time.time() - t0, 2)
    write_reports(result, args.output_dir)

    print(f"Processed {len(docs)} files in {result['run_info']['runtime_s']}s "
          f"(embeddings: {result['run_info']['embedding_model']}). Reports -> {args.output_dir}/")
    for r in sorted(result["documents"].values(), key=lambda r: r["document_id"]):
        s = r["scores"]
        print(f"  {r['total_integrity_risk']:<20} {r['document_id']:<32} peer={s['peer_similarity_percentage']} "
              f"ext={s['external_similarity_percentage']} ai={s['ai_generated_probability']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
