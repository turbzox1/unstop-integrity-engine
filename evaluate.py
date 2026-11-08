#!/usr/bin/env python3
"""Score the pipeline against sample_submissions/dataset_manifest.json.

    python evaluate.py --reports_dir ./reports --input_dir ./sample_submissions

Writes reports/evaluation_metrics.json and rewrites the auto-generated block between the
<!-- METRICS:START --> / <!-- METRICS:END --> markers in EVALUATION.md, so the numbers
in the write-up always come from the latest run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

FLAG_PCT = 15.0      # a pair is "flagged" when either side has >= 15% coverage by the other
AI_FLAG = 60.0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", type=Path, default=Path("reports"))
    ap.add_argument("--input_dir", type=Path, default=Path("sample_submissions"))
    ap.add_argument("--evaluation_md", type=Path, default=Path("EVALUATION.md"))
    args = ap.parse_args()

    manifest = json.loads((args.input_dir / "dataset_manifest.json").read_text())
    summary = json.loads((args.reports_dir / "summary.json").read_text())
    reports = {f: json.loads((args.reports_dir / f"{f}.json").read_text()) for f in manifest["files"]}
    matrix = summary["pair_matrix"]
    info = summary["run_info"]

    # ---------------------------------------------------------------- 1. pair detection
    truth = {frozenset((p["a"], p["b"])): p["type"] for p in manifest["pairs"]}
    usable = sorted(matrix)
    rows = []
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(usable, 2):
        score = max(matrix[a][b] or 0, matrix[b][a] or 0)
        pred = score >= FLAG_PCT
        gt = frozenset((a, b)) in truth
        tp += pred and gt
        fp += pred and not gt
        fn += (not pred) and gt
        tn += (not pred) and not gt
        if pred or gt:
            rows.append((a, b, truth.get(frozenset((a, b)), "none"), score, pred))
    # ground-truth pairs that involve an unusable file count as misses
    for pair, typ in truth.items():
        if not pair <= set(usable):
            fn += 1
            rows.append((*sorted(pair), typ, None, False))
    pair_p, pair_r, pair_f = prf(tp, fp, fn)

    # ---------------------------------------------------------------- 2. per-scenario
    def best_chunk(doc, other):
        r = reports.get(doc, {})
        ms = [m for m in r.get("peer_matches", []) + r.get("multimodal_matches", [])
              if m["matched_file"] == other]
        return max(ms, key=lambda m: m["similarity_score"]) if ms else None

    scen = []
    m = best_chunk("pitch_01_reads_sub06.mp4", "sub_06_cold_chain.pdf")
    scen.append(("1 Cross-modal duplication", m is not None and "query_timestamp" in m["location"],
                 f"best match cos={m['similarity_score']} at {m['location']}" if m else "no match"))
    m = best_chunk("sub_05_ev_paraphrase.docx", "sub_04_ev_original.docx")
    scen.append(("2 Paraphrased content", m is not None and m["lexical_overlap"]["trigram_jaccard"] < 0.2,
                 f"cos={m['similarity_score']}, Jaccard={m['lexical_overlap']['trigram_jaccard']}, "
                 f"char ratio={m['lexical_overlap']['char_ratio']}" if m else
                 f"not flagged (doc coverage {matrix.get('sub_05_ev_paraphrase.docx', {}).get('sub_04_ev_original.docx')}%)"))
    bad = [c for c in manifest["corrupted"] if reports[c]["status"] == "ok"]
    scen.append(("3 Corrupted / empty", not bad and len(reports) == len(manifest["files"]),
                 ", ".join(f"{c}: {reports[c]['status']}" for c in manifest["corrupted"])))
    tmpl = manifest["template_docs"]
    orig_tmpl = [d for d in tmpl if manifest["files"][d]["label"] == "original"
                 and d != "sub_01_original_a.pdf"]
    tmpl_fp = [d for d in orig_tmpl if reports[d]["total_integrity_risk"] in ("HIGH", "MEDIUM")]
    excl = {d: len(reports[d]["boilerplate_excluded"]) for d in tmpl}
    scen.append(("4 Benign common template", not tmpl_fp,
                 f"template sentences excluded per doc: {excl}; originals flagged: {tmpl_fp or 'none'}"))
    scen.append(("5 Batch run", len(list(args.reports_dir.glob('*.md'))) >= len(manifest["files"]),
                 f"{info['n_documents']} files, {info.get('runtime_s')}s, summary matrix written"))

    # ---------------------------------------------------------------- 3. external + AI
    ext_truth = {e["doc"] for e in manifest["external"]}
    ext_pred = {d for d, r in reports.items() if (r["scores"]["external_similarity_percentage"] or 0) >= FLAG_PCT}
    ext_p, ext_r, _ = prf(len(ext_pred & ext_truth), len(ext_pred - ext_truth), len(ext_truth - ext_pred))
    ai_truth = set(manifest["ai_generated"])
    ai_scored = {d: r["scores"]["ai_generated_probability"] for d, r in reports.items()
                 if r["scores"]["ai_generated_probability"] is not None}
    ai_pred = {d for d, p in ai_scored.items() if p >= AI_FLAG}
    ai_tp, ai_fp = len(ai_pred & ai_truth), len(ai_pred - ai_truth)
    ai_fn = len(ai_truth - ai_pred)
    ai_tn = len(set(ai_scored) - ai_pred - ai_truth)
    ai_p, ai_r, _ = prf(ai_tp, ai_fp, ai_fn)

    # ---------------------------------------------------------------- 4. risk vs label
    risk_rows = [(d, manifest["files"][d]["label"], r["total_integrity_risk"],
                  r["scores"]["peer_similarity_percentage"], r["scores"]["external_similarity_percentage"],
                  r["scores"]["ai_generated_probability"]) for d, r in sorted(reports.items())]

    metrics = {"run_info": info, "pair_detection": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "precision": pair_p, "recall": pair_r, "f1": pair_f},
               "external": {"precision": ext_p, "recall": ext_r},
               "ai_detection": {"tp": ai_tp, "fp": ai_fp, "fn": ai_fn, "tn": ai_tn,
                                "precision": ai_p, "recall": ai_r},
               "scenarios": [{"scenario": s, "pass": bool(ok), "detail": d} for s, ok, d in scen]}
    (args.reports_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------------------------------------------------------------- markdown block
    L = [f"_Auto-generated by `evaluate.py`. Embeddings: `{info['embedding_model']}` "
         f"({info['embedding_backend']}), ASR: {', '.join(info.get('asr_backends') or ['-'])}, "
         f"index: `{info['vector_index']}`, peer threshold {info['peer_threshold']}, "
         f"pair flag = coverage >= {FLAG_PCT}%, AI flag = probability >= {AI_FLAG}%._", "",
         "### Document-pair plagiarism detection", "",
         f"All {tp + fp + fn + tn} unordered pairs of submissions (usable files + pairs touching unusable files).", "",
         "| | Predicted copied | Predicted clean |", "|---|---|---|",
         f"| **Actually copied** | TP = {tp} | FN = {fn} |",
         f"| **Actually clean** | FP = {fp} | TN = {tn} |", "",
         f"**Precision {pair_p} · Recall {pair_r} · F1 {pair_f}**", "",
         "| Pair | Ground truth | Max coverage % | Flagged |", "|---|---|---|---|"]
    L += [f"| {a} ⇄ {b} | {t} | {s} | {'✅' if p else '❌'} |" for a, b, t, s, p in rows]
    L += ["", "### Scenario checklist", "", "| Scenario | Result | Evidence |", "|---|---|---|"]
    L += [f"| {s} | {'PASS' if ok else 'FAIL'} | {d} |" for s, ok, d in scen]
    L += ["", "### External match and AI-generation detection", "",
          f"- External/web copy: precision {ext_p}, recall {ext_r} (flag = external coverage >= {FLAG_PCT}%).",
          f"- AI-generated: TP {ai_tp}, FP {ai_fp}, FN {ai_fn}, TN {ai_tn} over {len(ai_scored)} scorable "
          f"documents; precision {ai_p}, recall {ai_r}. Only one positive exists, so treat this as a smoke test.", "",
          "### Per-document outcome", "",
          "| Document | Ground-truth label | Risk | Peer % | External % | AI % |", "|---|---|---|---|---|---|"]
    L += [f"| {d} | {lab} | {risk} | {p} | {e} | {a} |" for d, lab, risk, p, e, a in risk_rows]
    block = "\n".join(L)

    if args.evaluation_md.exists():
        text = args.evaluation_md.read_text()
        new = re.sub(r"<!-- METRICS:START -->.*<!-- METRICS:END -->",
                     f"<!-- METRICS:START -->\n{block}\n<!-- METRICS:END -->", text, flags=re.S)
        args.evaluation_md.write_text(new)
    print(block)


if __name__ == "__main__":
    main()
