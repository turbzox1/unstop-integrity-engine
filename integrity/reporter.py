"""JSON + Markdown report writers (per document and batch summary)."""
from __future__ import annotations

import json
from pathlib import Path

RISK_ICON = {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🟢 LOW",
             "NEEDS_MANUAL_REVIEW": "⚪ NEEDS MANUAL REVIEW"}


def _fmt(v, suffix="%"):
    return "n/a" if v is None else f"{v}{suffix}"


def _loc(loc: dict, side: str) -> str:
    if f"{side}_timestamp" in loc:
        return f"⏱ {loc[f'{side}_timestamp']}"
    return f"p. {loc.get(f'{side}_page')}"


def _match_block(m: dict, n: int, key: str = "matched_file") -> str:
    loc = m["location"]
    lines = [
        f"**{n}. `{m[key]}`** - cosine **{m['similarity_score']}** - {m['match_type']} "
        f"(trigram Jaccard {m['lexical_overlap']['trigram_jaccard']}, "
        f"char ratio {m['lexical_overlap']['char_ratio']})",
        "",
        f"| This submission ({_loc(loc, 'query')}) | Matched source ({_loc(loc, 'matched')}) |",
        "|---|---|",
        f"| {m['query_passage']} | {m['matched_passage']} |",
    ]
    if m.get("aligned_sentences"):
        lines += ["", "Closest sentence pairs (verify these first):", ""]
        for a in m["aligned_sentences"]:
            ql = a.get("query_timestamp") or f"p. {a.get('query_page')}"
            sl = a.get("matched_timestamp") or f"p. {a.get('matched_page')}"
            lines.append(f"- `{a['similarity']}` [{ql}] \"{a['query_sentence']}\" ⇄ [{sl}] \"{a['matched_sentence']}\"")
    if m.get("source_url"):
        lines.append(f"\nSource: {m['source_url']}")
    return "\n".join(lines) + "\n"


def document_markdown(r: dict) -> str:
    s = r["scores"]
    out = [f"# Integrity report: `{r['document_id']}`", "",
           f"**Risk: {RISK_ICON[r['total_integrity_risk']]}** · modality: {r['modality']} · "
           f"status: {r['status']}", "",
           "| Internal peer similarity | External match | AI-content likelihood |",
           "|---|---|---|",
           f"| {_fmt(s['peer_similarity_percentage'])} | {_fmt(s['external_similarity_percentage'])} "
           f"| {_fmt(s['ai_generated_probability'])} |", "",
           "## Summary", "", r["explanations"], ""]
    if r["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    for title, key, field in [("Peer matches", "peer_matches", "matched_file"),
                              ("Cross-modal matches (video ⇄ document)", "multimodal_matches", "matched_file"),
                              ("External / web reference matches", "external_matches", "matched_source")]:
        if r.get(key):
            out += [f"## {title}", ""]
            out += [_match_block(m, i, field) for i, m in enumerate(r[key], 1)]
    ai = r.get("ai_analysis")
    if ai and ai.get("features"):
        f, c = ai["features"], ai["contributions"]
        out += ["## AI-generation signals", "",
                f"Method: {ai['method']}. Positive contribution pushes toward 'AI-like'.", "",
                "| Signal | Value | Contribution |", "|---|---|---|"]
        out += [f"| {k} | {f[k]} | {c[k]:+} |" for k in c]
        if ai.get("note"):
            out += ["", f"_{ai['note']}_"]
        out.append("")
    elif ai and ai.get("note"):
        out += ["## AI-generation signals", "", ai["note"], ""]
    if r.get("boilerplate_excluded"):
        out += ["<details><summary>Template / prompt text excluded from scoring "
                f"({len(r['boilerplate_excluded'])} sentences)</summary>", ""]
        out += [f"- ({b.get('page') and 'p. ' + str(b['page']) or b.get('timestamp')}) {b['text']}"
                for b in r["boilerplate_excluded"]]
        out += ["", "</details>", ""]
    return "\n".join(out)


def summary_markdown(result: dict) -> str:
    info = result["run_info"]
    docs = result["documents"]
    out = ["# Batch integrity summary", "",
           f"Embedding: `{info['embedding_model']}` ({info['embedding_backend']}), index: "
           f"`{info['vector_index']}`, peer threshold {info['peer_threshold']}, external threshold "
           f"{info['external_threshold']}, AI detector: {info['ai_detector']}. "
           f"{info['n_usable']}/{info['n_documents']} documents usable, {info['n_chunks']} chunks, "
           f"{info['n_reference_docs']} reference sources.", "",
           "| Document | Risk | Peer % | External % | AI % | Top match | Status |",
           "|---|---|---|---|---|---|---|"]
    order = {"HIGH": 0, "MEDIUM": 1, "NEEDS_MANUAL_REVIEW": 2, "LOW": 3}
    for r in sorted(docs.values(), key=lambda r: (order[r["total_integrity_risk"]], r["document_id"])):
        s = r["scores"]
        ms = r["peer_matches"] + r["multimodal_matches"]
        top = max(ms, key=lambda m: m["similarity_score"]) if ms else None
        top_s = f"{top['matched_file']} ({top['similarity_score']})" if top else "-"
        out.append(f"| [{r['document_id']}]({r['document_id']}.md) | {RISK_ICON[r['total_integrity_risk']]} "
                   f"| {_fmt(s['peer_similarity_percentage'], '')} | {_fmt(s['external_similarity_percentage'], '')} "
                   f"| {_fmt(s['ai_generated_probability'], '')} | {top_s} | {r['status']} |")
    m = result["pair_matrix"]
    names = list(m)
    short = {n: n.split("_")[0] + "_" + n.split("_")[1] if "_" in n else n for n in names}
    out += ["", "## Pairwise peer-coverage matrix", "",
            "Cell = % of the **row** document's non-template words matched in the **column** document.", "",
            "| row \\ col | " + " | ".join(short[n] for n in names) + " |",
            "|---" * (len(names) + 1) + "|"]
    for a in names:
        cells = ["·" if m[a][b] is None else ("**%s**" % m[a][b] if m[a][b] >= 15 else str(m[a][b]))
                 for b in names]
        out.append(f"| {short[a]} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def write_reports(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, r in result["documents"].items():
        (out_dir / f"{doc_id}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"{doc_id}.md").write_text(document_markdown(r), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(
        {"run_info": result["run_info"], "pair_matrix": result["pair_matrix"],
         "documents": {k: {"risk": v["total_integrity_risk"], "status": v["status"], **v["scores"]}
                       for k, v in result["documents"].items()}}, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(result), encoding="utf-8")
