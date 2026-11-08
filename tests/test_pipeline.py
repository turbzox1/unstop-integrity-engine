"""Fast unit tests (TF-IDF backend, no model downloads).  Run: python -m pytest -q"""
from pathlib import Path

from integrity.chunking import chunk_document, detect_boilerplate
from integrity.config import Config
from integrity.engine import IntegrityEngine
from integrity.extractors import extract
from integrity.models import Document, Segment

CFG = Config(embedding_backend="tfidf")


def _doc(doc_id, sentences, page=1):
    return Document(doc_id, doc_id, "text", [Segment(s, page=page) for s in sentences])


def test_empty_and_garbage_files_do_not_crash(tmp_path: Path):
    (tmp_path / "empty.pdf").write_bytes(b"")
    (tmp_path / "garbage.pdf").write_bytes(b"not really a pdf")
    (tmp_path / "garbage.docx").write_bytes(b"PK\x03\x04broken")
    for name in ["empty.pdf", "garbage.pdf", "garbage.docx"]:
        d = extract(tmp_path / name, CFG)
        assert not d.usable and d.warnings


def test_self_matches_are_excluded():
    a = _doc("a.txt", ["The quick brown fox jumps over the lazy dog near the river bank today."] * 3)
    b = _doc("b.txt", ["Completely different text about quarterly revenue and supplier contracts."] * 3)
    res = IntegrityEngine(CFG).run([a, b], [])
    assert res["documents"]["a.txt"]["scores"]["peer_similarity_percentage"] == 0.0
    assert all(m["matched_file"] != "a.txt" for m in res["documents"]["a.txt"]["peer_matches"])


def test_exact_copy_is_detected_with_location():
    text = ["Dark stores are small warehouses closed to the public that exist only to fulfil orders.",
            "They sit in dense neighbourhoods so riders reach customers within a few kilometres."]
    a, b = _doc("a.txt", text, page=2), _doc("b.txt", text, page=1)
    c = _doc("c.txt", ["Our canteen app lets students reserve leftover meals at half price before closing."] * 2)
    res = IntegrityEngine(CFG).run([a, b, c], [])
    r = res["documents"]["a.txt"]
    assert r["scores"]["peer_similarity_percentage"] == 100.0
    assert r["peer_matches"][0]["location"] == {"query_page": 2, "matched_page": 1}
    assert res["documents"]["c.txt"]["scores"]["peer_similarity_percentage"] == 0.0


def test_shared_template_is_discounted():
    prompt = ["Participants must analyse the market and recommend a go to market strategy for the firm.",
              "Cite all external sources using a consistent referencing style in the final report."]
    docs = [_doc(f"d{i}.txt", prompt + [f"Original answer number {i} discusses topic {w} in depth here."])
            for i, w in enumerate(["pricing", "logistics", "hiring", "branding"])]
    boiler = detect_boilerplate(docs, CFG)
    assert all(boiler[d.doc_id] == {0, 1} for d in docs)
    chunks = chunk_document(docs[0], boiler[docs[0].doc_id], CFG)
    assert all("Participants" not in c.text for c in chunks)
