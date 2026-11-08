"""AI-generated text likelihood.

Default: a transparent, deterministic stylometric heuristic (no model download), combining
four signals that are well documented in the AI-text-detection literature:

1. LLM discourse markers per 100 words ("furthermore", "it is important to note", "delve",
   "landscape", "pivotal", "in conclusion", ...). Assistant-style prose over-uses them.
2. Burstiness: coefficient of variation of sentence length. Humans mix short and long
   sentences; LLM prose is unusually uniform.
3. Specificity: numbers / quantities per 100 words. Generic LLM essays rarely commit to
   concrete figures; real case analyses are full of them.
4. Lexical sophistication: share of long (>=9 char) words.

Each feature is z-scored against fixed reference means/stds (documented below, calibrated
by hand on human-written case-study prose, NOT on the evaluation set) and passed through a
logistic function. This is an *indicator*, not proof - reports say so explicitly.

Optional: ``--ai_detector roberta`` uses the Hugging Face model
``openai-community/roberta-base-openai-detector`` and averages it with the heuristic.
"""
from __future__ import annotations

import logging
import math
import re
import statistics

from .textnorm import split_sentences, tokens

log = logging.getLogger(__name__)

MARKERS = [
    "furthermore", "moreover", "additionally", "in conclusion", "it is important to note",
    "it is crucial", "it is essential", "delve", "landscape", "pivotal", "holistic",
    "multifaceted", "leverag", "foster", "navigate the complexities", "ever-evolving",
    "rapidly evolving", "in today's", "plays a crucial role", "plays a pivotal role",
    "comprehensive approach", "transformative", "unlock", "stakeholders", "seamless",
    "robust", "a testament to", "not only", "overall", "ultimately", "meaningful",
]
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "fourteen", "fifteen", "twenty", "thirty", "forty", "fifty",
                "sixty", "ninety", "hundred", "thousand", "million", "billion", "percent",
                "half", "quarter", "rupees", "dollars", "degrees", "kilometres", "minutes",
                "days", "months"}

# reference distribution for human case-study prose: (mean, std, weight)
# weight sign: + pushes toward "AI", - pushes toward "human"
REFERENCE = {
    "marker_density": (0.5, 0.8, +1.4),
    "burstiness": (0.45, 0.15, -0.6),
    "specificity": (3.5, 2.0, -0.9),
    "long_word_ratio": (0.14, 0.05, +0.6),
}
BIAS = -1.0
MIN_WORDS = 60


def stylometric_features(text: str) -> dict | None:
    words = tokens(text)
    if len(words) < MIN_WORDS:
        return None
    lower = " ".join(words)
    sents = split_sentences(text)
    lengths = [len(s.split()) for s in sents] or [len(words)]
    n100 = len(words) / 100.0
    markers = sum(lower.count(m) for m in MARKERS)
    numbers = sum(1 for w in words if w in NUMBER_WORDS or re.fullmatch(r"\d[\d.,%]*", w))
    burst = statistics.pstdev(lengths) / statistics.mean(lengths) if len(lengths) > 1 else REFERENCE["burstiness"][0]
    return {
        "marker_density": round(markers / n100, 3),
        "burstiness": round(burst, 3),
        "specificity": round(numbers / n100, 3),
        "long_word_ratio": round(sum(len(w) >= 9 for w in words) / len(words), 3),
        "n_words": len(words),
        "n_sentences": len(sents),
    }


def heuristic_probability(feats: dict) -> tuple[float, dict]:
    z = BIAS
    contrib = {}
    for name, (mean, std, weight) in REFERENCE.items():
        c = weight * (feats[name] - mean) / std
        c = max(-3.0, min(3.0, c))        # cap any single feature's influence
        contrib[name] = round(c, 2)
        z += c
    return 1.0 / (1.0 + math.exp(-z)), contrib


class AIDetector:
    def __init__(self, mode: str = "heuristic"):
        self.mode = mode
        self._clf = None
        if mode == "roberta":
            try:
                from transformers import pipeline
                self._clf = pipeline("text-classification",
                                     model="openai-community/roberta-base-openai-detector")
            except Exception as exc:  # noqa: BLE001
                log.warning("RoBERTa detector unavailable (%s); using heuristic only", exc)
                self.mode = "heuristic"

    def score(self, text: str, modality: str) -> dict:
        feats = stylometric_features(text)
        if feats is None:
            return {"probability": None, "method": self.mode,
                    "note": f"Insufficient text (<{MIN_WORDS} words) for a reliable estimate"}
        prob, contrib = heuristic_probability(feats)
        out = {"method": "stylometric-heuristic", "features": feats, "contributions": contrib}
        if self._clf is not None:
            res = self._clf(text[:2000], truncation=True)[0]
            p_model = res["score"] if res["label"].lower() in ("fake", "label_0") else 1 - res["score"]
            out["roberta_probability"] = round(100 * p_model, 1)
            prob = (prob + p_model) / 2
            out["method"] = "stylometric-heuristic + roberta-base-openai-detector (mean)"
        if modality == "video":
            out["note"] = "Computed on an ASR transcript (no reliable punctuation); low confidence"
        out["probability"] = round(100 * prob, 1)
        return out
