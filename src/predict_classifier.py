from __future__ import annotations

import math
import re
from functools import lru_cache

from utils import ALL_LABELS, CLASSIFIER_DIR


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if CLASSIFIER_DIR.exists() and (CLASSIFIER_DIR / "config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR)
            model.eval()
            return "transformer", (tokenizer, model, torch)
    except Exception:
        pass

    try:
        import joblib

        baseline = CLASSIFIER_DIR / "baseline_tfidf_logreg.joblib"
        if baseline.exists():
            return "baseline", joblib.load(baseline)
    except Exception:
        pass

    return "heuristic", None


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _heuristic(text: str) -> dict[str, float]:
    lower = text.lower()
    cues = {
        "fraud": ["fraud", "false", "misled", "inflated", "restatement", "altered", "backdated", "accounting"],
        "money_laundering": ["launder", "suspicious", "shell", "layered", "beneficial owner", "transfers", "aml"],
        "sanctions_violation": ["sanction", "restricted", "export control", "screening", "blocked", "embargo"],
    }
    scores = {}
    for label, terms in cues.items():
        hits = sum(1 for term in terms if term in lower)
        scores[label] = min(0.95, 0.08 + hits * 0.22)
    scores["none"] = max(0.02, 0.85 - max(scores[label] for label in cues))
    return scores


def classify(text: str) -> dict[str, float]:
    """Return label probabilities for one document."""
    kind, model = _load_model()
    if kind == "baseline":
        probs = model.predict_proba([text])
        return {label: float(probs[0][i]) for i, label in enumerate(ALL_LABELS)}
    if kind == "transformer":
        tokenizer, clf, torch = model
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = clf(**enc).logits.detach().cpu().tolist()[0]
        probs = [_sigmoid(value) for value in logits]
        return {label: float(probs[i]) for i, label in enumerate(ALL_LABELS)}
    cleaned = re.sub(r"\s+", " ", text.strip())
    return _heuristic(cleaned)
