from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
CLASSIFIER_DIR = MODELS / "classifier"
TOPIC_DIR = MODELS / "topic_model"

RISK_LABELS = ["fraud", "money_laundering", "sanctions_violation"]
ALL_LABELS = ["fraud", "money_laundering", "sanctions_violation", "none"]


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, CLASSIFIER_DIR, TOPIC_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return ["none"]
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = [part.strip() for part in text.split("|") if part.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    labels = [label for label in parsed if label in ALL_LABELS]
    if not labels:
        labels = ["none"]
    if "none" in labels and len(labels) > 1:
        labels = [label for label in labels if label != "none"]
    return labels


def labels_to_json(labels: Iterable[str]) -> str:
    values = list(labels)
    if not values:
        values = ["none"]
    if "none" in values and len(values) > 1:
        values = [label for label in values if label != "none"]
    return json.dumps(values)


def read_articles(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_RAW / "articles.csv"
    df = pd.read_csv(path)
    df["labels_list"] = df["labels"].apply(parse_labels)
    df["date"] = pd.to_datetime(df["date"])
    return df


def label_matrix(labels_series: pd.Series) -> pd.DataFrame:
    rows = []
    for labels in labels_series:
        parsed = parse_labels(labels)
        rows.append({label: int(label in parsed) for label in ALL_LABELS})
    return pd.DataFrame(rows)


def label_signature(labels: Iterable[str]) -> str:
    parsed = parse_labels(list(labels))
    return "|".join(label for label in ALL_LABELS if label in parsed)
