from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from utils import ALL_LABELS, CLASSIFIER_DIR, DATA_PROCESSED, MODELS, ensure_dirs, label_matrix, read_articles


def split_data(df: pd.DataFrame, seed: int):
    sig = df["labels_list"].apply(lambda labels: "|".join(sorted(labels)))
    stratify = sig if sig.value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=stratify)
    temp_sig = temp_df["labels_list"].apply(lambda labels: "|".join(sorted(labels)))
    temp_stratify = temp_sig if temp_sig.value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_stratify)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def metrics_table(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> tuple[pd.DataFrame, dict]:
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for i, label in enumerate(ALL_LABELS):
        p, r, f1, support = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        rows.append({
            "label": label, "precision": p, "recall": r, "f1": f1,
            "support": int(y_true[:, i].sum()), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    summary = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return pd.DataFrame(rows), summary


def train_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    y_train = label_matrix(train_df["labels_list"])[ALL_LABELS].values
    y_test = label_matrix(test_df["labels_list"])[ALL_LABELS].values
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))),
    ])
    pipe.fit(train_df["text"], y_train)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        probs = pipe.predict_proba(test_df["text"])
    else:
        probs = pipe.decision_function(test_df["text"])
        probs = 1 / (1 + np.exp(-probs))
    table, summary = metrics_table(y_test, probs)
    joblib.dump(pipe, CLASSIFIER_DIR / "baseline_tfidf_logreg.joblib")
    return table, summary, probs


def train_transformer(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, args):
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        return None, {"error": f"Transformer dependencies unavailable: {exc}"}, None

    class TextDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, tokenizer):
            self.texts = frame["text"].tolist()
            self.labels = label_matrix(frame["labels_list"])[ALL_LABELS].values.astype("float32")
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=args.max_length)
            enc = {key: torch.tensor(value) for key, value in enc.items()}
            enc["labels"] = torch.tensor(self.labels[idx])
            return enc

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(ALL_LABELS),
            problem_type="multi_label_classification",
        )
    except Exception as exc:
        return None, {"error": f"Could not load pretrained model '{args.model_name}': {exc}"}, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(TextDataset(train_df, tokenizer), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df, tokenizer), batch_size=args.batch_size)
    test_loader = DataLoader(TextDataset(test_df, tokenizer), batch_size=args.batch_size)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optim.step()
            losses.append(out.loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                val_losses.append(model(**batch).loss.item())
        print(f"epoch={epoch + 1} train_loss={np.mean(losses):.4f} val_loss={np.mean(val_losses):.4f}")

    y_true, y_prob = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels").numpy()
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            y_true.append(labels)
            y_prob.append(probs)
    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)
    table, summary = metrics_table(y_true, y_prob)
    model.save_pretrained(CLASSIFIER_DIR)
    tokenizer.save_pretrained(CLASSIFIER_DIR)
    return table, summary, y_prob

def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

def write_results(baseline_table, baseline_summary, transformer_table, transformer_summary):
    MODELS.mkdir(exist_ok=True)
    lines = ["# Classifier Results", ""]
    lines += ["## Baseline: TF-IDF + One-vs-Rest Logistic Regression", ""]
    lines += [f"- Micro F1: {baseline_summary['micro_f1']:.4f}", f"- Macro F1: {baseline_summary['macro_f1']:.4f}", ""]
    lines += [markdown_table(baseline_table), ""]
    lines += ["## Transformer: DistilBERT Multi-label Classifier", ""]
    if transformer_table is None:
        lines += [f"Transformer training was not completed: {transformer_summary.get('error', 'unknown error')}", ""]
    else:
        lines += [f"- Micro F1: {transformer_summary['micro_f1']:.4f}", f"- Macro F1: {transformer_summary['macro_f1']:.4f}", ""]
        lines += [markdown_table(transformer_table), ""]
    (MODELS / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-transformer", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    df = read_articles(args.articles)
    train_df, val_df, test_df = split_data(df, args.seed)
    baseline_table, baseline_summary, baseline_probs = train_baseline(train_df, test_df)
    transformer_table = transformer_summary = transformer_probs = None
    if not args.skip_transformer:
        transformer_table, transformer_summary, transformer_probs = train_transformer(train_df, val_df, test_df, args)
    else:
        transformer_summary = {"error": "Skipped by --skip-transformer"}

    probs = transformer_probs if transformer_probs is not None else baseline_probs
    pred_df = test_df[["doc_id", "date", "title", "text", "labels"]].copy()
    for i, label in enumerate(ALL_LABELS):
        pred_df[f"prob_{label}"] = probs[:, i]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(DATA_PROCESSED / "document_predictions.csv", index=False)

    write_results(baseline_table, baseline_summary, transformer_table, transformer_summary)
    print(f"Baseline micro/macro F1: {baseline_summary['micro_f1']:.4f}/{baseline_summary['macro_f1']:.4f}")
    if transformer_table is not None:
        print(f"Transformer micro/macro F1: {transformer_summary['micro_f1']:.4f}/{transformer_summary['macro_f1']:.4f}")
    else:
        print(transformer_summary["error"])


if __name__ == "__main__":
    main()



