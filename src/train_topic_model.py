from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils import DATA_PROCESSED, TOPIC_DIR, ensure_dirs, read_articles


def _clean_keywords(words):
    return ", ".join([str(word) for word in words if str(word).strip()][:10])


def fit_bertopic(df: pd.DataFrame):
    try:
        from bertopic import BERTopic
    except Exception as exc:
        return None, None, f"BERTopic unavailable: {exc}"
    try:
        model = BERTopic(min_topic_size=15, calculate_probabilities=False, verbose=False)
        topics, _ = model.fit_transform(df["text"].tolist())
        topic_info = model.get_topic_info()
        if len(topic_info[topic_info["Topic"] != -1]) < 2:
            return None, None, "BERTopic produced fewer than two non-outlier topics."
        topics_over_time = model.topics_over_time(df["text"].tolist(), pd.to_datetime(df["date"]).tolist(), nr_bins=12)
        return model, {"topics": topics, "topic_info": topic_info, "topics_over_time": topics_over_time}, None
    except Exception as exc:
        return None, None, f"BERTopic failed: {exc}"


def fit_lda(df: pd.DataFrame, n_topics: int, seed: int):
    vectorizer = CountVectorizer(stop_words="english", min_df=3, max_df=0.85, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["text"])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=seed, learning_method="batch")
    doc_topic = lda.fit_transform(matrix)
    topic_ids = doc_topic.argmax(axis=1)
    vocab = vectorizer.get_feature_names_out()
    keywords = {}
    for topic_idx, comp in enumerate(lda.components_):
        top = comp.argsort()[-10:][::-1]
        keywords[topic_idx] = [vocab[i] for i in top]
    return {"model": lda, "vectorizer": vectorizer, "doc_topic": doc_topic, "topic_ids": topic_ids, "keywords": keywords}


def representative_docs(df: pd.DataFrame, topic_ids, topic_id: int, limit: int = 5) -> list[str]:
    docs = df.loc[pd.Series(topic_ids) == topic_id, "doc_id"].head(limit).tolist()
    return docs


def build_lda_summary(df: pd.DataFrame, lda_bundle: dict) -> pd.DataFrame:
    rows = []
    topic_ids = lda_bundle["topic_ids"]
    for topic_id, words in lda_bundle["keywords"].items():
        rows.append({
            "topic_id": int(topic_id),
            "model": "lda",
            "top_keywords": _clean_keywords(words),
            "doc_count": int((topic_ids == topic_id).sum()),
            "representative_doc_ids": json.dumps(representative_docs(df, topic_ids, topic_id)),
        })
    return pd.DataFrame(rows).sort_values("doc_count", ascending=False)


def build_bertopic_summary(df: pd.DataFrame, model, topic_ids) -> pd.DataFrame:
    rows = []
    for topic_id in sorted(set(topic_ids)):
        if topic_id == -1:
            continue
        words = [word for word, _ in model.get_topic(topic_id)[:10]]
        rows.append({
            "topic_id": int(topic_id),
            "model": "bertopic",
            "top_keywords": _clean_keywords(words),
            "doc_count": int(sum(1 for tid in topic_ids if tid == topic_id)),
            "representative_doc_ids": json.dumps(representative_docs(df, pd.Series(topic_ids), topic_id)),
        })
    return pd.DataFrame(rows).sort_values("doc_count", ascending=False)


def topics_over_time_from_assignments(df: pd.DataFrame, topic_ids) -> pd.DataFrame:
    tmp = df[["doc_id", "date"]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["period"] = tmp["date"].dt.to_period("M").astype(str)
    tmp["topic_id"] = topic_ids
    return tmp.groupby(["period", "topic_id"]).size().reset_index(name="count")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=Path, default=None)
    parser.add_argument("--n-topics", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bertopic", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    df = read_articles(args.articles)
    bert_model = bert_result = bert_error = None
    if not args.skip_bertopic:
        bert_model, bert_result, bert_error = fit_bertopic(df)

    lda_bundle = fit_lda(df, args.n_topics, args.seed)
    lda_summary = build_lda_summary(df, lda_bundle)

    if bert_model is not None:
        primary = "bertopic"
        summary = build_bertopic_summary(df, bert_model, bert_result["topics"])
        over_time = topics_over_time_from_assignments(df, pd.Series(bert_result["topics"]))
        bert_model.save(str(TOPIC_DIR / "bertopic_model"), serialization="pickle")
    else:
        primary = "lda"
        summary = lda_summary
        over_time = topics_over_time_from_assignments(df, lda_bundle["topic_ids"])

    joblib.dump(lda_bundle, TOPIC_DIR / "lda_model.joblib")
    summary.to_csv(DATA_PROCESSED / "topics_summary.csv", index=False)
    lda_summary.to_csv(DATA_PROCESSED / "lda_topics_summary.csv", index=False)
    over_time.to_csv(DATA_PROCESSED / "topics_over_time.csv", index=False)
    (TOPIC_DIR / "primary_model.txt").write_text(primary, encoding="utf-8")
    if bert_error:
        (TOPIC_DIR / "bertopic_status.txt").write_text(bert_error, encoding="utf-8")
    print(f"Primary topic model: {primary}")
    print(f"Topics found: {len(summary)}")
    for _, row in summary.head(5).iterrows():
        print(f"Topic {row['topic_id']}: {row['top_keywords']}")


if __name__ == "__main__":
    main()
