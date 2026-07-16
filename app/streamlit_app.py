from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extract_entities import extract_entities
from predict_classifier import classify
from utils import ALL_LABELS, DATA_PROCESSED, DATA_RAW, label_matrix, parse_labels

st.set_page_config(page_title="Adverse Media Classifier", layout="wide")


@st.cache_data
def load_articles():
    path = DATA_RAW / "articles.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["labels_list"] = df["labels"].apply(parse_labels)
    return df


@st.cache_data
def load_optional_csv(name: str):
    path = DATA_PROCESSED / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def highlight_entities(text: str, ents: dict[str, list[str]]) -> str:
    html = escape(text)
    colors = {"orgs": "#dbeafe", "people": "#dcfce7", "locations": "#fef3c7"}
    for key, values in ents.items():
        for value in sorted(values, key=len, reverse=True):
            safe = escape(value)
            html = html.replace(safe, f"<mark style='background:{colors[key]}; padding:2px 4px'>{safe}</mark>")
    return html


def overview(df: pd.DataFrame):
    st.title("Adverse Media & SAR-Style Document Classifier")
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", f"{len(df):,}")
    c2.metric("Date range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    c3.metric("Multi-label docs", sum(len(x) > 1 for x in df["labels_list"]))
    y = label_matrix(df["labels_list"])
    counts = y.sum().reset_index()
    counts.columns = ["label", "count"]
    st.plotly_chart(px.bar(counts, x="label", y="count", title="Class Distribution"), width="stretch")
    co = y.T.dot(y)
    st.plotly_chart(px.imshow(co, text_auto=True, title="Label Co-occurrence"), width="stretch")


def document_explorer(df: pd.DataFrame):
    st.title("Document Explorer")
    entities = load_optional_csv("entities.csv")
    preds = load_optional_csv("document_predictions.csv")
    left, right = st.columns([1, 2])
    with left:
        query = st.text_input("Search")
        label = st.selectbox("True label filter", ["any"] + ALL_LABELS)
        date_range = st.date_input("Date range", [df["date"].min().date(), df["date"].max().date()])
    view = df.copy()
    if query:
        view = view[view["text"].str.contains(query, case=False, na=False) | view["title"].str.contains(query, case=False, na=False)]
    if label != "any":
        view = view[view["labels_list"].apply(lambda labels: label in labels)]
    if len(date_range) == 2:
        view = view[(view["date"].dt.date >= date_range[0]) & (view["date"].dt.date <= date_range[1])]
    selected_id = st.selectbox("Document", view["doc_id"].tolist()[:500] if not view.empty else [])
    st.dataframe(view[["doc_id", "date", "title", "labels"]].head(300), width="stretch", hide_index=True)
    if selected_id:
        row = df[df["doc_id"] == selected_id].iloc[0]
        ents = extract_entities(row["text"])
        st.subheader(row["title"])
        st.caption(f"{row['doc_id']} | {row['date'].date()} | true labels: {', '.join(row['labels_list'])}")
        st.markdown(highlight_entities(row["text"], ents), unsafe_allow_html=True)
        probs = classify(row["text"])
        st.plotly_chart(px.bar(x=list(probs.keys()), y=list(probs.values()), labels={"x": "label", "y": "probability"}), width="stretch")
        st.json(ents)


def performance_page():
    st.title("Classifier Performance")
    results = ROOT / "models" / "results.md"
    if results.exists():
        st.markdown(results.read_text(encoding="utf-8"))
    else:
        st.info("Run `python src/train_classifier.py` to generate model metrics.")


def topic_page(df: pd.DataFrame):
    st.title("Topic Explorer")
    summary = load_optional_csv("topics_summary.csv")
    over_time = load_optional_csv("topics_over_time.csv")
    if summary.empty:
        st.info("Run `python src/train_topic_model.py` to generate topics.")
        return
    st.dataframe(summary, width="stretch", hide_index=True)
    if not over_time.empty:
        st.plotly_chart(px.area(over_time, x="period", y="count", color="topic_id", title="Topics Over Time"), width="stretch")
    doc_id = st.selectbox("Representative document", df["doc_id"].tolist())
    row = df[df["doc_id"] == doc_id].iloc[0]
    st.subheader(row["title"])
    st.write(row["text"])


def live_scoring():
    st.title("Live Scoring")
    text = st.text_area("Article text", height=240)
    if not text.strip():
        return
    probs = classify(text)
    ents = extract_entities(text)
    st.plotly_chart(px.bar(x=list(probs.keys()), y=list(probs.values()), labels={"x": "label", "y": "probability"}), width="stretch")
    st.json(ents)
    summary = load_optional_csv("topics_summary.csv")
    if not summary.empty:
        st.write("Nearest topic approximation")
        lower = text.lower()
        scored = []
        for _, row in summary.iterrows():
            kws = [kw.strip() for kw in str(row["top_keywords"]).split(",")]
            score = sum(1 for kw in kws if kw and kw in lower)
            scored.append((score, row["topic_id"], row["top_keywords"]))
        scored.sort(reverse=True)
        st.write({"topic_id": int(scored[0][1]), "keywords": scored[0][2]})


def main():
    df = load_articles()
    if df.empty:
        st.warning("No corpus found. Run `python src/generate_data.py --n-docs 1800` first.")
        return
    page = st.sidebar.radio("Navigate", ["Overview", "Document Explorer", "Classifier Performance", "Topic Explorer", "Live Scoring"])
    if page == "Overview":
        overview(df)
    elif page == "Document Explorer":
        document_explorer(df)
    elif page == "Classifier Performance":
        performance_page()
    elif page == "Topic Explorer":
        topic_page(df)
    else:
        live_scoring()


if __name__ == "__main__":
    main()

