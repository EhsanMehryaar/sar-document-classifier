from __future__ import annotations

import argparse
import json

import pandas as pd

from utils import DATA_PROCESSED, ensure_dirs, read_articles


def _load_nlp():
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
            ruler = nlp.add_pipe("entity_ruler")
            patterns = []
            for name in ["Northgate Minerals", "Blue Harbor Capital", "Arden Trade Group", "Cobalt Bridge Bank"]:
                patterns.append({"label": "ORG", "pattern": name})
            for name in ["Maya Chen", "Daniel Ortiz", "Priya Nair", "Omar Haddad", "Elena Petrova"]:
                patterns.append({"label": "PERSON", "pattern": name})
            for name in ["United States", "United Kingdom", "Singapore", "Germany", "UAE", "Brazil", "Canada"]:
                patterns.append({"label": "GPE", "pattern": name})
            ruler.add_patterns(patterns)
            return nlp
    except Exception:
        return None


NLP = None


def extract_entities(text: str) -> dict[str, list[str]]:
    global NLP
    if NLP is None:
        NLP = _load_nlp()
    buckets = {"orgs": [], "people": [], "locations": []}
    if NLP is None:
        return buckets
    doc = NLP(text)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            buckets["orgs"].append(ent.text)
        elif ent.label_ == "PERSON":
            buckets["people"].append(ent.text)
        elif ent.label_ in {"GPE", "LOC"}:
            buckets["locations"].append(ent.text)
    return {key: sorted(set(value)) for key, value in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", default=None)
    args = parser.parse_args()

    ensure_dirs()
    df = read_articles(args.articles)
    rows = []
    for _, row in df.iterrows():
        ents = extract_entities(row["text"])
        rows.append({
            "doc_id": row["doc_id"],
            "orgs": json.dumps(ents["orgs"]),
            "people": json.dumps(ents["people"]),
            "locations": json.dumps(ents["locations"]),
        })
    out = DATA_PROCESSED / "entities.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote entities to {out}")


if __name__ == "__main__":
    main()
