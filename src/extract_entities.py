from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from utils import DATA_PROCESSED, ensure_dirs, read_articles

KNOWN_ORGS = [
    "Northgate Minerals", "Blue Harbor Capital", "Arden Trade Group", "Crescent Foods",
    "Lydian Telecom", "Silverline Logistics", "Marula Energy", "Keystone Importers",
    "Vista Biomedical", "Helios Shipping", "Cobalt Bridge Bank", "Trident Metals",
    "Atlas Pharma", "Pioneer Rail", "Greenfield Renewables", "Orchid Payments",
]
KNOWN_PEOPLE = [
    "Maya Chen", "Daniel Ortiz", "Priya Nair", "Omar Haddad", "Elena Petrova",
    "Thomas Reed", "Sofia Alvarez", "Marcus Blake", "Nadia Volkov", "Kenji Sato",
]
KNOWN_LOCATIONS = [
    "United States", "United Kingdom", "Singapore", "Germany", "UAE", "Brazil",
    "South Africa", "Turkey", "Cyprus", "Panama", "Malaysia", "Canada",
]
ORG_SUFFIXES = r"(?:Bank|Capital|Group|Foods|Telecom|Logistics|Energy|Importers|Biomedical|Shipping|Metals|Pharma|Rail|Renewables|Payments|Minerals)"


def _load_nlp():
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
            ruler = nlp.add_pipe("entity_ruler")
            patterns = []
            for name in KNOWN_ORGS:
                patterns.append({"label": "ORG", "pattern": name})
            for name in KNOWN_PEOPLE:
                patterns.append({"label": "PERSON", "pattern": name})
            for name in KNOWN_LOCATIONS:
                patterns.append({"label": "GPE", "pattern": name})
            ruler.add_patterns(patterns)
            return nlp
    except Exception:
        return None


NLP = None


def _regex_entities(text: str) -> dict[str, list[str]]:
    orgs = set(name for name in KNOWN_ORGS if name in text)
    people = set(name for name in KNOWN_PEOPLE if name in text)
    locations = set(name for name in KNOWN_LOCATIONS if name in text)

    for match in re.finditer(rf"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+{ORG_SUFFIXES}\b", text):
        orgs.add(match.group(0))
    for match in re.finditer(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text):
        value = match.group(0)
        if value not in orgs and value not in locations:
            people.add(value)

    return {
        "orgs": sorted(orgs),
        "people": sorted(people),
        "locations": sorted(locations),
    }


def extract_entities(text: str) -> dict[str, list[str]]:
    global NLP
    if NLP is None:
        NLP = _load_nlp()
    if NLP is None:
        return _regex_entities(text)

    buckets = {"orgs": [], "people": [], "locations": []}
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
