from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

from utils import ALL_LABELS, DATA_RAW, RISK_LABELS, ensure_dirs, labels_to_json

COMPANIES = [
    "Northgate Minerals", "Blue Harbor Capital", "Arden Trade Group", "Crescent Foods",
    "Lydian Telecom", "Silverline Logistics", "Marula Energy", "Keystone Importers",
    "Vista Biomedical", "Helios Shipping", "Cobalt Bridge Bank", "Trident Metals",
    "Atlas Pharma", "Pioneer Rail", "Greenfield Renewables", "Orchid Payments",
]
PEOPLE = [
    "Maya Chen", "Daniel Ortiz", "Priya Nair", "Omar Haddad", "Elena Petrova",
    "Thomas Reed", "Sofia Alvarez", "Marcus Blake", "Nadia Volkov", "Kenji Sato",
]
COUNTRIES = [
    "United States", "United Kingdom", "Singapore", "Germany", "UAE", "Brazil",
    "South Africa", "Turkey", "Cyprus", "Panama", "Malaysia", "Canada",
]
INDUSTRIES = ["shipping", "payments", "energy", "pharmaceuticals", "metals", "telecom", "construction"]
AUTHORITIES = ["financial regulators", "federal prosecutors", "market supervisors", "customs officials", "bank examiners"]

TEMPLATES = {
    "fraud": [
        "{company} disclosed that {authority} are reviewing accounting entries tied to {amount} in revenue booked by its {industry} unit. People familiar with the matter said {person}, a former executive, approved invoices that investigators believe may have overstated customer demand. The company said it is cooperating and has hired outside counsel in {country}. Analysts said the allegations could delay financing and trigger restatements if the review confirms that contracts were misrepresented.",
        "Shares of {company} fell after a filing described internal findings around false vendor records, inflated receivables, and unusual bonus payments. The review focuses on transactions between {company} and distributors in {country}. {person}, who led regional sales, denied wrongdoing through a spokesperson. The board said no customer funds were lost, but lenders requested more detail on controls and revenue recognition.",
        "{authority} opened a civil inquiry into whether {company} misled investors about the performance of a {industry} project. The complaint cites backdated purchase orders, altered emails, and payments totaling {amount}. {person} said the company would contest the allegations and described the matter as a contract dispute rather than a fraud scheme.",
    ],
    "money_laundering": [
        "{company} is under scrutiny after banks flagged a chain of transfers through shell suppliers in {country}. Compliance staff reported round-dollar payments, rapid movement of funds, and invoices that did not match shipping records. {person} said the {industry} firm strengthened monitoring after suspicious activity alerts involving {amount}.",
        "Investigators are examining whether {company} helped clients move proceeds through layered accounts and trade invoices. The activity involved counterparties in {country} and payments routed through several intermediaries within days. {person}, the firm's compliance head, said suspicious accounts were closed and reports were filed with authorities.",
        "A leaked audit described weak anti-money-laundering controls at {company}, including missing beneficial-owner records and repeated cash-equivalent transfers. The review identified {amount} in transactions connected to high-risk customers in {country}. The company said the issues were historical and remediation is underway.",
    ],
    "sanctions_violation": [
        "{company} said it received questions from {authority} about shipments that may have reached restricted counterparties through brokers in {country}. The review covers equipment, payments of {amount}, and screening failures in the {industry} division. {person} said the company suspended the intermediaries while it reviews sanctions controls.",
        "{authority} are investigating whether {company} processed transactions for entities later identified on a sanctions list. Internal messages cited urgency around a customer in {country} and concerns that names were misspelled in payment instructions. {person} said the company voluntarily disclosed the matter.",
        "{company} warned investors that possible sanctions violations could result in penalties after a distributor sold goods into a restricted market. The shipments were booked by the {industry} unit and involved invoices worth {amount}. Management said it enhanced restricted-party screening and export controls.",
    ],
    "none": [
        "{company} announced a new {industry} partnership in {country}, with {person} saying the agreement will expand service coverage and improve delivery times. The company expects the project to create jobs and increase annual revenue, subject to routine regulatory approvals.",
        "{company} reported steady quarterly earnings as demand improved across its {industry} customers. Executives said operating margins increased and cash flow remained positive. {person} told analysts the firm is investing in technology and supply-chain resilience in {country}.",
        "{company} opened a regional office in {country} to support clients in the {industry} sector. The announcement focused on hiring, infrastructure, and new customer contracts. {person} said the expansion reflects long-term confidence in local markets.",
        "{company} completed a refinancing package worth {amount}, extending debt maturities and lowering interest expense. The company said the transaction was supported by banks in {country} and will fund routine capital spending for its {industry} operations.",
    ],
}

BRIDGES = [
    "The matter remains at an early stage, and no charges have been filed.",
    "Several customers said service continued without interruption.",
    "The company said its audit committee is reviewing documents and interviewing employees.",
    "Market analysts said the case highlights the importance of stronger controls and board oversight.",
]


def money(rng: random.Random) -> str:
    return f"${rng.randint(2, 950)} million"


def random_date(rng: random.Random) -> date:
    start = date(2021, 1, 1)
    return start + timedelta(days=rng.randint(0, 365 * 4 + 180))


def choose_labels(rng: random.Random) -> list[str]:
    if rng.random() < 0.55:
        return ["none"]
    primary = rng.choice(RISK_LABELS)
    labels = {primary}
    if rng.random() < 0.13:
        labels.add(rng.choice([label for label in RISK_LABELS if label != primary]))
    if rng.random() < 0.03:
        labels.update(RISK_LABELS)
    return sorted(labels)


def make_article(doc_id: int, labels: list[str], rng: random.Random) -> dict[str, str]:
    company = rng.choice(COMPANIES)
    person = rng.choice(PEOPLE)
    country = rng.choice(COUNTRIES)
    industry = rng.choice(INDUSTRIES)
    authority = rng.choice(AUTHORITIES)
    amount = money(rng)
    label_for_template = "none" if labels == ["none"] else rng.choice(labels)
    text = rng.choice(TEMPLATES[label_for_template]).format(
        company=company, person=person, country=country, industry=industry,
        authority=authority, amount=amount,
    )
    if len(labels) > 1:
        other = rng.choice([label for label in labels if label != label_for_template])
        text += " " + rng.choice(TEMPLATES[other]).format(
            company=company, person=rng.choice(PEOPLE), country=rng.choice(COUNTRIES),
            industry=industry, authority=authority, amount=money(rng),
        )
    while len(text.split()) < 150:
        text += " " + rng.choice(BRIDGES)
    words = text.split()
    if len(words) > 400:
        text = " ".join(words[:400])
    title_prefix = {
        "fraud": "Regulators Review",
        "money_laundering": "Banks Flag Transfers At",
        "sanctions_violation": "Sanctions Controls Questioned At",
        "none": "Expansion Announced By",
    }[label_for_template]
    return {
        "doc_id": f"DOC-{doc_id:05d}",
        "date": random_date(rng).isoformat(),
        "title": f"{title_prefix} {company}",
        "text": text,
        "labels": labels_to_json(labels),
        "source": "synthetic_template",
    }


def try_real_negatives(limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    try:
        from datasets import load_dataset

        ds = load_dataset("ag_news", split=f"train[:{limit}]")
    except Exception:
        return []
    rows = []
    for idx, item in enumerate(ds):
        text = str(item.get("text", ""))
        if len(text.split()) < 25:
            text = text + " The report did not describe allegations of financial crime, sanctions breaches, or misconduct."
        rows.append({
            "doc_id": f"REAL-{idx:05d}",
            "date": (date(2021, 1, 1) + timedelta(days=idx % 1300)).isoformat(),
            "title": text[:90],
            "text": text,
            "labels": json.dumps(["none"]),
            "source": "ag_news_none",
        })
    return rows


def generate(n_docs: int, seed: int, include_real_negatives: bool) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)
    real_rows = try_real_negatives(min(150, n_docs // 10)) if include_real_negatives else []
    synthetic_n = max(0, n_docs - len(real_rows))
    rows = [make_article(i, choose_labels(rng), rng) for i in range(synthetic_n)]
    rows.extend(real_rows)
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-real-negatives", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    df = generate(args.n_docs, args.seed, args.include_real_negatives)
    out = DATA_RAW / "articles.csv"
    df.to_csv(out, index=False)
    counts = Counter(label for labels in df["labels"].map(json.loads) for label in labels)
    multi = sum(1 for labels in df["labels"].map(json.loads) if len(labels) > 1)
    print(f"Wrote {len(df)} documents to {out}")
    print(f"Label counts: {dict(counts)}")
    print(f"Multi-label adverse documents: {multi}")
    if args.include_real_negatives and not (df["source"] == "ag_news_none").any():
        print("Real negative import was requested but unavailable; generated synthetic-only corpus.")


if __name__ == "__main__":
    main()
