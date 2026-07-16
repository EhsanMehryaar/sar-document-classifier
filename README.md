# Adverse Media & SAR-Style Document Classifier + Topic Model

Portfolio/demo project for an AML/KYC adverse media screening workflow. It generates a synthetic news-style corpus, trains multi-label risk classifiers, extracts named entities, models themes over time, and serves the results through Streamlit.

This is not a production AML system. The labels, articles, and risk patterns are synthetic and intended for demonstrating an end-to-end ML pipeline.

## What It Does

- Multi-label document classification for `fraud`, `money_laundering`, `sanctions_violation`, and mutually exclusive `none`
- Entity extraction for organizations, people, and locations
- Topic modeling with BERTopic when available, plus an LDA fallback/comparison
- Streamlit dashboard for corpus stats, document exploration, model metrics, topics, and live scoring

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If the environment has no internet access, install dependencies from a local wheel cache. The code falls back where possible, but transformer, spaCy model, and BERTopic quality depend on their packages and pretrained assets.

## Run The Pipeline

```bash
python src/generate_data.py --n-docs 1800
python src/train_classifier.py --epochs 1 --batch-size 8
python src/extract_entities.py
python src/train_topic_model.py
streamlit run app/streamlit_app.py
```

For a faster constrained run:

```bash
python src/generate_data.py --n-docs 600
python src/train_classifier.py --skip-transformer
python src/extract_entities.py
python src/train_topic_model.py --skip-bertopic
streamlit run app/streamlit_app.py
```

## Outputs

- `data/raw/articles.csv`: synthetic article corpus
- `data/processed/entities.csv`: extracted entities by document
- `data/processed/topics_summary.csv`: topic keywords and representative documents
- `data/processed/document_predictions.csv`: classifier probabilities by document
- `models/classifier/`: saved transformer model when training succeeds, plus baseline artifacts
- `models/topic_model/`: saved topic model artifacts
- `models/results.md`: real metrics from the latest training run

## Notes And Limitations

- The corpus is synthetic and may contain template artifacts.
- Real Reuters/AG News style negatives are optionally imported through HuggingFace `datasets` only if available locally/online; otherwise the generator documents that no real external text was added.
- DistilBERT fine-tuning can be slow on CPU. Use `--skip-transformer` for a baseline-only demo.
- BERTopic requires `bertopic`, `sentence-transformers`, `umap-learn`, and `hdbscan`; LDA is used as a robust fallback.
- This project demonstrates architecture and workflow, not compliance-grade accuracy or regulatory fitness.
