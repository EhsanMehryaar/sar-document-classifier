# Classifier Results

## Baseline: TF-IDF + One-vs-Rest Logistic Regression

- Micro F1: 0.9898
- Macro F1: 0.9852

| label | precision | recall | f1 | support | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fraud | 1.0000 | 0.9375 | 0.9677 | 16 | 74 | 0 | 1 | 15 |
| money_laundering | 1.0000 | 1.0000 | 1.0000 | 17 | 73 | 0 | 0 | 17 |
| sanctions_violation | 1.0000 | 0.9474 | 0.9730 | 19 | 71 | 0 | 1 | 18 |
| none | 1.0000 | 1.0000 | 1.0000 | 47 | 43 | 0 | 0 | 47 |

## Transformer: DistilBERT Multi-label Classifier

Transformer training was not completed: Skipped by --skip-transformer
