import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extract_entities import extract_entities
from predict_classifier import classify


def test_classify_shape():
    probs = classify("Regulators reviewed suspicious transfers and possible sanctions screening failures.")
    assert set(probs) == {"fraud", "money_laundering", "sanctions_violation", "none"}
    assert all(isinstance(value, float) for value in probs.values())


def test_extract_entities_shape():
    ents = extract_entities("Maya Chen at Northgate Minerals opened an office in Singapore.")
    assert set(ents) == {"orgs", "people", "locations"}
    assert all(isinstance(value, list) for value in ents.values())
