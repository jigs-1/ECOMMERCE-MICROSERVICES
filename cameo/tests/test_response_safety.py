import csv
from pathlib import Path

from cameo.core.models.response import ResponseEngine


def _rows():
    with Path("data/response_safety_cases.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _contains_all(text: str, terms: str) -> bool:
    values = [t.strip().lower() for t in terms.split("|") if t.strip()]
    low = text.lower()
    return all(term in low for term in values)


def _contains_any(text: str, terms: str) -> bool:
    values = [t.strip().lower() for t in terms.split("|") if t.strip()]
    low = text.lower()
    return any(term in low for term in values)


def test_response_safety_cases_pass():
    engine = ResponseEngine()

    for row in _rows():
        result = engine(
            text=row["text"],
            intent=row["intent"],
            emotion=row["emotion"],
            intensity=float(row["intensity"]),
            confidence=float(row["confidence"]),
        )
        assert result.mode == row["expected_mode"]
        assert _contains_all(result.text, row["required_contains"])
        assert not _contains_any(result.text, row["forbidden_contains"])
