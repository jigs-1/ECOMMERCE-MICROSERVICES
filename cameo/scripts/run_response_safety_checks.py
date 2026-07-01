import argparse
import csv
import json
from pathlib import Path

from cameo.core.models.response import ResponseEngine


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def contains_all_terms(text: str, terms: str) -> bool:
    values = [t.strip().lower() for t in terms.split("|") if t.strip()]
    text_lower = text.lower()
    return all(term in text_lower for term in values)


def contains_any_term(text: str, terms: str) -> bool:
    values = [t.strip().lower() for t in terms.split("|") if t.strip()]
    text_lower = text.lower()
    return any(term in text_lower for term in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/response_safety_cases.csv")
    parser.add_argument("--output", default="artifacts/response_safety_results.json")
    args = parser.parse_args()

    engine = ResponseEngine()
    rows = load_rows(Path(args.cases))

    detailed = []
    passed = 0
    for row in rows:
        result = engine(
            text=row["text"],
            intent=row["intent"],
            emotion=row["emotion"],
            intensity=float(row["intensity"]),
            confidence=float(row["confidence"]),
        )
        mode_ok = result.mode == row["expected_mode"]
        required_ok = contains_all_terms(result.text, row["required_contains"])
        forbidden_ok = not contains_any_term(result.text, row["forbidden_contains"])
        case_pass = mode_ok and required_ok and forbidden_ok
        passed += int(case_pass)
        detailed.append(
            {
                "text": row["text"],
                "intent": row["intent"],
                "emotion": row["emotion"],
                "mode": result.mode,
                "expected_mode": row["expected_mode"],
                "passed": case_pass,
                "required_ok": required_ok,
                "forbidden_ok": forbidden_ok,
                "response": result.text,
            }
        )

    results = {
        "summary": {
            "cases": len(rows),
            "passed": passed,
            "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        },
        "cases": detailed,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
