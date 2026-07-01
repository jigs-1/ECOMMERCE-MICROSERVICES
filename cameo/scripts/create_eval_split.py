import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "image_path", "emotion", "intensity", "intent"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input manifest CSV")
    parser.add_argument("--train-output", default="data/presentation_train_manifest.csv")
    parser.add_argument("--eval-output", default="data/presentation_eval_manifest.csv")
    parser.add_argument("--eval-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = read_rows(Path(args.manifest))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["emotion"], row["intent"])].append(row)

    rng = random.Random(args.seed)
    train_rows = []
    eval_rows = []
    for key, items in sorted(grouped.items()):
        rng.shuffle(items)
        eval_count = min(args.eval_per_class, len(items))
        eval_rows.extend(items[:eval_count])
        train_rows.extend(items[eval_count:])

    write_rows(Path(args.train_output), train_rows)
    write_rows(Path(args.eval_output), eval_rows)
    print(f"train_rows={len(train_rows)} eval_rows={len(eval_rows)} classes={len(grouped)}")


if __name__ == "__main__":
    main()
