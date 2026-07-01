import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from cameo.core.inference.pipeline import CameoPipeline
from cameo.core.preprocess.image import preprocess_image
from cameo.core.preprocess.text import clean_text


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/realistic_eval_cases.csv")
    parser.add_argument("--weights", default="artifacts/cameo_cached_eval.pt")
    parser.add_argument("--output", default="artifacts/realistic_eval_results.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    model = CameoPipeline(device=device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rows = load_rows(Path(args.cases))
    detailed = []
    correct_emotion = 0
    correct_intent = 0
    both_correct = 0

    for row in rows:
        cleaned = clean_text(row["text"])
        toks = tokenizer(cleaned, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = preprocess_image(image).unsqueeze(0)
        pred = model(toks, image_tensor, raw_text=cleaned)

        emotion_ok = pred.emotion == row["expected_emotion"]
        intent_ok = pred.intent == row["expected_intent"]
        correct_emotion += int(emotion_ok)
        correct_intent += int(intent_ok)
        both_correct += int(emotion_ok and intent_ok)

        detailed.append(
            {
                "text": row["text"],
                "notes": row["notes"],
                "expected_emotion": row["expected_emotion"],
                "predicted_emotion": pred.emotion,
                "expected_intent": row["expected_intent"],
                "predicted_intent": pred.intent,
                "intensity": round(float(pred.intensity), 4),
                "confidence": round(float(pred.confidence), 4),
                "emotion_match": emotion_ok,
                "intent_match": intent_ok,
                "response_mode": pred.response.mode,
            }
        )

    total = len(rows)
    results = {
        "summary": {
            "cases": total,
            "emotion_accuracy": round(correct_emotion / total, 4),
            "intent_accuracy": round(correct_intent / total, 4),
            "joint_accuracy": round(both_correct / total, 4),
        },
        "cases": detailed,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
