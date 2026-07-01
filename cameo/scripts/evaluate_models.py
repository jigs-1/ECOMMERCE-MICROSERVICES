import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
from transformers import AutoTokenizer

from cameo.core.inference.pipeline import CameoPipeline
from cameo.core.preprocess.image import preprocess_image
from cameo.core.preprocess.text import clean_text


EMOTIONS = ["happy", "sad", "angry", "neutral", "anxious", "hopeful"]
INTENTS = ["distress", "celebration", "frustration", "neutral", "seeking_help", "gratitude"]


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def accuracy(y_true, y_pred):
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mae(y_true, y_pred):
    return mean(abs(a - b) for a, b in zip(y_true, y_pred))


def macro_f1(y_true, y_pred, labels):
    scores = []
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return mean(scores)


def majority_baseline(train_rows, eval_rows):
    train_emotions = [int(r["emotion"]) for r in train_rows]
    train_intents = [int(r["intent"]) for r in train_rows]
    train_intensity = [float(r["intensity"]) for r in train_rows]

    emotion_majority = Counter(train_emotions).most_common(1)[0][0]
    intent_majority = Counter(train_intents).most_common(1)[0][0]
    intensity_mean = mean(train_intensity)

    return {
        "emotion_pred": [emotion_majority] * len(eval_rows),
        "intent_pred": [intent_majority] * len(eval_rows),
        "intensity_pred": [intensity_mean] * len(eval_rows),
    }


def heuristic_baseline(eval_rows):
    emotion_pred = []
    intent_pred = []
    intensity_pred = []
    for row in eval_rows:
        calibrated = CameoPipeline._calibrate_from_text(clean_text(row["text"]))
        if calibrated is None:
            intent = "neutral"
            emotion = "neutral"
            intensity = 0.4
        else:
            intent = calibrated["intent"]
            emotion = calibrated["emotion"]
            intensity = calibrated["intensity"]
        emotion_pred.append(EMOTIONS.index(emotion))
        intent_pred.append(INTENTS.index(intent))
        intensity_pred.append(float(intensity))
    return {
        "emotion_pred": emotion_pred,
        "intent_pred": intent_pred,
        "intensity_pred": intensity_pred,
    }


def model_predictions(weights_path: Path, eval_rows, device: str):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    model = CameoPipeline(device=device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    emotion_pred = []
    intent_pred = []
    intensity_pred = []
    for row in eval_rows:
        cleaned = clean_text(row["text"])
        toks = tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = preprocess_image(image).unsqueeze(0)
        pred = model(toks, image_tensor, raw_text=cleaned)
        emotion_pred.append(EMOTIONS.index(pred.emotion))
        intent_pred.append(INTENTS.index(pred.intent))
        intensity_pred.append(float(pred.intensity))
    return {
        "emotion_pred": emotion_pred,
        "intent_pred": intent_pred,
        "intensity_pred": intensity_pred,
    }


def summarize(name, preds, eval_rows):
    y_emotion = [int(r["emotion"]) for r in eval_rows]
    y_intent = [int(r["intent"]) for r in eval_rows]
    y_intensity = [float(r["intensity"]) for r in eval_rows]
    return {
        "emotion_accuracy": round(accuracy(y_emotion, preds["emotion_pred"]), 4),
        "emotion_macro_f1": round(macro_f1(y_emotion, preds["emotion_pred"], sorted(set(y_emotion))), 4),
        "intent_accuracy": round(accuracy(y_intent, preds["intent_pred"]), 4),
        "intent_macro_f1": round(macro_f1(y_intent, preds["intent_pred"], sorted(set(y_intent))), 4),
        "intensity_mae": round(mae(y_intensity, preds["intensity_pred"]), 4),
        "samples": len(eval_rows),
        "name": name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="artifacts/eval_metrics.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_rows = load_rows(Path(args.train_manifest))
    eval_rows = load_rows(Path(args.eval_manifest))

    results = {
        "dataset": {
            "train_manifest": args.train_manifest,
            "eval_manifest": args.eval_manifest,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
        },
        "majority_baseline": summarize("majority_baseline", majority_baseline(train_rows, eval_rows), eval_rows),
        "heuristic_baseline": summarize("heuristic_baseline", heuristic_baseline(eval_rows), eval_rows),
        "cameo_model": summarize("cameo_model", model_predictions(Path(args.weights), eval_rows, args.device), eval_rows),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
