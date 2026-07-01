import json

import torch
from PIL import Image
from transformers import AutoTokenizer

from cameo.core.inference.pipeline import CameoPipeline
from cameo.core.preprocess.image import preprocess_image
from cameo.core.preprocess.text import clean_text


cases = [
    {
        "id": "celebration_standard",
        "text": "I got selected and I am so happy right now.",
        "image": "data/presentation_images/happy_celebration_018.png",
        "expected_emotion": "happy",
        "expected_intent": "celebration",
        "note": "Canonical positive success case",
    },
    {
        "id": "distress_standard",
        "text": "I feel very low and I do not know how to handle this.",
        "image": "data/presentation_images/sad_distress_014.png",
        "expected_emotion": "sad",
        "expected_intent": "distress",
        "note": "Canonical distress case",
    },
    {
        "id": "frustration_standard",
        "text": "I am frustrated because nothing is going right.",
        "image": "data/presentation_images/angry_frustration_021.png",
        "expected_emotion": "angry",
        "expected_intent": "frustration",
        "note": "Canonical frustration case",
    },
    {
        "id": "neutral_standard",
        "text": "Today was normal and nothing unusual happened.",
        "image": "data/presentation_images/neutral_neutral_023.png",
        "expected_emotion": "neutral",
        "expected_intent": "neutral",
        "note": "Canonical neutral case",
    },
    {
        "id": "help_standard",
        "text": "I feel anxious and I need help deciding what to do next.",
        "image": "data/presentation_images/anxious_seeking_help_032.png",
        "expected_emotion": "anxious",
        "expected_intent": "seeking_help",
        "note": "Canonical help-seeking case",
    },
    {
        "id": "gratitude_standard",
        "text": "Thank you, I feel hopeful about what comes next.",
        "image": "data/presentation_images/hopeful_gratitude_020.png",
        "expected_emotion": "hopeful",
        "expected_intent": "gratitude",
        "note": "Canonical gratitude case",
    },
    {
        "id": "edge_neg_help",
        "text": "I am overwhelmed by deadlines and need someone to help me decide what to do first.",
        "image": "data/presentation_images/anxious_seeking_help_032.png",
        "expected_emotion": "anxious",
        "expected_intent": "seeking_help",
        "note": "Edge case: negative wording can be over-pulled to distress",
    },
    {
        "id": "edge_success_neg_word",
        "text": "I finally got the offer letter and I cannot stop smiling.",
        "image": "data/presentation_images/happy_celebration_018.png",
        "expected_emotion": "happy",
        "expected_intent": "celebration",
        "note": "Edge case: contains cannot, may trigger distress calibration",
    },
    {
        "id": "edge_neutral_subtle",
        "text": "Things are steady now. I am not excited or upset, just moving through the day.",
        "image": "data/presentation_images/neutral_neutral_010.png",
        "expected_emotion": "neutral",
        "expected_intent": "neutral",
        "note": "Edge case: subtle neutral language can drift positive",
    },
    {
        "id": "edge_low_conf_distress",
        "text": "I may be overreacting, but I feel low and alone.",
        "image": "data/presentation_images/sad_distress_029.png",
        "expected_emotion": "sad",
        "expected_intent": "distress",
        "note": "Edge case: low confidence distress wording",
    },
]


device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
model = CameoPipeline(device=device)
state = torch.load("artifacts/cameo_cached_eval.pt", map_location=device)
model.load_state_dict(state)
model.eval()

rows = []
for case in cases:
    cleaned = clean_text(case["text"])
    toks = tokenizer(cleaned, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    img = Image.open(case["image"]).convert("RGB")
    img_tensor = preprocess_image(img).unsqueeze(0)
    pred = model(toks, img_tensor, raw_text=cleaned)
    rows.append(
        {
            **case,
            "cleaned_text": cleaned,
            "predicted_emotion": pred.emotion,
            "predicted_intent": pred.intent,
            "predicted_intensity": round(float(pred.intensity), 4),
            "confidence": round(float(pred.confidence), 4),
            "attn_text": round(float(pred.attn_weights["text"]), 4),
            "attn_image": round(float(pred.attn_weights["image"]), 4),
            "gate_text": round(float(pred.gates["text"]), 4),
            "gate_image": round(float(pred.gates["image"]), 4),
            "response_mode": pred.response.mode,
            "response_text": pred.response.text,
        }
    )

print(json.dumps(rows, indent=2))
