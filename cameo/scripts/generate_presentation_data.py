"""
Generate a small synthetic multimodal dataset for presentation-time fine-tuning.
This is not a production dataset; it is a bootstrap set to make demo behavior stable.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw


EMOTIONS = ["happy", "sad", "angry", "neutral", "anxious", "hopeful"]
INTENTS = ["distress", "celebration", "frustration", "neutral", "seeking_help", "gratitude"]

EMOTION_TO_LABEL = {name: i for i, name in enumerate(EMOTIONS)}
INTENT_TO_LABEL = {name: i for i, name in enumerate(INTENTS)}


TEXT_BANK = {
    ("happy", "celebration"): [
        "I feel amazing today and everything is finally working out.",
        "I got selected and I am so happy right now.",
        "This is one of my best days in months.",
        "I am proud of myself and grateful for this moment.",
        "I achieved my goal and I feel excited and thankful.",
    ],
    ("sad", "distress"): [
        "I feel very low and I do not know how to handle this.",
        "Everything feels heavy and I am struggling today.",
        "I feel alone and overwhelmed right now.",
        "I am tired, upset, and I need support.",
        "I cannot stop worrying and it is exhausting me.",
    ],
    ("angry", "frustration"): [
        "I am frustrated because nothing is going right.",
        "I am angry and exhausted with these repeated failures.",
        "This keeps breaking and I feel very annoyed.",
        "I put in effort but I keep getting blocked.",
        "I am upset and stressed with this situation.",
    ],
    ("neutral", "neutral"): [
        "Today was normal and nothing unusual happened.",
        "I am just sharing a regular update.",
        "It is an average day and I am doing okay.",
        "No big emotions today, just routine work.",
        "Things are steady and mostly fine.",
    ],
    ("anxious", "seeking_help"): [
        "I am worried and could use some guidance.",
        "I feel anxious and I need help deciding what to do next.",
        "Can someone support me with this? I am feeling uncertain.",
        "I am stressed and looking for practical help.",
        "I need advice because this is making me nervous.",
    ],
    ("hopeful", "gratitude"): [
        "I am grateful for the support I received today.",
        "Thank you, I feel hopeful about what comes next.",
        "I really appreciate this help and I feel positive now.",
        "I am thankful and optimistic after this progress.",
        "This kindness helped me a lot, and I feel hopeful.",
    ],
}


COLOR_BY_EMOTION = {
    "happy": (253, 226, 82),
    "sad": (95, 145, 210),
    "angry": (220, 90, 90),
    "neutral": (170, 170, 170),
    "anxious": (160, 130, 210),
    "hopeful": (120, 190, 145),
}


def build_demo_image(emotion: str, out_path: Path) -> None:
    bg = COLOR_BY_EMOTION[emotion]
    img = Image.new("RGB", (224, 224), bg)
    draw = ImageDraw.Draw(img)
    # Draw simple face-like pattern to avoid identical image tensors.
    draw.ellipse((60, 60, 164, 164), outline=(20, 20, 20), width=4)
    draw.ellipse((90, 95, 102, 107), fill=(20, 20, 20))
    draw.ellipse((122, 95, 134, 107), fill=(20, 20, 20))
    if emotion == "happy":
        draw.arc((90, 110, 134, 145), start=10, end=170, fill=(20, 20, 20), width=3)
    elif emotion == "sad":
        draw.arc((90, 128, 134, 160), start=190, end=350, fill=(20, 20, 20), width=3)
    elif emotion == "angry":
        draw.line((90, 140, 134, 125), fill=(20, 20, 20), width=3)
    else:
        draw.line((94, 136, 130, 136), fill=(20, 20, 20), width=3)
    img.save(out_path, format="PNG")


def main() -> None:
    random.seed(7)
    root = Path("data")
    image_dir = root / "presentation_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "presentation_manifest.csv"

    rows = []
    for (emotion, intent), texts in TEXT_BANK.items():
        for i in range(40):
            text = random.choice(texts)
            # Vary intensity per class to help regression head.
            if emotion == "happy":
                intensity = round(random.uniform(0.15, 0.55), 3)
            elif emotion == "neutral":
                intensity = round(random.uniform(0.25, 0.55), 3)
            else:
                intensity = round(random.uniform(0.55, 0.95), 3)

            image_name = f"{emotion}_{intent}_{i:03d}.png"
            image_path = image_dir / image_name
            build_demo_image(emotion, image_path)
            rows.append(
                {
                    "text": text,
                    "image_path": str(image_path).replace("\\", "/"),
                    "emotion": EMOTION_TO_LABEL[emotion],
                    "intensity": intensity,
                    "intent": INTENT_TO_LABEL[intent],
                }
            )

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "image_path", "emotion", "intensity", "intent"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
