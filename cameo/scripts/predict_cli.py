import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from PIL import Image

from cameo.core.inference.pipeline import CameoPipeline
from cameo.core.preprocess.text import clean_text
from cameo.core.preprocess.image import preprocess_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None, help="Path to saved .pt weights (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    pipeline = CameoPipeline(device=device)
    if args.weights:
        pipeline.load_state_dict(torch.load(args.weights, map_location=device))

    text_clean = clean_text(args.text)
    toks = tokenizer(text_clean, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    img = Image.open(args.image).convert("RGB")
    img_tensor = preprocess_image(img).unsqueeze(0)

    pred = pipeline(toks, img_tensor)
    print("--- Prediction ---")
    print(f"Emotion: {pred.emotion} probs={pred.emotion_probs}")
    print(f"Intensity: {pred.intensity:.3f}")
    print(f"Intent: {pred.intent}")
    print(f"Attention: {pred.attn_weights}  Gates: {pred.gates}")
    print(f"Response ({pred.response.mode}): {pred.response.text}")


if __name__ == "__main__":
    main()
