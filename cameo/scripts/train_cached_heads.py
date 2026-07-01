"""
Faster CAMEO training path for CPU evaluation runs.
Caches frozen encoder outputs once, then trains only projection/fusion/head layers.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from cameo.core.data.dataset import MultimodalDataset
from cameo.core.inference.pipeline import CameoPipeline


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cache_features(model, dataloader, device):
    text_pooled = []
    image_pooled = []
    emotions = []
    intensities = []
    intents = []

    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)

            t_out = model.text_encoder(input_ids, attention_mask)
            i_out = model.image_encoder(images)

            text_pooled.append(t_out.pooled.cpu())
            image_pooled.append(i_out.pooled.cpu())
            emotions.append(batch["emotion"].cpu())
            intensities.append(batch["intensity"].cpu())
            intents.append(batch["intent"].cpu())

    return TensorDataset(
        torch.cat(text_pooled, dim=0),
        torch.cat(image_pooled, dim=0),
        torch.cat(emotions, dim=0),
        torch.cat(intensities, dim=0),
        torch.cat(intents, dim=0),
    )


def train_epoch(model, dataloader, optimizer, scaler, device, lambda_intensity: float):
    model.train()
    total_loss = 0.0
    for text_pooled, image_pooled, emotion, intensity, intent in dataloader:
        text_pooled = text_pooled.to(device).to(model.text_encoder.proj.weight.dtype)
        image_pooled = image_pooled.to(device).to(model.image_encoder.proj.weight.dtype)
        emotion = emotion.to(device)
        intensity = intensity.to(device)
        intent = intent.to(device)

        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            text_feat = model.text_encoder.proj(text_pooled)
            image_feat = model.image_encoder.proj(image_pooled)
            fused, _, _ = model.fusion(text_feat, image_feat)
            emo_logits, intensity_pred = model.emotion_head(fused)
            intent_logits = model.intent_head(fused)

            loss_emo = F.cross_entropy(emo_logits, emotion)
            loss_intent = F.cross_entropy(intent_logits, intent)
            loss_intensity = F.mse_loss(intensity_pred, intensity)
            loss = loss_emo + loss_intent + lambda_intensity * loss_intensity

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * emotion.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lambda-intensity", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="artifacts/cameo_cached_eval.pt")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    raw_dataset = MultimodalDataset(args.manifest, tokenizer=tokenizer)
    cache_loader = DataLoader(raw_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CameoPipeline(device=device)
    model.to(device)

    cached_dataset = cache_features(model, cache_loader, device)
    train_loader = DataLoader(cached_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    params = (
        list(model.text_encoder.proj.parameters())
        + list(model.image_encoder.proj.parameters())
        + list(model.fusion.parameters())
        + list(model.emotion_head.parameters())
        + list(model.intent_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device, args.lambda_intensity)
        print(f"Epoch {epoch} loss {loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved weights to {output_path}")


if __name__ == "__main__":
    main()
