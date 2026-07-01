"""
Lightweight training script for CAMEO.
Assumes a CSV manifest with columns: text,image_path,emotion,intensity,intent
Label indices should align with the default emotion/intent lists in CameoPipeline.
"""
import argparse
from pathlib import Path
from contextlib import nullcontext
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer

from cameo.core.data.dataset import MultimodalDataset
from cameo.core.inference.pipeline import CameoPipeline


def make_dataloaders(manifest: str, tokenizer, batch_size: int, num_workers: int):
    ds = MultimodalDataset(manifest, tokenizer=tokenizer)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_epoch(model, dataloader, optimizer, scaler, device, lambda_intensity: float):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        emotion = batch["emotion"].to(device)
        intensity = batch["intensity"].to(device)
        intent = batch["intent"].to(device)

        optimizer.zero_grad()
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            t_out = model.text_encoder(input_ids, attention_mask)
            i_out = model.image_encoder(images)
            fused, _, _ = model.fusion(t_out.features, i_out.features)
            emo_logits, intensity_pred = model.emotion_head(fused)
            intent_logits = model.intent_head(fused)

            loss_emo = F.cross_entropy(emo_logits, emotion)
            loss_intent = F.cross_entropy(intent_logits, intent)
            loss_intensity = F.mse_loss(intensity_pred, intensity)
            loss = loss_emo + loss_intent + lambda_intensity * loss_intensity

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="CSV manifest path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lambda-intensity", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="artifacts/cameo.pt", help="Output weights path")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    dataloader = make_dataloaders(args.manifest, tokenizer, args.batch_size, args.num_workers)

    model = CameoPipeline(device=device)
    model.to(device)

    # Only train projection + fusion + heads by default
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
        loss = train_epoch(model, dataloader, optimizer, scaler, device, args.lambda_intensity)
        print(f"Epoch {epoch} loss {loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved weights to {output_path}")


if __name__ == "__main__":
    main()
