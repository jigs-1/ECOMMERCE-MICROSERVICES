"""
Export CAMEO pipeline to ONNX for faster inference.
Exports text encoder (projected CLS), image encoder (projected pooled), and fusion+heads block.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cameo.core.inference.pipeline import CameoPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to place onnx files")
    parser.add_argument("--text-model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--image-model", type=str, default="resnet50")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    pipeline = CameoPipeline(text_model=args.text_model, image_model=args.image_model, device=args.device)
    pipeline.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Export text encoder projection
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
    dummy = tokenizer("hello world", return_tensors="pt", padding="max_length", truncation=True, max_length=16)
    dummy = {k: v.to(device) for k, v in dummy.items()}
    text_enc = pipeline.text_encoder
    text_path = out_dir / "text_encoder.onnx"
    torch.onnx.export(
        text_enc,
        (dummy["input_ids"], dummy["attention_mask"]),
        text_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["features", "pooled"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "features": {0: "batch"}, "pooled": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved {text_path}")

    # Export image encoder projection
    img_dummy = torch.randn(1, 3, 224, 224, device=device)
    img_enc = pipeline.image_encoder
    img_path = out_dir / "image_encoder.onnx"
    torch.onnx.export(
        img_enc,
        img_dummy,
        img_path,
        input_names=["image"],
        output_names=["features", "pooled"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}, "pooled": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved {img_path}")

    # Export fusion + heads block (takes fused text/image features)
    fusion_block = torch.nn.Sequential(
        pipeline.fusion,
        torch.nn.Identity(),  # placeholder to keep names consistent
    )

    # Wrapper to match inputs
    class FusionHeads(torch.nn.Module):
        def __init__(self, fusion, emo_head, intent_head):
            super().__init__()
            self.fusion = fusion
            self.emo = emo_head
            self.intent = intent_head

        def forward(self, text_feat, image_feat):
            fused, attn, gates = self.fusion(text_feat, image_feat)
            emo_logits, intensity = self.emo(fused)
            intent_logits = self.intent(fused)
            return fused, attn, gates, emo_logits, intent_logits, intensity

    fusion_heads = FusionHeads(pipeline.fusion, pipeline.emotion_head, pipeline.intent_head)
    text_feat_dummy = torch.randn(1, 128, device=device)
    img_feat_dummy = torch.randn(1, 128, device=device)
    fusion_path = out_dir / "fusion_heads.onnx"
    torch.onnx.export(
        fusion_heads,
        (text_feat_dummy, img_feat_dummy),
        fusion_path,
        input_names=["text_feat", "image_feat"],
        output_names=["fused", "attn", "gates", "emo_logits", "intent_logits", "intensity"],
        dynamic_axes={
            "text_feat": {0: "batch"},
            "image_feat": {0: "batch"},
            "fused": {0: "batch"},
            "attn": {0: "batch"},
            "gates": {0: "batch"},
            "emo_logits": {0: "batch"},
            "intent_logits": {0: "batch"},
            "intensity": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Saved {fusion_path}")


if __name__ == "__main__":
    main()
