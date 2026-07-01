from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torchvision.models as tvm
import timm


@dataclass
class EncoderOutputs:
    features: torch.Tensor  # shape (batch, dim)
    pooled: Optional[torch.Tensor] = None


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", out_dim: int = 128, trainable: bool = False):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.proj = nn.Linear(self.encoder.config.hidden_size, out_dim)
        self.trainable = trainable
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def tokenizer(model_name: str):
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EncoderOutputs:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token pooled output
        cls = outputs.last_hidden_state[:, 0, :]
        # Some pretrained checkpoints can emit fp16; keep projection dtype consistent.
        cls = cls.to(self.proj.weight.dtype)
        proj = self.proj(cls)
        return EncoderOutputs(features=proj, pooled=cls)


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "resnet50", out_dim: int = 128, pretrained: bool = True, use_timm: bool = False):
        super().__init__()
        self.model_name = model_name
        self.use_timm = use_timm
        if use_timm:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
            feat_dim = self.backbone.num_features
        else:
            backbone = getattr(tvm, model_name)(weights="IMAGENET1K_V2" if pretrained else None)
            modules = list(backbone.children())[:-1]  # drop fc
            self.backbone = nn.Sequential(*modules)
            feat_dim = backbone.fc.in_features
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, images: torch.Tensor) -> EncoderOutputs:
        feats = self.backbone(images)
        if feats.dim() == 4:  # (B, C, 1, 1)
            feats = torch.flatten(feats, 1)
        proj = self.proj(feats)
        return EncoderOutputs(features=proj, pooled=feats)
