import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGatingFusion(nn.Module):
    """
    Simple modality attention + gating.
    Inputs: text_feat (B, D), image_feat (B, D)
    Outputs: fused (B, D), attn_weights (B, 2), gates (B, 2)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim * 2, 2)
        self.gate = nn.Linear(dim * 2, 2)

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        stacked = torch.cat([text_feat, image_feat], dim=-1)  # (B, 2D)
        attn_logits = self.attn(stacked)  # (B, 2)
        attn_weights = F.softmax(attn_logits, dim=-1)
        gates = torch.sigmoid(self.gate(stacked))  # (B, 2)
        # expand weights to match feature dims
        t_weight = attn_weights[:, 0:1] * gates[:, 0:1]
        i_weight = attn_weights[:, 1:2] * gates[:, 1:2]
        fused = t_weight * text_feat + i_weight * image_feat
        return fused, attn_weights, gates
