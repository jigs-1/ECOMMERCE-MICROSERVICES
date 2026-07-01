import torch
import torch.nn as nn


class EmotionIntensityHead(nn.Module):
    def __init__(self, in_dim: int, num_emotions: int = 4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.emotion = nn.Linear(64, num_emotions)
        self.intensity = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, fused: torch.Tensor):
        h = self.trunk(fused)
        logits = self.emotion(h)
        intensity = self.intensity(h).squeeze(-1)
        return logits, intensity


class IntentHead(nn.Module):
    def __init__(self, in_dim: int, num_intents: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_intents),
        )

    def forward(self, fused: torch.Tensor):
        return self.net(fused)
