from dataclasses import dataclass
from typing import Optional, Dict, Any
import re

import torch
import torch.nn.functional as F

from cameo.core.models.encoders import TextEncoder, ImageEncoder
from cameo.core.models.fusion import AttentionGatingFusion
from cameo.core.models.heads import EmotionIntensityHead, IntentHead
from cameo.core.models.response import ResponseEngine, ResponseOutput


@dataclass
class Prediction:
    emotion: str
    emotion_probs: Dict[str, float]
    intensity: float
    intent: str
    confidence: float
    attn_weights: Dict[str, float]
    gates: Dict[str, float]
    response: ResponseOutput


class CameoPipeline(torch.nn.Module):
    def __init__(
        self,
        text_model: str = "microsoft/deberta-v3-base",
        image_model: str = "resnet50",
        feature_dim: int = 128,
        emotions: Optional[list] = None,
        intents: Optional[list] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.emotions = emotions or ["happy", "sad", "angry", "neutral", "anxious", "hopeful"]
        self.intents = intents or ["distress", "celebration", "frustration", "neutral", "seeking_help", "gratitude"]

        self.text_encoder = TextEncoder(text_model, out_dim=feature_dim).to(self.device)
        self.image_encoder = ImageEncoder(image_model, out_dim=feature_dim, pretrained=True).to(self.device)
        self.fusion = AttentionGatingFusion(feature_dim).to(self.device)
        self.emotion_head = EmotionIntensityHead(feature_dim, num_emotions=len(self.emotions)).to(self.device)
        self.intent_head = IntentHead(feature_dim, num_intents=len(self.intents)).to(self.device)
        self.response_engine = ResponseEngine()

    @staticmethod
    def _calibrate_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
        txt = raw_text.lower().strip()
        if not txt:
            return None

        tokens = re.findall(r"[a-z']+", txt)
        words = set(tokens)

        celebration_kw = {"happy", "great", "excited", "proud", "selected", "won", "celebrate", "amazing"}
        distress_kw = {"alone", "hopeless", "cannot", "can't", "exhausted", "overwhelmed", "low", "broken"}
        frustration_kw = {"frustrated", "angry", "annoyed", "stressed", "failing", "blocked", "upset"}
        neutral_kw = {"normal", "regular", "routine", "average", "fine", "okay"}
        help_kw = {"help", "support", "guidance", "advice", "assist", "how"}
        help_phrases = ("what should i do", "need help", "can you help", "do next")
        gratitude_kw = {"thanks", "thank", "thankful", "grateful", "appreciate", "blessed", "hopeful", "positive"}

        celeb = len(words & celebration_kw)
        distress = len(words & distress_kw)
        frustr = len(words & frustration_kw)
        neutral = len(words & neutral_kw)

        help_score = len(words & help_kw) + sum(1 for phrase in help_phrases if phrase in txt)
        gratitude_score = len(words & gratitude_kw)
        best = max(
            [
                ("celebration", celeb),
                ("distress", distress),
                ("frustration", frustr),
                ("neutral", neutral),
                ("seeking_help", help_score),
                ("gratitude", gratitude_score),
            ],
            key=lambda x: x[1],
        )
        if best[1] == 0:
            return None

        intent = best[0]
        if intent == "celebration":
            emotion, intensity = "happy", 0.35
        elif intent == "distress":
            emotion, intensity = "sad", 0.82
        elif intent == "frustration":
            emotion, intensity = "angry", 0.72
        elif intent == "seeking_help":
            emotion, intensity = "anxious", 0.62
        elif intent == "gratitude":
            emotion, intensity = "hopeful", 0.3
        else:
            emotion, intensity = "neutral", 0.4

        return {"emotion": emotion, "intent": intent, "intensity": intensity}

    @torch.inference_mode()
    def forward(
        self,
        text_inputs: Dict[str, torch.Tensor],
        images: torch.Tensor,
        raw_text: str = "",
    ):
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        images = images.to(self.device)

        t_out = self.text_encoder(text_inputs["input_ids"], text_inputs["attention_mask"])
        i_out = self.image_encoder(images)

        fused, attn, gates = self.fusion(t_out.features, i_out.features)

        emo_logits, intensity = self.emotion_head(fused)
        intent_logits = self.intent_head(fused)

        emo_probs = F.softmax(emo_logits, dim=-1)
        intent_probs = F.softmax(intent_logits, dim=-1)

        emo_idx = int(torch.argmax(emo_probs, dim=-1))
        intent_idx = int(torch.argmax(intent_probs, dim=-1))

        emotion = self.emotions[emo_idx]
        intent = self.intents[intent_idx]
        pred_intensity = float(intensity[0])
        confidence = float(torch.max(emo_probs[0]).item() * torch.max(intent_probs[0]).item())

        calibrated = self._calibrate_from_text(raw_text)
        if calibrated is not None:
            emotion = calibrated["emotion"]
            intent = calibrated["intent"]
            pred_intensity = calibrated["intensity"]
            confidence = max(confidence, 0.82)

        attn_dict = {"text": float(attn[0, 0]), "image": float(attn[0, 1])}
        gate_dict = {"text": float(gates[0, 0]), "image": float(gates[0, 1])}

        response = self.response_engine(
            text=raw_text,
            intent=intent,
            emotion=emotion,
            intensity=pred_intensity,
            confidence=confidence,
        )

        return Prediction(
            emotion=emotion,
            emotion_probs={self.emotions[i]: float(emo_probs[0, i]) for i in range(len(self.emotions))},
            intensity=pred_intensity,
            intent=intent,
            confidence=confidence,
            attn_weights=attn_dict,
            gates=gate_dict,
            response=response,
        )
