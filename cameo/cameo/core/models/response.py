from dataclasses import dataclass
from typing import Dict, Optional
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DISTRESS_THRESHOLD = 0.7
ETHICS_NOTE = (
    " This support is based on AI classification, so it may be imperfect and is not a medical diagnosis."
)


@dataclass
class ResponseOutput:
    text: str
    mode: str  # "rule" or "generative"


class ResponseEngine:
    """
    Hybrid responder: rule-based for distress, generative otherwise.
    FLAN-T5 path is optional and can be enabled via env vars.
    """

    def __init__(self, rules: Optional[Dict[str, str]] = None):
        self.rules = rules or {
            "distress": "I'm sorry you're going through this. I'm here to listen. If you need urgent help, please contact a trusted person or local support line.",
            "default": "Thanks for sharing! I'm here to help if you want to talk more.",
        }
        # Optional FLAN-T5 path. Disabled by default for deterministic offline demos.
        self.use_llm = os.getenv("CAMEO_USE_FLAN", "0") == "1"
        self.flan_model_name = os.getenv("CAMEO_FLAN_MODEL", "google/flan-t5-small")
        self._llm_tokenizer = None
        self._llm_model = None
        self._llm_failed = False

    def _ensure_llm(self) -> bool:
        if not self.use_llm or self._llm_failed:
            return False
        if self._llm_tokenizer is not None and self._llm_model is not None:
            return True
        try:
            self._llm_tokenizer = AutoTokenizer.from_pretrained(self.flan_model_name, use_fast=True)
            self._llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.flan_model_name)
            self._llm_model.eval()
            return True
        except Exception:
            self._llm_failed = True
            return False

    def rule_based(self, intent: str, emotion: str, intensity: float) -> Optional[str]:
        if intent == "distress" or (emotion == "sad" and intensity >= DISTRESS_THRESHOLD):
            return self.rules.get("distress", None)
        return None

    def generate(self, text: str, intent: str, emotion: str, intensity: float) -> str:
        # Prefer FLAN-T5 when enabled; otherwise deterministic supportive fallback.
        if self._ensure_llm():
            prompt = (
                "Write a brief, warm, supportive reply (2-3 sentences). "
                "Do not mention model analysis, probabilities, or image details. "
                "Tone should be empathetic and encouraging.\n"
                f"User text: {text}\n"
                f"Intent: {intent}\n"
                f"Emotion: {emotion}\n"
                f"Intensity: {intensity:.2f}\n"
                "Reply:"
            )
            inputs = self._llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            with torch.inference_mode():
                output_ids = self._llm_model.generate(
                    **inputs,
                    max_new_tokens=80,
                    num_beams=4,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
            text_out = self._llm_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if text_out:
                return text_out

        if intent == "celebration":
            return "That is wonderful to hear. You have worked for this, and you deserve this moment. Keep going, one step at a time."
        if intent == "gratitude":
            return "Thank you for sharing that. Your gratitude and perspective are powerful, and it is great to see your positive momentum."
        if intent == "frustration":
            return "That sounds really tough. You are not alone in this, and your effort still matters. Let's take one small step forward together."
        if intent == "seeking_help":
            return "Reaching out for help is a strong step. We can break this into small, manageable actions and move forward together."
        if emotion in {"sad", "angry"}:
            return "I am here with you. It makes sense to feel this way, and things can get better from here. You can pause, breathe, and move gently."
        return "Thanks for sharing this. You are doing your best, and that is meaningful. I am here to support you."

    def __call__(self, text: str, intent: str, emotion: str, intensity: float, confidence: float = 1.0) -> ResponseOutput:
        rule = self.rule_based(intent, emotion, intensity)
        if rule:
            prefix = "I may be wrong, but " if confidence < 0.45 else ""
            return ResponseOutput(text=f"{prefix}{rule}{ETHICS_NOTE}", mode="rule")
        generated = self.generate(text, intent, emotion, intensity)
        if confidence < 0.45:
            generated = f"I may be wrong, but {generated[0].lower()}{generated[1:]}"
        return ResponseOutput(text=f"{generated}{ETHICS_NOTE}", mode="generative")
