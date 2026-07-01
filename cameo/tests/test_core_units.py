import torch

from cameo.core.models.fusion import AttentionGatingFusion
from cameo.core.models.response import ResponseEngine
from cameo.core.inference.pipeline import CameoPipeline


@torch.inference_mode()
def test_attention_weights_form_distribution():
    fusion = AttentionGatingFusion(dim=8)
    text = torch.randn(2, 8)
    image = torch.randn(2, 8)

    _, attn, gates = fusion(text, image)
    sums = attn.sum(dim=-1)

    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert torch.all(gates >= 0.0)
    assert torch.all(gates <= 1.0)


def test_response_engine_rule_and_generative_modes():
    engine = ResponseEngine()

    distress = engine(text="help", intent="distress", emotion="sad", intensity=0.9)
    assert distress.mode == "rule"
    assert "sorry" in distress.text.lower()
    assert "based on ai classification" in distress.text.lower()

    neutral = engine(text="things are okay", intent="neutral", emotion="happy", intensity=0.2)
    assert neutral.mode == "generative"
    assert "support" in neutral.text.lower()
    assert "based on ai classification" in neutral.text.lower()


def test_text_calibration_detects_help_phrases():
    calibrated = CameoPipeline._calibrate_from_text("I feel stuck and need help with what should I do next.")

    assert calibrated is not None
    assert calibrated["intent"] == "seeking_help"
    assert calibrated["emotion"] == "anxious"
