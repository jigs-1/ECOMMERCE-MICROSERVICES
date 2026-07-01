import os
import torch
import pytest
from transformers import AutoTokenizer

from cameo.core.inference.pipeline import CameoPipeline

pytestmark = pytest.mark.skipif(
    os.getenv("CAMEO_RUN_INTEGRATION") != "1",
    reason="Set CAMEO_RUN_INTEGRATION=1 to run heavy model-integration tests.",
)


@torch.inference_mode()
def test_fusion_weights_sum_to_one():
    pipeline = CameoPipeline(device="cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    toks = tokenizer("hello world", return_tensors="pt", padding="max_length", truncation=True, max_length=16)
    img = torch.randn(1, 3, 224, 224)

    pred = pipeline(toks, img)
    w_sum = sum(pred.attn_weights.values())
    assert abs(w_sum - 1.0) < 1e-4
    assert 0.0 <= pred.gates["text"] <= 1.0
    assert 0.0 <= pred.gates["image"] <= 1.0


@torch.inference_mode()
def test_output_shapes_and_keys():
    pipeline = CameoPipeline(device="cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    toks = tokenizer("sample caption", return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    img = torch.randn(1, 3, 224, 224)

    pred = pipeline(toks, img)
    assert set(pred.emotion_probs.keys()) == set(pipeline.emotions)
    assert pred.intent in pipeline.intents
    assert isinstance(pred.intensity, float)
