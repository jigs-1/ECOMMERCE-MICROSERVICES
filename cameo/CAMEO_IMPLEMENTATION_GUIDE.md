# CAMEO Implementation Guide

## Goal

This checklist defines:

- what each module should output
- how to know when each stage is complete
- how to verify the system in this repository

---

## 1. Text Module

### Input

- Raw text, such as a social media caption

### Process

- preprocessing
- tokenization
- transformer encoding

### Expected Output

- text feature vector
- typical shape:
  - encoder output near `[1, 768]`
  - projected feature near `[1, 128]`

### Completion Check

- emotion-related text patterns are reflected in predictions on sample inputs
- feature vector extraction succeeds without errors

### Repo Touchpoints

- [text.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/preprocess/text.py)
- [encoders.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/encoders.py)

---

## 2. Image Module

### Input

- image input, such as a face or post image

### Process

- resize
- normalize
- CNN encoding

### Expected Output

- image feature vector
- typical shape:
  - backbone pooled feature near `[1, 512]` or model-specific equivalent
  - projected feature near `[1, 128]`

### Completion Check

- image preprocessing succeeds on sample images
- feature extraction works consistently
- predictions react to visually different inputs

### Repo Touchpoints

- [image.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/preprocess/image.py)
- [encoders.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/encoders.py)

---

## 3. Preprocessing Module

### Completion Check

- text is cleaned consistently
- URLs, mentions, hashtags, and noise are removed properly
- images are resized consistently
- images are normalized consistently

### Repo Touchpoints

- [text.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/preprocess/text.py)
- [image.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/preprocess/image.py)

---

## 4. Fusion Module (Attention + Gating)

### Input

- text features
- image features

### Process

- concatenate features
- compute attention weights
- apply gating
- combine modalities into one fused vector

### Expected Output

- fused feature vector, typically `[1, 128]`

### Completion Check

- attention weights sum to `1`
- different inputs can produce different attention distributions
- gating values stay in valid range

### Repo Touchpoints

- [fusion.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/fusion.py)
- [test_core_units.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_core_units.py)

---

## 5. Emotion + Intensity Module

### Input

- fused vector

### Expected Output

- emotion label
- intensity score

### Completion Check

- emotion predictions are reasonable for validation inputs
- intensity changes with meaningfully different inputs

### Repo Touchpoints

- [heads.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/heads.py)
- [pipeline.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/inference/pipeline.py)

---

## 6. Intent Detection Module

### Input

- text or fused features

### Expected Output

- intent label

### Completion Check

- correct classification for:
  - distress
  - celebration or happy cases
  - neutral cases
  - seeking-help cases

### Repo Touchpoints

- [pipeline.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/inference/pipeline.py)
- [test_core_units.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_core_units.py)

---

## 7. Response Engine

### Input

- emotion
- intent
- intensity
- confidence

### Process

- decision layer:
  - if distress, use rule-based response
  - otherwise, use deterministic or generative response path

### Expected Output

- safe supportive response text

### Completion Check

- responses match the predicted context
- distress uses the guarded rule-based path
- outputs do not contain unsafe guidance

### Repo Touchpoints

- [response.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/response.py)
- [test_core_units.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_core_units.py)

---

## 8. End-to-End System

### Input

- text
- image

### Expected Output

- emotion
- intensity
- intent
- response

### Completion Check

- the full pipeline runs without errors
- outputs are coherent and meaningful
- API returns the expected contract
- UI displays the returned outputs clearly

### Repo Touchpoints

- [main.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/api/main.py)
- [App.tsx](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/ui/src/App.tsx)
- [test_api.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_api.py)

---

## 9. Final Validation Checklist

- [x] text model path working
- [x] image preprocessing and feature path working
- [x] fusion produces valid weights
- [x] intent detection has baseline rule calibration checks
- [x] response system functional
- [x] end-to-end API integration working
- [x] UI build passes
- [x] benchmarked improvement over baseline
- [x] formal evaluation dataset metrics recorded

Evaluation artifacts:

- [eval_metrics_clean.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/eval_metrics_clean.json)
- [EVALUATION_REPORT.md](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/EVALUATION_REPORT.md)

---

## 10. Success Criteria

- model quality improves over baseline
- system produces meaningful responses
- UI clearly displays:
  - emotion
  - intensity
  - intent
  - confidence
  - supportive response

---

## 11. Common Failure Points

- feature size mismatch between modules
- poor preprocessing quality
- overfitting during training
- incorrect fusion implementation
- invalid or corrupt image uploads
- lazy-load assumptions not reflected in deployment expectations

---

## 12. Final Deliverable

A working system that:

- accepts multimodal input
- understands emotion and intent
- generates intelligent supportive responses
- exposes the pipeline through API and UI

---

## Execution Commands

### Run tests

```powershell
.\.venv313\Scripts\python.exe -m pytest -q
```

### Run API

```powershell
.\.venv313\Scripts\python.exe -m uvicorn cameo.api.main:app --reload
```

### Run UI

```powershell
cd ui
npm run dev
```

### Build UI

```powershell
cd ui
npm run build
```
