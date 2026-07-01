# CAMEO Evaluation Report

## Summary

This report records a clean holdout evaluation for CAMEO using:

- train split: `180` samples
- eval split: `60` samples
- class layout: balanced across all `6` emotion and intent classes

Artifacts:

- [presentation_train_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_train_manifest.csv)
- [presentation_eval_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_eval_manifest.csv)
- [realistic_eval_cases.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/realistic_eval_cases.csv)
- [response_safety_cases.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/response_safety_cases.csv)
- [cameo_cached_eval.pt](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/cameo_cached_eval.pt)
- [eval_metrics_clean.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/eval_metrics_clean.json)
- [realistic_eval_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/realistic_eval_results.json)
- [response_safety_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/response_safety_results.json)

## Compared Systems

### Majority baseline

- predicts the most frequent train-set emotion and intent for every sample
- predicts mean train-set intensity for every sample

### Heuristic baseline

- uses text calibration keywords and phrases only
- ignores learned multimodal weights

### CAMEO model

- trained on the train split only
- evaluated on the holdout split
- uses cached frozen encoder outputs for faster CPU training of projection, fusion, and head layers

## Results

| Model | Emotion Acc | Emotion Macro-F1 | Intent Acc | Intent Macro-F1 | Intensity MAE |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.1667 | 0.0476 | 0.1667 | 0.0476 | 0.1975 |
| Heuristic baseline | 0.8000 | 0.7994 | 0.8000 | 0.7994 | 0.1623 |
| CAMEO model | 0.8500 | 0.8521 | 0.8500 | 0.8521 | 0.1550 |

## Interpretation

- CAMEO clearly improves over the majority baseline.
- CAMEO also improves over the text-only heuristic baseline:
  - `+0.05` emotion accuracy
  - `+0.05` intent accuracy
  - lower intensity error
- This is enough to satisfy the current repository success criterion of improving over baseline.

## Realistic Unseen Cases

The model was also checked on `12` hand-written unseen cases intended to feel more like real user phrasing.

- emotion accuracy: `0.6667`
- intent accuracy: `0.6667`
- joint accuracy: `0.6667`

### Main failure pattern

Several cases with heavy negative language were pulled into the `distress` class even when the better label was:

- `seeking_help`
- `frustration`
- `neutral`
- one celebration case

This suggests the current calibration rules are still too eager to map strong negative wording into distress.

### Example misses

- "I finally got the offer letter and I cannot stop smiling."
  - expected: `happy` / `celebration`
  - predicted: `sad` / `distress`
- "I am overwhelmed by deadlines and need someone to help me decide what to do first."
  - expected: `anxious` / `seeking_help`
  - predicted: `sad` / `distress`
- "Things are steady now. I am not excited or upset, just moving through the day."
  - expected: `neutral` / `neutral`
  - predicted: `happy` / `celebration`

## Response Safety Checks

The response engine was checked on `6` curated prompts covering:

- distress
- seeking help
- frustration
- celebration
- gratitude
- low-confidence distress

Results:

- pass rate: `1.0000`
- required supportive and safety phrases present in all cases
- forbidden unsafe phrases absent in all cases
- distress cases correctly stayed on the rule-based path

This does not replace a full red-team pass, but it gives the repo a repeatable safety sanity check.

## Caveats

- The dataset is synthetic and presentation-oriented, so these metrics should not be treated as production evidence.
- A stronger future evaluation should use more realistic unseen multimodal examples and safety review prompts.

## Reproduction Commands

### Create split

```powershell
.\.venv313\Scripts\python.exe scripts\create_eval_split.py --manifest data\presentation_manifest.csv
```

### Train clean eval checkpoint

```powershell
$env:PYTHONPATH=(Get-Location).Path
.\.venv313\Scripts\python.exe scripts\train_cached_heads.py --manifest data\presentation_train_manifest.csv --epochs 6 --batch-size 16 --num-workers 0 --device cpu --output artifacts\cameo_cached_eval.pt
```

### Evaluate against baselines

```powershell
$env:PYTHONPATH=(Get-Location).Path
.\.venv313\Scripts\python.exe scripts\evaluate_models.py --train-manifest data\presentation_train_manifest.csv --eval-manifest data\presentation_eval_manifest.csv --weights artifacts\cameo_cached_eval.pt --output artifacts\eval_metrics_clean.json --device cpu
```

### Evaluate realistic unseen cases

```powershell
$env:PYTHONPATH=(Get-Location).Path
.\.venv313\Scripts\python.exe scripts\evaluate_realistic_cases.py --cases data\realistic_eval_cases.csv --weights artifacts\cameo_cached_eval.pt --output artifacts\realistic_eval_results.json --device cpu
```

### Run response safety checks

```powershell
$env:PYTHONPATH=(Get-Location).Path
.\.venv313\Scripts\python.exe scripts\run_response_safety_checks.py --cases data\response_safety_cases.csv --output artifacts\response_safety_results.json
```
