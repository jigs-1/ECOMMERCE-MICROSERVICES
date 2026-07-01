# CAMEO Presentation Narrative Plan

## Audience
College viva / project presentation panel.

## Objective
Explain CAMEO clearly as a compact multimodal system that is technically sound, visually understandable, and easy to present within a short talk.

## Narrative Arc
1. Introduce the problem and why multimodal understanding is needed.
2. Show the full workflow from input to output.
3. Explain the core neural architecture and math.
4. Explain fusion, heads, confidence, calibration, and safety.
5. Show what data was used and why.
6. Explain training and evaluation logic.
7. Close with strengths, limitations, future work, and Member 3 contribution.

## Slide List
1. Cover and project promise
2. Problem, objective, and end-to-end flow
3. Neural architecture from projection to output
4. Attention and gating fusion logic
5. Prediction heads, confidence, calibration, and response engine
6. Dataset design and why stylized images were used
7. Training logic and loss functions
8. Evaluation logic and holdout results
9. Realistic testing, safety checks, and product layer
10. Strengths, limitations, future work, and Member 3 role

## Source Plan
- README.md
- PROJECT_DOSSIER_READABLE.md
- EVALUATION_REPORT.md
- artifacts/eval_metrics_clean.json
- code in pipeline.py, fusion.py, heads.py, response.py, train_multimodal.py, train_cached_heads.py, api/store.py

## Visual System
- Warm academic palette with emerald, orange, sky, gold, and coral accents
- Light background, dense content panels, clear section badges
- Minimal but confident technical style
- One chart slide using native chart objects

## Editability Plan
- All meaningful text remains editable PowerPoint text
- Metrics and formulas are editable text boxes
- Evaluation figure is a native chart object
- Speaker notes hold slide intent
