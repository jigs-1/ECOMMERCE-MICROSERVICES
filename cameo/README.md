# CAMEO: Context-Aware Multimodal Emotion & Intent Engine

This repo scaffolds a performance-first multimodal pipeline that:
- accepts text + image,
- extracts features with pretrained encoders (BERT + ResNet/ConvNeXt),
- fuses them via attention + gating,
- predicts emotion class, intensity, intent,
- generates a safe response (rule-first, generative fallback),
- and now supports local user accounts, saved prediction history, and trend analysis.

## Stack
- Python 3.10+
- PyTorch 2.x, TorchVision / timm
- Hugging Face `transformers`
- FastAPI for serving

## Layout
```
cameo/
  api/               FastAPI entrypoint
  core/
    preprocess/      Text & image preprocessing
    models/          Encoders, fusion, heads, response engine
    inference/       End-to-end pipeline wiring
  tests/             Shape + logic checks
```

## Quick start
```
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
pytest
uvicorn cameo.api.main:app --reload
```

If you are using the project-created Python 3.13 env:
```
.\.venv313\Scripts\python.exe -m pip install -r requirements.txt
.\.venv313\Scripts\python.exe -m pytest -q
.\.venv313\Scripts\python.exe -m uvicorn cameo.api.main:app --reload
```

Run heavier end-to-end model tests (downloads pretrained weights/tokenizer on first run):
```
$env:CAMEO_RUN_INTEGRATION="1"
python -m pytest -q
```

## Train (baseline)
```
python scripts/train_multimodal.py --manifest data/sample_manifest.csv --epochs 1 --batch-size 2
```
This trains only projection/fusion/heads by default and saves `artifacts/cameo.pt`.
Default label spaces:
- Emotions: `happy, sad, angry, neutral, anxious, hopeful`
- Intents: `distress, celebration, frustration, neutral, seeking_help, gratitude`

## Presentation-ready tuning
Generate a larger synthetic demo dataset and fine-tune:
```
python scripts/generate_presentation_data.py
python scripts/train_multimodal.py --manifest data/presentation_manifest.csv --epochs 2 --batch-size 8 --num-workers 0 --device cpu
```
When `artifacts/cameo.pt` exists, the API auto-loads it at startup.

## CLI predict
```
python scripts/predict_cli.py --text "I feel great" --image path/to/img.jpg --weights artifacts/cameo.pt
```

## API predict
POST `/predict` with `multipart/form-data` fields `text` and `image` (png/jpeg). Returns emotion, intensity, intent, attention/gates, and response text.

Operational endpoints:
- `GET /health` is a liveness probe and returns only API status/version.
- `GET /ready` is a readiness probe and returns `503` until the app has the assets it needs to serve traffic.
- `POST /auth/register` and `POST /auth/login` create local demo accounts and sessions.
- `GET /history` and `GET /trends` return per-user saved checks and summary analytics.

Runtime env vars:
```powershell
CAMEO_DEVICE=cpu
CAMEO_WARM_START=1
CAMEO_REQUIRE_WEIGHTS=1
CAMEO_WEIGHTS_PATH=artifacts/cameo.pt
CAMEO_DATABASE_PATH=artifacts/cameo.db
CAMEO_LOCAL_FILES_ONLY=1
CAMEO_CORS_ORIGINS=https://your-frontend.example.com
CAMEO_RATE_LIMIT_REQUESTS=30
CAMEO_RATE_LIMIT_WINDOW_SECONDS=60
CAMEO_API_KEY=replace-with-a-strong-secret
```

Recommended production settings:
- `CAMEO_WARM_START=1` so startup pays model-load cost before the first request.
- `CAMEO_REQUIRE_WEIGHTS=1` so the service fails closed if trained weights are missing.
- `CAMEO_LOCAL_FILES_ONLY=1` so production does not try to fetch model assets from the network at request time.
- `CAMEO_WEIGHTS_PATH` if the model file is stored outside the default `artifacts/cameo.pt`.
- `CAMEO_DATABASE_PATH` if you want user accounts and saved analytics stored somewhere else.
- `CAMEO_API_KEY` to gate `/predict` behind a shared secret header.
- `CAMEO_RATE_LIMIT_*` to protect the demo from bursts during judging.

Optional FLAN-T5 generation (instead of deterministic template replies):
```
$env:CAMEO_USE_FLAN="1"
$env:CAMEO_FLAN_MODEL="google/flan-t5-small"
```

## ONNX export
```
python scripts/export_onnx.py --output-dir artifacts
```
Exports `text_encoder.onnx`, `image_encoder.onnx`, `fusion_heads.onnx`.

## Frontend (React + Vite)
```
cd ui
npm install
npm run dev        # localhost:5173
npm run build      # outputs ui/dist
```
Optionally set API base URL in `ui/.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```
After `npm run build`, FastAPI will serve the static UI automatically from `ui/dist` when you run `uvicorn cameo.api.main:app`.

## Tests
```
.\.venv313\Scripts\python.exe -m pytest -q
```
Current automated coverage includes:
- fusion and response-engine unit checks
- text calibration logic checks
- `/health` and `/predict` API contract/validation tests
- auth, saved history, and trend endpoint flow
- response safety prompt checks

## Results

Holdout evaluation on the balanced presentation split:

| Model | Emotion Acc | Intent Acc | Intensity MAE |
|---|---:|---:|---:|
| Majority baseline | 0.1667 | 0.1667 | 0.1975 |
| Heuristic baseline | 0.8000 | 0.8000 | 0.1623 |
| CAMEO | 0.8500 | 0.8500 | 0.1550 |

Realistic unseen-case check:

- `12` hand-written cases
- emotion accuracy: `0.6667`
- intent accuracy: `0.6667`
- main failure pattern: distress over-prediction on strongly negative wording

Response safety check:

- `6` curated prompts
- pass rate: `1.0000`

Detailed artifacts:

- [EVALUATION_REPORT.md](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/EVALUATION_REPORT.md)
- [eval_metrics_clean.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/eval_metrics_clean.json)
- [realistic_eval_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/realistic_eval_results.json)
- [response_safety_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/response_safety_results.json)

## Production Run
Local process:
```powershell
$env:CAMEO_WARM_START="1"
$env:CAMEO_REQUIRE_WEIGHTS="1"
$env:CAMEO_LOCAL_FILES_ONLY="1"
.\.venv313\Scripts\python.exe -m uvicorn cameo.api.main:app --host 0.0.0.0 --port 8000
```

Docker:
```powershell
docker build -t cameo:prod .
docker run --rm -p 8000:8000 cameo:prod
```

Docker Compose:
```powershell
Copy-Item .env.example .env
docker compose up --build
```

Operational notes:
- Launch from any working directory safely; model weights are now resolved from the repository root unless `CAMEO_WEIGHTS_PATH` overrides them.
- Prefer one process per model copy unless you intentionally want multiple workers and the extra memory usage that comes with duplicated weights.
- Use `/health` for liveness and `/ready` for traffic gating.
- Responses include `X-Request-ID` and `X-Process-Time` headers for tracing and timing.
- `/predict` now supports optional API-key auth plus lightweight in-memory rate limiting.
- The frontend now surfaces live service/readiness status for demos and submission screenshots.

## Status
Repository now includes trained weights and a production-oriented API contract, but real production readiness still depends on your deployment standards: monitoring, external auth/rate limiting, and validated model quality on your real traffic.
