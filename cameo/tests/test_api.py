from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from cameo.api.config import Settings
from cameo.api.main import create_app


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(120, 180, 200))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class _DummyPrediction:
    emotion = "anxious"
    intensity = 0.62
    intent = "seeking_help"
    confidence = 0.91
    attn_weights = {"text": 0.7, "image": 0.3}
    gates = {"text": 0.8, "image": 0.4}

    class response:
        text = "We can take this one step at a time."
        mode = "generative"


class _DummyPipeline:
    def __call__(self, tokenized, image_tensor, raw_text=""):
        return _DummyPrediction()


class _DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = Settings(
        app_name="CAMEO API",
        app_version="test",
        project_root=tmp_path,
        frontend_dist_dir=tmp_path / "ui" / "dist",
        weights_path=tmp_path / "artifacts" / "cameo.pt",
        database_path=tmp_path / "artifacts" / "cameo.db",
        text_model="microsoft/deberta-v3-base",
        device="cpu",
        warm_start=False,
        require_weights=False,
        local_files_only=False,
        cors_origins=(),
        api_key=None,
        rate_limit_requests=30,
        rate_limit_window_seconds=60,
        max_text_chars=1200,
        max_image_bytes=8 * 1024 * 1024,
    )
    return Settings(**{**base.__dict__, **overrides})


def test_health_reports_liveness(tmp_path):
    client = TestClient(create_app(_settings(tmp_path)))
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {"status": "ok", "version": "test"}
    assert "x-request-id" in response.headers
    assert "x-process-time" in response.headers


def test_ready_reports_degraded_before_model_load(tmp_path):
    client = TestClient(create_app(_settings(tmp_path)))
    response = client.get("/ready")

    assert response.status_code == 503
    payload = response.json()["detail"]
    assert payload["status"] == "degraded"
    assert payload["weights_present"] is False
    assert payload["weights_loaded"] is False


def test_predict_accepts_valid_request(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    monkeypatch.setattr(app.state.registry, "get_models", lambda: (_DummyPipeline(), _DummyTokenizer()))
    client = TestClient(app)
    response = client.post(
        "/predict",
        data={"text": "I need help figuring out what to do next."},
        files={"image": ("demo.png", _make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["emotion"] == "anxious"
    assert payload["intent"] == "seeking_help"
    assert payload["response"]["mode"] == "generative"


def test_predict_rejects_invalid_image_bytes(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    monkeypatch.setattr(app.state.registry, "get_models", lambda: (_DummyPipeline(), _DummyTokenizer()))
    client = TestClient(app)
    response = client.post(
        "/predict",
        data={"text": "hello"},
        files={"image": ("broken.png", b"not-a-real-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."


def test_predict_rejects_oversized_payload(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    monkeypatch.setattr(app.state.registry, "get_models", lambda: (_DummyPipeline(), _DummyTokenizer()))
    client = TestClient(app)
    too_big = b"x" * (_settings(tmp_path).max_image_bytes + 1)
    response = client.post(
        "/predict",
        data={"text": "hello"},
        files={"image": ("huge.png", too_big, "image/png")},
    )

    assert response.status_code == 400
    assert "Image is too large" in response.json()["detail"]


def test_predict_returns_service_unavailable_when_weights_are_required(tmp_path):
    client = TestClient(create_app(_settings(tmp_path, require_weights=True)))
    response = client.post(
        "/predict",
        data={"text": "hello"},
        files={"image": ("demo.png", _make_png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "Required weights file not found" in response.json()["detail"]


def test_predict_requires_api_key_when_configured(tmp_path):
    client = TestClient(create_app(_settings(tmp_path, api_key="secret-key")))
    response = client.post(
        "/predict",
        data={"text": "hello"},
        files={"image": ("demo.png", _make_png_bytes(), "image/png")},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key."


def test_predict_rate_limit_is_enforced(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path, rate_limit_requests=1, rate_limit_window_seconds=60))
    monkeypatch.setattr(app.state.registry, "get_models", lambda: (_DummyPipeline(), _DummyTokenizer()))
    client = TestClient(app)

    first = client.post(
        "/predict",
        data={"text": "hello"},
        files={"image": ("demo.png", _make_png_bytes(), "image/png")},
    )
    second = client.post(
        "/predict",
        data={"text": "hello again"},
        files={"image": ("demo.png", _make_png_bytes(), "image/png")},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Rate limit exceeded" in second.json()["detail"]


def test_register_login_history_and_trends_flow(tmp_path, monkeypatch):
    app = create_app(_settings(tmp_path))
    monkeypatch.setattr(app.state.registry, "get_models", lambda: (_DummyPipeline(), _DummyTokenizer()))
    client = TestClient(app)

    register = client.post("/auth/register", json={"username": "demo_user", "password": "secret12"})
    assert register.status_code == 200
    token = register.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["username"] == "demo_user"

    predict = client.post(
        "/predict",
        headers=headers,
        data={"text": "I need help figuring out what to do next."},
        files={"image": ("demo.png", _make_png_bytes(), "image/png")},
    )
    assert predict.status_code == 200

    history = client.get("/history", headers=headers)
    assert history.status_code == 200
    payload = history.json()
    assert len(payload) == 1
    assert payload[0]["emotion"] == "anxious"

    trends = client.get("/trends", headers=headers)
    assert trends.status_code == 200
    trend_payload = trends.json()
    assert trend_payload["total_checks"] == 1
    assert trend_payload["top_emotion"] == "anxious"
    assert trend_payload["top_intent"] == "seeking_help"

    login = client.post("/auth/login", json={"username": "demo_user", "password": "secret12"})
    assert login.status_code == 200
    assert login.json()["user"]["username"] == "demo_user"
