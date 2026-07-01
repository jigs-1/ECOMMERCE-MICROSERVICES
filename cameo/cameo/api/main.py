from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import logging
from logging.config import dictConfig
from threading import Lock
from time import monotonic
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer

from cameo.api.config import Settings, load_settings
from cameo.api.store import DataStore, SessionUser
from cameo.core.inference.pipeline import CameoPipeline
from cameo.core.preprocess.image import preprocess_image
from cameo.core.preprocess.text import clean_text


def configure_logging() -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "INFO",
            },
        }
    )


configure_logging()
logger = logging.getLogger(__name__)


class PredictResponse(BaseModel):
    emotion: str
    intensity: float
    intent: str
    confidence: float
    attn_weights: dict
    gates: dict
    response: dict


class AuthRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: str


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class HistoryItem(BaseModel):
    id: int
    text: str
    emotion: str
    intent: str
    intensity: float
    confidence: float
    response_text: str
    created_at: str


class TrendResponse(BaseModel):
    total_checks: int
    top_emotion: Optional[str]
    top_intent: Optional[str]
    average_intensity: float
    average_confidence: float
    recent_emotions: list[str]
    recent_intents: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    version: str
    device: str
    model_loaded: bool
    tokenizer_loaded: bool
    weights_present: bool
    weights_loaded: bool
    weights_path: str
    frontend_built: bool


class RateLimiter:
    def __init__(self, requests: int, window_seconds: int):
        self.requests = requests
        self.window_seconds = window_seconds
        self.lock = Lock()
        self.buckets: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = monotonic()
        with self.lock:
            bucket = [stamp for stamp in self.buckets.get(key, []) if now - stamp < self.window_seconds]
            if len(bucket) >= self.requests:
                self.buckets[key] = bucket
                return False
            bucket.append(now)
            self.buckets[key] = bucket
            return True


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[CameoPipeline] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.weights_loaded = False
        self.lock = Lock()

    def get_models(self):
        with self.lock:
            if self.settings.require_weights and not self.settings.weights_path.exists():
                raise RuntimeError(f"Required weights file not found: {self.settings.weights_path}")
            if self.pipeline is None:
                logger.info("Initializing CAMEO pipeline on %s", self.settings.device)
                self.pipeline = CameoPipeline(
                    text_model=self.settings.text_model,
                    device=self.settings.device,
                )
                self.pipeline.eval()
            if not self.weights_loaded:
                if self.settings.weights_path.exists():
                    state = torch.load(self.settings.weights_path, map_location=self.settings.device)
                    self.pipeline.load_state_dict(state)
                    self.pipeline.eval()
                    self.weights_loaded = True
                    logger.info("Loaded trained weights from %s", self.settings.weights_path)
                else:
                    logger.warning(
                        "No trained weights found at %s; predictions may use untrained layers",
                        self.settings.weights_path,
                    )
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.text_model,
                    use_fast=True,
                    fix_mistral_regex=True,
                    local_files_only=self.settings.local_files_only,
                )
        return self.pipeline, self.tokenizer

    def readiness(self) -> ReadinessResponse:
        weights_present = self.settings.weights_path.exists()
        ready = self.pipeline is not None and self.tokenizer is not None and (
            self.weights_loaded or not self.settings.require_weights
        )
        return ReadinessResponse(
            status="ready" if ready else "degraded",
            version=self.settings.app_version,
            device=self.settings.device,
            model_loaded=self.pipeline is not None,
            tokenizer_loaded=self.tokenizer is not None,
            weights_present=weights_present,
            weights_loaded=self.weights_loaded,
            weights_path=str(self.settings.weights_path),
            frontend_built=self.settings.frontend_dist_dir.is_dir(),
        )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    registry = ModelRegistry(settings)
    rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
    store = DataStore(settings.database_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warm_start:
            logger.info("Warm start enabled; loading model assets during startup")
            registry.get_models()
        yield

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.state.rate_limiter = rate_limiter
    app.state.store = store

    def to_user_response(user: SessionUser) -> UserResponse:
        return UserResponse(id=user.user_id, username=user.username, created_at=user.created_at)

    def get_current_user(authorization: str | None) -> SessionUser:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required.")
        token = authorization.split(" ", 1)[1].strip()
        user = store.get_user_by_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired session.")
        return user

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        started = monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{monotonic() - started:.4f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    @app.middleware("http")
    async def protect_predict_endpoint(request: Request, call_next):
        if request.url.path == "/predict":
            if settings.api_key and request.headers.get("X-API-Key") != settings.api_key:
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key."})
            client_host = request.client.host if request.client else "unknown"
            if not rate_limiter.allow(client_host):
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Rate limit exceeded: max {settings.rate_limit_requests} requests per "
                            f"{settings.rate_limit_window_seconds} seconds."
                        )
                    },
                )
        return await call_next(request)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server error for request %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(
        text: str = Form(...),
        image: UploadFile = File(...),
        authorization: str | None = Header(default=None),
    ):
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Please provide a non-empty caption.")
        if len(text) > settings.max_text_chars:
            raise HTTPException(
                status_code=400,
                detail=f"Caption is too long (max {settings.max_text_chars} characters).",
            )
        if image.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
            raise HTTPException(status_code=400, detail="Image must be PNG or JPEG.")

        try:
            pipeline, tokenizer = registry.get_models()
        except RuntimeError as exc:
            logger.exception("Model assets are not ready")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        cleaned = clean_text(text)
        tokenized = tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        img_bytes = await image.read()
        if len(img_bytes) > settings.max_image_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Image is too large (max {settings.max_image_bytes // (1024 * 1024)} MB).",
            )
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

        img_tensor = preprocess_image(img).unsqueeze(0)
        pred = pipeline(tokenized, img_tensor, raw_text=cleaned)
        if authorization and authorization.startswith("Bearer "):
            user = store.get_user_by_token(authorization.split(" ", 1)[1].strip())
            if user:
                store.record_prediction(
                    user_id=user.user_id,
                    text=cleaned,
                    emotion=pred.emotion,
                    intent=pred.intent,
                    intensity=pred.intensity,
                    confidence=pred.confidence,
                    response_text=pred.response.text,
                )
        return PredictResponse(
            emotion=pred.emotion,
            intensity=pred.intensity,
            intent=pred.intent,
            confidence=pred.confidence,
            attn_weights=pred.attn_weights,
            gates=pred.gates,
            response={"text": pred.response.text, "mode": pred.response.mode},
        )

    @app.post("/auth/register", response_model=AuthResponse)
    async def register(payload: AuthRequest):
        try:
            user = store.create_user(payload.username, payload.password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        token = store.create_session(user)
        return AuthResponse(token=token, user=to_user_response(user))

    @app.post("/auth/login", response_model=AuthResponse)
    async def login(payload: AuthRequest):
        user = store.authenticate_user(payload.username, payload.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        token = store.create_session(user)
        return AuthResponse(token=token, user=to_user_response(user))

    @app.get("/auth/me", response_model=UserResponse)
    async def me(authorization: str | None = Header(default=None)):
        user = get_current_user(authorization)
        return to_user_response(user)

    @app.get("/history", response_model=list[HistoryItem])
    async def history(authorization: str | None = Header(default=None)):
        user = get_current_user(authorization)
        return [HistoryItem(**item) for item in store.get_history(user.user_id)]

    @app.get("/trends", response_model=TrendResponse)
    async def trends(authorization: str | None = Header(default=None)):
        user = get_current_user(authorization)
        return TrendResponse(**store.get_trends(user.user_id))

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", version=settings.app_version)

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready():
        payload = registry.readiness()
        if payload.status != "ready":
            raise HTTPException(status_code=503, detail=payload.model_dump())
        return payload

    try:
        frontend_dir = settings.frontend_dist_dir
        if frontend_dir.is_dir():
            app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
    except Exception:
        logger.exception("Failed to mount frontend static files from %s", settings.frontend_dist_dir)

    return app


app = create_app()
