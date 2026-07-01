from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple

import torch


def _parse_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_env(name: str) -> Tuple[str, ...]:
    value = os.getenv(name, "")
    if not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    project_root: Path
    frontend_dist_dir: Path
    weights_path: Path
    database_path: Path
    text_model: str
    device: str
    warm_start: bool
    require_weights: bool
    local_files_only: bool
    cors_origins: Tuple[str, ...]
    api_key: str | None
    rate_limit_requests: int
    rate_limit_window_seconds: int
    max_text_chars: int
    max_image_bytes: int


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_env = os.getenv("CAMEO_WEIGHTS_PATH")
    weights_path = Path(weights_env).expanduser() if weights_env else project_root / "artifacts" / "cameo.pt"
    if not weights_path.is_absolute():
        weights_path = project_root / weights_path
    database_env = os.getenv("CAMEO_DATABASE_PATH")
    database_path = Path(database_env).expanduser() if database_env else project_root / "artifacts" / "cameo.db"
    if not database_path.is_absolute():
        database_path = project_root / database_path

    return Settings(
        app_name="CAMEO API",
        app_version=os.getenv("CAMEO_APP_VERSION", "1.0.0"),
        project_root=project_root,
        frontend_dist_dir=project_root / "ui" / "dist",
        weights_path=weights_path,
        database_path=database_path,
        text_model=os.getenv("CAMEO_TEXT_MODEL", "microsoft/deberta-v3-base"),
        device=os.getenv("CAMEO_DEVICE", default_device),
        warm_start=_parse_bool("CAMEO_WARM_START", False),
        require_weights=_parse_bool("CAMEO_REQUIRE_WEIGHTS", False),
        local_files_only=_parse_bool("CAMEO_LOCAL_FILES_ONLY", False),
        cors_origins=_parse_csv_env("CAMEO_CORS_ORIGINS"),
        api_key=os.getenv("CAMEO_API_KEY") or None,
        rate_limit_requests=int(os.getenv("CAMEO_RATE_LIMIT_REQUESTS", "30")),
        rate_limit_window_seconds=int(os.getenv("CAMEO_RATE_LIMIT_WINDOW_SECONDS", "60")),
        max_text_chars=int(os.getenv("CAMEO_MAX_TEXT_CHARS", "1200")),
        max_image_bytes=int(os.getenv("CAMEO_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))),
    )
