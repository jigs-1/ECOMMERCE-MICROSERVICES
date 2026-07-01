from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import jwt
from fastapi import HTTPException, Request, status
from jwt import InvalidTokenError
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware

from shared.config import get_settings

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, additional_claims: dict[str, Any] | None = None) -> str:
    settings = get_settings()
    expire_at = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiration_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "exp": expire_at,
    }
    if additional_claims:
        payload.update(additional_claims)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc


def extract_token_from_request(request: Request) -> str:
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    return authorization.split(" ", 1)[1]


def get_current_user_payload(request: Request) -> dict[str, Any]:
    return decode_access_token(extract_token_from_request(request))


class JWTValidationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, excluded_paths: set[str] | None = None, excluded_prefixes: tuple[str, ...] = ()):
        super().__init__(app)
        self.excluded_paths = excluded_paths or set()
        self.excluded_prefixes = excluded_prefixes

    def _is_excluded(self, path: str) -> bool:
        return path in self.excluded_paths or any(path.startswith(prefix) for prefix in self.excluded_prefixes)

    async def dispatch(self, request: Request, call_next: Callable):
        if self._is_excluded(request.url.path):
            return await call_next(request)

        token = extract_token_from_request(request)
        request.state.user = decode_access_token(token)
        return await call_next(request)
