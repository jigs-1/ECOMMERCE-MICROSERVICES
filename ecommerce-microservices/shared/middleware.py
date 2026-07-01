import time
import uuid
from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from shared.logging import configure_logging, request_id_context


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service_name: str):
        super().__init__(app)
        self.logger = configure_logging(service_name)

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_context.set(request_id)
        started_at = time.perf_counter()

        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started_at) * 1000

        response.headers["X-Request-ID"] = request_id
        self.logger.info(
            "%s %s -> %s in %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
