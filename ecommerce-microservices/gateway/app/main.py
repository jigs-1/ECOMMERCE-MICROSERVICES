from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from shared.auth import JWTValidationMiddleware
from shared.config import get_settings
from shared.logging import configure_logging
from shared.middleware import RequestContextMiddleware

settings = get_settings()
logger = configure_logging(settings.app_name)

app = FastAPI(
    title="API Gateway",
    description="Single entrypoint for the e-commerce microservices platform.",
    version="1.0.0",
)
app.add_middleware(RequestContextMiddleware, service_name=settings.app_name)
app.add_middleware(
    JWTValidationMiddleware,
    excluded_paths={"/", "/health", "/auth/register", "/auth/login"},
    excluded_prefixes=("/docs", "/openapi.json", "/redoc"),
)

SERVICE_MAP = {
    "auth": settings.user_service_url,
    "users": settings.user_service_url,
    "products": settings.product_service_url,
    "orders": settings.order_service_url,
    "notifications": settings.notification_service_url,
}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.app_name}


@app.get("/")
async def gateway_overview():
    return {
        "name": "fanatics-ecommerce-platform",
        "gateway": settings.app_name,
        "services": list(SERVICE_MAP.keys()),
        "docs": "/docs",
    }


@app.get("/health/dependencies")
async def dependency_health():
    checks: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, service_url in {
            "user-service": settings.user_service_url,
            "product-service": settings.product_service_url,
            "order-service": settings.order_service_url,
            "notification-service": settings.notification_service_url,
        }.items():
            try:
                response = await client.get(f"{service_url}/health")
                checks[service_name] = {"status": response.status_code, "body": response.json()}
            except Exception as exc:
                checks[service_name] = {"status": "error", "detail": str(exc)}
    return checks


async def forward_request(service_url: str, request: Request, path: str) -> Response:
    url = urljoin(f"{service_url}/", path)
    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    logger.info("Forwarding %s %s to %s", request.method, request.url.path, url)

    async with httpx.AsyncClient(timeout=15.0) as client:
        upstream_response = await client.request(
            method=request.method,
            url=url,
            params=request.query_params,
            content=body,
            headers=headers,
        )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers={
            key: value
            for key, value in upstream_response.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        },
        media_type=upstream_response.headers.get("content-type"),
    )


@app.api_route("/{service}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def gateway_proxy_root(service: str, request: Request):
    service_url = SERVICE_MAP.get(service)
    if not service_url:
        raise HTTPException(status_code=404, detail=f"Unknown service '{service}'")
    return await forward_request(service_url, request, "")


@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def gateway_proxy(service: str, path: str, request: Request):
    service_url = SERVICE_MAP.get(service)
    if not service_url:
        raise HTTPException(status_code=404, detail=f"Unknown service '{service}'")
    return await forward_request(service_url, request, path)
