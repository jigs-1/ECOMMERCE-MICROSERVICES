import asyncio

from fastapi import Depends, FastAPI, Request
from sqlalchemy.orm import Session

from services.notification_service.app.db import SessionLocal, get_db, init_db
from services.notification_service.app.models import Notification
from services.notification_service.app.schemas import NotificationResponse
from shared.auth import JWTValidationMiddleware
from shared.config import get_settings
from shared.events import subscribe_forever
from shared.logging import configure_logging
from shared.middleware import RequestContextMiddleware

settings = get_settings()
logger = configure_logging(settings.app_name)

app = FastAPI(title="Notification Service", version="1.0.0")
app.add_middleware(RequestContextMiddleware, service_name=settings.app_name)
app.add_middleware(
    JWTValidationMiddleware,
    excluded_paths={"/health"},
    excluded_prefixes=("/docs", "/openapi.json", "/redoc"),
)


async def handle_order_created(event: dict) -> None:
    db = SessionLocal()
    try:
        existing = db.query(Notification).filter(Notification.order_id == event["order_id"]).first()
        if existing:
            logger.info("Skipping duplicate notification for order %s", event["order_id"])
            return

        message = (
            f"Order {event['order_id']} confirmed for {event['quantity']} x "
            f"{event['product_name']} totalling ${event['total_price']:.2f}"
        )
        notification = Notification(
            user_id=event["user_id"],
            order_id=event["order_id"],
            recipient_email=event["user_email"],
            message=message,
        )
        db.add(notification)
        db.commit()
        logger.info("Notification stored for order %s", event["order_id"])
    finally:
        db.close()


@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Notification service started")
    app.state.subscriber_task = asyncio.create_task(
        subscribe_forever("order.created", handle_order_created)
    )


@app.on_event("shutdown")
async def shutdown_event():
    subscriber_task = getattr(app.state, "subscriber_task", None)
    if subscriber_task:
        subscriber_task.cancel()
        try:
            await subscriber_task
        except asyncio.CancelledError:
            logger.info("Notification subscriber stopped")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.app_name}


@app.get("/meta")
async def service_overview():
    return {"service": settings.app_name, "capabilities": ["list-notifications", "consume-order-events"]}


@app.get("/", response_model=list[NotificationResponse])
def list_notifications(request: Request, db: Session = Depends(get_db)):
    user_id = int(request.state.user["sub"])
    return db.query(Notification).filter(Notification.user_id == user_id).all()
