from fastapi import Depends, FastAPI, HTTPException, Request, status
from sqlalchemy.orm import Session

from services.order_service.app.clients import fetch_product, reserve_inventory
from services.order_service.app.db import get_db, init_db
from services.order_service.app.models import Order
from services.order_service.app.schemas import OrderCreate, OrderResponse
from shared.auth import JWTValidationMiddleware
from shared.config import get_settings
from shared.events import publish_event
from shared.logging import configure_logging
from shared.middleware import RequestContextMiddleware

settings = get_settings()
logger = configure_logging(settings.app_name)

app = FastAPI(title="Order Service", version="1.0.0")
app.add_middleware(RequestContextMiddleware, service_name=settings.app_name)
app.add_middleware(
    JWTValidationMiddleware,
    excluded_paths={"/health"},
    excluded_prefixes=("/docs", "/openapi.json", "/redoc"),
)


@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("Order service started")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.app_name}


@app.get("/meta")
async def service_overview():
    return {"service": settings.app_name, "capabilities": ["create-order", "list-user-orders"]}


@app.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(payload: OrderCreate, request: Request, db: Session = Depends(get_db)):
    user_id = int(request.state.user["sub"])
    token = request.headers["Authorization"].split(" ", 1)[1]
    user_email = request.state.user.get("email", "unknown@example.com")

    product = await fetch_product(payload.product_id, token)
    await reserve_inventory(payload.product_id, payload.quantity, token)

    order = Order(
        user_id=user_id,
        product_id=payload.product_id,
        quantity=payload.quantity,
        total_price=product["price"] * payload.quantity,
    )
    db.add(order)
    db.commit()
    db.refresh(order)

    try:
        await publish_event(
            "order.created",
            {
                "order_id": order.id,
                "user_id": user_id,
                "user_email": user_email,
                "product_id": product["id"],
                "product_name": product["name"],
                "quantity": order.quantity,
                "total_price": order.total_price,
                "status": order.status,
            },
        )
    except Exception:
        order.status = "event_publish_failed"
        db.commit()
        db.refresh(order)
        logger.exception("Order %s created but event publishing failed", order.id)
        return order

    logger.info("Order created: %s", order.id)
    return order


@app.get("/", response_model=list[OrderResponse])
def list_orders(request: Request, db: Session = Depends(get_db)):
    user_id = int(request.state.user["sub"])
    return db.query(Order).filter(Order.user_id == user_id).all()
