from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session

from services.product_service.app.db import get_db, init_db
from services.product_service.app.models import Product
from services.product_service.app.schemas import (
    InventoryReservation,
    InventoryReservationResponse,
    ProductCreate,
    ProductResponse,
)
from shared.auth import JWTValidationMiddleware
from shared.config import get_settings
from shared.logging import configure_logging
from shared.middleware import RequestContextMiddleware

settings = get_settings()
logger = configure_logging(settings.app_name)

app = FastAPI(title="Product Service", version="1.0.0")
app.add_middleware(RequestContextMiddleware, service_name=settings.app_name)
app.add_middleware(
    JWTValidationMiddleware,
    excluded_paths={"/health"},
    excluded_prefixes=("/docs", "/openapi.json", "/redoc"),
)


@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("Product service started")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.app_name}


@app.get("/", response_model=list[ProductResponse])
def list_products(db: Session = Depends(get_db)):
    return db.query(Product).all()


@app.post("/", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    product = Product(**payload.model_dump())
    db.add(product)
    db.commit()
    db.refresh(product)
    logger.info("Product created: %s", product.name)
    return product


@app.get("/meta")
async def service_overview():
    return {"service": settings.app_name, "capabilities": ["create-product", "list-products", "reserve-inventory"]}


@app.get("/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.post("/{product_id}/reserve", response_model=InventoryReservationResponse)
def reserve_inventory(product_id: int, payload: InventoryReservation, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    if product.inventory < payload.quantity:
        raise HTTPException(status_code=409, detail="Insufficient inventory")

    product.inventory -= payload.quantity
    db.commit()
    logger.info("Reserved %s units for product %s", payload.quantity, product_id)
    return InventoryReservationResponse(product_id=product.id, remaining_inventory=product.inventory)
