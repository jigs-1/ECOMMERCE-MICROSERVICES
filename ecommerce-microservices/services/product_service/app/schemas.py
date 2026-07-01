from datetime import datetime

from pydantic import BaseModel, Field


class ProductCreate(BaseModel):
    name: str
    description: str
    price: float = Field(gt=0)
    inventory: int = Field(ge=0)


class ProductResponse(ProductCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class InventoryReservation(BaseModel):
    quantity: int = Field(gt=0)


class InventoryReservationResponse(BaseModel):
    product_id: int
    remaining_inventory: int
