import httpx
from fastapi import HTTPException

from shared.config import get_settings

settings = get_settings()


async def fetch_product(product_id: int, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{settings.product_service_url}/{product_id}", headers=headers)
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Product not found")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Product service unavailable")
    return response.json()


async def reserve_inventory(product_id: int, quantity: int, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{settings.product_service_url}/{product_id}/reserve",
            headers=headers,
            json={"quantity": quantity},
        )
    if response.status_code == 409:
        raise HTTPException(status_code=409, detail="Insufficient inventory")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Product not found")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Inventory reservation failed")
    return response.json()
