import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

os.environ["APP_NAME"] = "order-service-test"
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.gettempdir()}/order-service-test.db"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["JWT_ALGORITHM"] = "HS256"

from fastapi.testclient import TestClient

from services.order_service.app.db import Base, engine
from services.order_service.app.main import app
from shared.auth import create_access_token


class OrderServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)
        cls.token = create_access_token("1", {"email": "buyer@example.com"})
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def test_create_order_reserves_inventory_and_publishes_event(self):
        with patch(
            "services.order_service.app.main.fetch_product",
            AsyncMock(return_value={"id": 7, "name": "Jersey", "price": 99.0}),
        ), patch(
            "services.order_service.app.main.reserve_inventory",
            AsyncMock(return_value={"product_id": 7, "remaining_inventory": 3}),
        ), patch(
            "services.order_service.app.main.publish_event",
            AsyncMock(),
        ) as publish_event:
            response = self.client.post(
                "/",
                json={"product_id": 7, "quantity": 2},
                headers=self.headers,
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["total_price"], 198.0)
        self.assertEqual(payload["user_id"], 1)
        publish_event.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
