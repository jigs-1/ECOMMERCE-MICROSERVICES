# Python E-Commerce Microservices Backend

This project includes:

- `api-gateway` for routing client requests
- `user-service` with JWT registration and login
- `product-service` for catalog management
- `order-service` for order creation and synchronous service-to-service REST calls
- `notification-service` for async notifications via Redis pub-sub
- shared JWT validation middleware, config, database helpers, and centralized logging

## Stack

- FastAPI
- SQLAlchemy
- SQLite per service
- Redis pub-sub
- Docker Compose
- Request-scoped logging with `X-Request-ID`
- Built-in unit tests using `unittest`

## Run locally with Docker

```bash
docker compose up --build
```

Gateway endpoints are available at:

- `POST /auth/register`
- `POST /auth/login`
- `GET /users/me`
- `GET /health/dependencies`
- `POST /products/`
- `GET /products/`
- `POST /orders/`
- `GET /orders/`
- `GET /notifications/`

All protected routes require:

```text
Authorization: Bearer <jwt>
```

## Service layout

```text
ecommerce-microservices/
  gateway/
  services/
    user_service/
    product_service/
    order_service/
    notification_service/
  shared/
  docker-compose.yml
```

## Example flow

1. Register and log in through the gateway.
2. Create a product with the returned token.
3. Create an order for that product.
4. The order service reserves product inventory over REST before persisting the order.
5. The order service publishes `order.created` to Redis.
6. The notification service consumes the event and stores a notification in its own SQLite database.

## Quality checks

```bash
python -m compileall .
python -m unittest discover -s tests
```

## What makes this interview-ready

- Shared JWT middleware and request logging keep cross-service concerns consistent.
- Each service owns its database and only collaborates over HTTP or Redis events.
- The order workflow now models an actual business step by reserving stock before order creation.
- Notification persistence demonstrates async event handling without coupling the order API to downstream work.
