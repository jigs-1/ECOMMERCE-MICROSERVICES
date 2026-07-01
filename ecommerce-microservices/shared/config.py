from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "service"
    service_port: int = 8000
    database_url: str = "sqlite:///./service.db"
    jwt_secret: str = "super-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    redis_url: str = "redis://localhost:6379/0"
    user_service_url: str = "http://localhost:8001"
    product_service_url: str = "http://localhost:8002"
    order_service_url: str = "http://localhost:8003"
    notification_service_url: str = "http://localhost:8004"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
