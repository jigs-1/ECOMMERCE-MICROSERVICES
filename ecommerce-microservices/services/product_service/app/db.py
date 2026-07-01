from collections.abc import Generator

from shared.config import get_settings
from shared.database import Base, build_engine, build_session_factory

settings = get_settings()
engine = build_engine(settings.database_url)
SessionLocal = build_session_factory(engine)


def init_db():
    from services.product_service.app import models

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
