from fastapi import Depends, FastAPI, HTTPException, Request, status
from sqlalchemy.orm import Session

from services.user_service.app.db import get_db, init_db
from services.user_service.app.models import User
from services.user_service.app.schemas import TokenResponse, UserCreate, UserLogin, UserResponse
from shared.auth import (
    JWTValidationMiddleware,
    create_access_token,
    hash_password,
    verify_password,
)
from shared.config import get_settings
from shared.logging import configure_logging
from shared.middleware import RequestContextMiddleware

settings = get_settings()
logger = configure_logging(settings.app_name)

app = FastAPI(title="User Service", version="1.0.0")
app.add_middleware(RequestContextMiddleware, service_name=settings.app_name)
app.add_middleware(
    JWTValidationMiddleware,
    excluded_paths={"/", "/health", "/register", "/login"},
    excluded_prefixes=("/docs", "/openapi.json", "/redoc"),
)


@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("User service started")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.app_name}


@app.get("/")
async def service_overview():
    return {"service": settings.app_name, "capabilities": ["register", "login", "profile"]}


@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("User registered: %s", user.email)
    return user


@app.post("/login", response_model=TokenResponse)
def login_user(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        subject=str(user.id),
        additional_claims={"email": user.email, "full_name": user.full_name},
    )
    logger.info("User logged in: %s", user.email)
    return TokenResponse(access_token=token)


@app.get("/me", response_model=UserResponse)
def get_current_user(request: Request, db: Session = Depends(get_db)):
    user_id = int(request.state.user["sub"])
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user_by_id(user_id: int, request: Request, db: Session = Depends(get_db)):
    if int(request.state.user["sub"]) != user_id:
        raise HTTPException(status_code=403, detail="Not allowed to access this user")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
