import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.pg import get_db_session
from app.models.user_model import User

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

_PASSWORD_MIN_LENGTH = 10
_DIGIT_RE = re.compile(r"\d")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def validate_password_strength(password: str) -> None:
    if len(password) < _PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {_PASSWORD_MIN_LENGTH} characters")
    if not _DIGIT_RE.search(password):
        raise ValueError("Password must contain at least one digit")


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def register_user(db: Session, username: str, email: str, password: str) -> User:
    """Creates a new user. Raises ValueError on invalid password or duplicate
    username/email — callers (routes) translate this to an HTTP error.

    Bootstrap pattern: the very first user ever registered is auto-promoted
    to admin, since there's no admin yet to grant that role.
    """
    validate_password_strength(password)

    existing = db.execute(
        select(User).where((User.username == username) | (User.email == email))
    ).scalar_one_or_none()
    if existing:
        raise ValueError("Username or email already registered")

    is_first_user = db.execute(select(User.id).limit(1)).scalar_one_or_none() is None

    user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        is_admin=is_first_user,
    )
    db.add(user)
    db.flush()
    db.refresh(user)
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db_session),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exception
    return user
