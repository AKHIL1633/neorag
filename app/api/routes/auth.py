from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.auth import authenticate_user, create_access_token, register_user
from app.core.pg import get_db_session
from app.models.schemas import Token, UserCreate, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="Register a new user")
async def register(payload: UserCreate, db: Session = Depends(get_db_session)):
    """
    Create a new user account. The very first user ever registered is
    auto-promoted to admin (bootstrap pattern) since no admin exists yet.
    """
    try:
        user = register_user(db, payload.username, payload.email, payload.password)
    except ValueError as e:
        message = str(e)
        code = status.HTTP_409_CONFLICT if "already registered" in message else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=message)
    return user


@router.post("/token", response_model=Token, summary="Get JWT access token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db_session)):
    """Get a JWT token for API authentication. Register an account first via `/auth/register`."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user.username})
    return Token(access_token=token)
