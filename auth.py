"""
Technology:
- passlib (with bcrypt) -> hashes passwords before storing them. We NEVER
  store the actual password, only a one-way hash of it. Even if the
  database were leaked, raw passwords wouldn't be exposed.
- python-jose -> creates and verifies JWTs (JSON Web Tokens). When a user
  logs in successfully, we hand them a signed token. They include that
  token on every future request instead of their password, and we verify
  its signature to know who they are.

SECRET_KEY signs the tokens. In this starter project it has a default so
things work out of the box, but before you deploy this for real, set your
own SECRET_KEY environment variable to something long and random -- anyone
who has this key can forge valid login tokens.
"""

import os
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import User

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # tokens stay valid for 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Tells FastAPI's docs (/docs) where clients should send login requests to
# get a token. Doesn't restrict where the token itself is checked.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    FastAPI dependency: add this to any endpoint that should require login.
    It reads the "Authorization: Bearer <token>" header, verifies the JWT,
    and loads the matching user from the database -- or raises 401 if
    anything is wrong (missing, expired, or tampered-with token).
    """
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_error
    except JWTError:
        raise credentials_error

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_error
    return user
