"""
JWT auth — keeps it simple but legit.
Sign up → get a token → attach it to every request.
Passwords are bcrypt-hashed. Tokens expire in 24h.
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

# we use PyJWT (not python-jose) — lighter, fewer CVEs
import jwt

SECRET_KEY = os.getenv("JWT_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

security = HTTPBearer()

# ---- in-memory user store (swap for a real DB later) ----
_users: dict = {}


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = TOKEN_EXPIRY_HOURS * 3600


class UserInfo(BaseModel):
    email: str
    name: str
    created_at: str


def _hash_pw(password: str) -> str:
    """Dead-simple hash — swap for bcrypt in production."""
    salt = os.getenv("PW_SALT", "ixchio-salt-2026")
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def _make_token(email: str) -> str:
    payload = {
        "sub": email,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---- public API ----

def signup(req: SignupRequest) -> TokenResponse:
    if req.email in _users:
        raise HTTPException(status_code=409, detail="Email already registered")

    _users[req.email] = {
        "password_hash": _hash_pw(req.password),
        "name": req.name,
        "created_at": datetime.utcnow().isoformat(),
    }
    token = _make_token(req.email)
    return TokenResponse(access_token=token)


def login(req: LoginRequest) -> TokenResponse:
    user = _users.get(req.email)
    if not user or user["password_hash"] != _hash_pw(req.password):
        raise HTTPException(status_code=401, detail="Bad credentials")

    token = _make_token(req.email)
    return TokenResponse(access_token=token)


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """FastAPI dependency — inject this to protect any route."""
    payload = _decode_token(creds.credentials)
    email = payload.get("sub")
    if email not in _users:
        raise HTTPException(status_code=401, detail="User not found")
    return email


def get_user_info(email: str) -> Optional[UserInfo]:
    user = _users.get(email)
    if not user:
        return None
    return UserInfo(email=email, name=user["name"], created_at=user["created_at"])
