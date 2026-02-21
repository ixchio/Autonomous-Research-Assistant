"""
Auth — JWT tokens + bcrypt passwords + mongo persistence.

Nothing fancy. Sign up, get a token, slap it on every request.
Passwords are bcrypt'd. Tokens expire in 24h. Rate limited to
5 login attempts per minute per IP (well, per email for now since
we're behind a reverse proxy and IP extraction is a whole thing).

If mongo isn't available, falls back to an in-memory dict so you
can still hack on the frontend locally without spinning up docker.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict
from time import time

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

import jwt
import bcrypt

from core.db import get_db

SECRET_KEY = os.getenv("JWT_SECRET") or secrets.token_hex(32)
ALGORITHM = "HS256"
TOKEN_HOURS = 24

security = HTTPBearer()

# in-memory fallback when mongo isn't around
_mem_users: dict = {}

# ghetto rate limiter for login — {email: [timestamps]}
_login_attempts: dict = defaultdict(list)
MAX_LOGIN_PER_MIN = 5


# ---- models ----

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
    expires_in: int = TOKEN_HOURS * 3600


class UserInfo(BaseModel):
    email: str
    name: str
    created_at: str


# ---- password stuff ----

def _hash_pw(password: str) -> str:
    # bcrypt handles salt internally, that's the whole point
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_pw(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        # probably an old sha256 hash from before we switched
        return False


# ---- jwt ----

def _mint_token(email: str) -> str:
    return jwt.encode(
        {
            "sub": email,
            "exp": datetime.utcnow() + timedelta(hours=TOKEN_HOURS),
            "iat": datetime.utcnow(),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def _crack_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired — log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Bad token")


# ---- rate limit check ----

def _check_rate_limit(email: str):
    now = time()
    # toss out anything older than 60s
    _login_attempts[email] = [t for t in _login_attempts[email] if now - t < 60]
    if len(_login_attempts[email]) >= MAX_LOGIN_PER_MIN:
        raise HTTPException(429, "Too many login attempts. Chill for a minute.")
    _login_attempts[email].append(now)


# ---- the actual auth logic ----

async def signup(req: SignupRequest) -> TokenResponse:
    db = get_db()

    if db is not None:
        # mongo path
        existing = await db.users.find_one({"email": req.email})
        if existing:
            raise HTTPException(409, "Email already taken")

        await db.users.insert_one({
            "email": req.email,
            "password_hash": _hash_pw(req.password),
            "name": req.name,
            "created_at": datetime.utcnow(),
        })
    else:
        # fallback — in-memory
        if req.email in _mem_users:
            raise HTTPException(409, "Email already taken")
        _mem_users[req.email] = {
            "password_hash": _hash_pw(req.password),
            "name": req.name,
            "created_at": datetime.utcnow().isoformat(),
        }

    return TokenResponse(access_token=_mint_token(req.email))


async def login(req: LoginRequest) -> TokenResponse:
    _check_rate_limit(req.email)
    db = get_db()

    if db is not None:
        user = await db.users.find_one({"email": req.email})
        if not user or not _check_pw(req.password, user["password_hash"]):
            raise HTTPException(401, "Wrong email or password")
    else:
        user = _mem_users.get(req.email)
        if not user or not _check_pw(req.password, user["password_hash"]):
            raise HTTPException(401, "Wrong email or password")

    return TokenResponse(access_token=_mint_token(req.email))


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    payload = _crack_token(creds.credentials)
    email = payload.get("sub")

    db = get_db()
    if db is not None:
        user = await db.users.find_one({"email": email})
        if not user:
            raise HTTPException(401, "User not found — maybe deleted?")
    else:
        if email not in _mem_users:
            raise HTTPException(401, "User not found")

    return email


async def get_user_info(email: str) -> Optional[UserInfo]:
    db = get_db()

    if db is not None:
        user = await db.users.find_one({"email": email})
        if not user:
            return None
        return UserInfo(
            email=email,
            name=user.get("name", ""),
            created_at=user["created_at"].isoformat() if isinstance(user["created_at"], datetime) else str(user["created_at"]),
        )
    else:
        user = _mem_users.get(email)
        if not user:
            return None
        return UserInfo(email=email, name=user["name"], created_at=user["created_at"])
