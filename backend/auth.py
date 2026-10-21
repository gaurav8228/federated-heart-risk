r"""
auth.py  --  Milestone M3
Login security for the API.

  * Passwords are stored as a HASH (one-way), never in plain text.
    We use PBKDF2-SHA256 from Python's standard library (hashlib),
    so no extra package is needed. Format: pbkdf2_sha256$iterations$salt$hash
  * After login the server returns a JWT token (JSON Web Token).
    The token carries the user's email and role, is signed with
    SECRET_KEY (from .env) and expires after 8 hours.
  * Dependencies (get_current_user, require_admin) are used by FastAPI
    to protect endpoints.
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .database import get_db
from .models import User

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))            # read SECRET_KEY from .env

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is missing - check D:\\heartRisk\\.env")

ALGORITHM = "HS256"           # how the token is signed
TOKEN_HOURS = 8               # token expiry, as agreed

# Asks FastAPI to read the "Authorization: Bearer <token>" header.
# tokenUrl tells Swagger UI where to send the login request.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# --------------------------------------------------------------------------- #
# Password hashing (PBKDF2, standard library only)
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    """Turn a password into a safe-to-store string with a random salt."""
    salt = secrets.token_hex(16)                       # random 32-char salt
    iterations = 200_000                               # slow on purpose
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
    )
    return f"pbkdf2_sha256${iterations}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against the stored hash (True / False)."""
    try:
        _, iterations, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        )
        return secrets.compare_digest(digest.hex(), expected)   # constant-time compare
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# JWT tokens
# --------------------------------------------------------------------------- #
def create_access_token(user: User) -> str:
    """Build a signed token that says 'this is user X, role Y' for 8 hours."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.email,                 # subject = the user's email
        "role": user.role,                 # "doctor" or "admin"
        "name": user.full_name,
    }
    payload["iat"] = int(now.timestamp())
    payload["exp"] = int((now + timedelta(hours=TOKEN_HOURS)).timestamp())
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Read the token back. Raises an exception if it is invalid/expired."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# --------------------------------------------------------------------------- #
# FastAPI dependencies
# --------------------------------------------------------------------------- #
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Protect an endpoint: only requests with a valid token get through."""
    error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        raise error
    user = db.query(User).filter(User.email == payload.get("sub")).first()
    if user is None:
        raise error
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Protect an endpoint: only admins get through."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin rights required",
        )
    return user
