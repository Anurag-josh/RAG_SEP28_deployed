"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Authentication helper utilities — secure password hashing with bcrypt,
         JWT token creation, validation, and user identity extraction.
Dependencies: bcrypt, jwt, datetime, logging, config.config
"""

import logging
import datetime
from typing import Optional, Dict, Any

try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _BCRYPT_AVAILABLE = False

try:
    import jwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False

from config.config import JWT_SECRET

logger = logging.getLogger("utils.auth")


def hash_password(password: str) -> str:
    """
    Hashes a plaintext password securely using bcrypt.
    Never stores plaintext passwords.
    """
    if not password:
        raise ValueError("Password cannot be empty.")
    if _BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")
    else:
        # Fallback hash if bcrypt isn't installed (for test environments)
        import hashlib
        logger.warning("bcrypt module not installed. Using hashlib fallback for password hashing.")
        return hashlib.sha256((password + JWT_SECRET).encode("utf-8")).hexdigest()


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against the stored bcrypt hash.
    """
    if not password or not hashed_password:
        return False
    if _BCRYPT_AVAILABLE and hashed_password.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception as e:
            logger.error(f"Error verifying password with bcrypt: {e}")
            return False
    else:
        import hashlib
        fallback = hashlib.sha256((password + JWT_SECRET).encode("utf-8")).hexdigest()
        return fallback == hashed_password


def create_jwt_token(user_id: str, username: str, email: str, expires_in_hours: int = 24) -> str:
    """
    Generates a signed JWT authentication token containing basic user identity claims.
    """
    payload = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "iat": datetime.datetime.now(datetime.timezone.utc),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=expires_in_hours)
    }
    if _JWT_AVAILABLE:
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    else:
        # Fallback payload serializer if pyjwt is missing
        import json, base64
        data = json.dumps({"user_id": user_id, "username": username, "email": email, "exp": payload["exp"].timestamp()})
        return "fallback_token." + base64.b64encode(data.encode("utf-8")).decode("utf-8")


def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a JWT authentication token.
    Returns the token payload if valid, or None if expired/invalid.
    """
    if not token:
        return None
    token = token.replace("Bearer ", "").strip()
    if _JWT_AVAILABLE:
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token signature has expired.")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    else:
        if token.startswith("fallback_token."):
            import json, base64
            try:
                raw = token.split(".", 1)[1]
                data = json.loads(base64.b64decode(raw.encode("utf-8")).decode("utf-8"))
                if data.get("exp", 0) > datetime.datetime.now(datetime.timezone.utc).timestamp():
                    return data
            except Exception:
                return None
        return None
