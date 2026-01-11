"""
Authentication service for admin-backend.

MULTI-TENANCY: Supports organization-scoped authentication with org_id in JWT tokens.
Provides both legacy single-admin auth and new Firebase-based multi-user auth.
"""
import secrets
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional
import jwt
from fastapi.security import HTTPBearer
from ..config import settings, logger

security = HTTPBearer(auto_error=False)


def verify_credentials(username: str, password: str) -> Tuple[bool, str]:
    """
    LEGACY: Verify admin credentials against settings.
    
    Use Firebase auth for multi-tenant authentication instead.
    This is kept for backwards compatibility with single-admin deployments.
    """
    username_match = secrets.compare_digest(username, settings.ADMIN_USERNAME)
    password_match = settings.verify_password(password)
    if not username_match or not password_match:
        return False, "invalid_credentials"
    return True, ""


def create_jwt_token(
    username: str, 
    role: str = "admin", 
    org_id: str = None, 
    email: str = None,
    uid: str = None
) -> str:
    """
    Create JWT token with organization scope and user identification.
    
    MULTI-TENANCY: Includes org_id and uid claims for data isolation and user tracking.
    
    Args:
        username: User's username (display name)
        role: User's role (superuser, admin, viewer, analyser)
        org_id: Organization ID for data scoping
        email: User's email address (optional)
        uid: User's unique ID (optional, defaults to username)
    
    Returns:
        JWT token string
        
    Security Claims:
        - sub: Subject (username for backwards compatibility)
        - uid: User's unique identifier within the organization
        - org_id: Organization ID for multi-tenant data isolation
        - role: User's access level (superuser has full access)
        - email: User's email for identification
    """
    if not username or not isinstance(username, str):
        raise ValueError("Invalid username")
    if len(username) > 100:
        raise ValueError("Username too long")
    
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    
    payload = {
        "sub": username,
        "uid": uid or username,  # Explicit user ID for multi-user tracking
        "role": role,
        "iat": now,
        "exp": exp,
        "jti": secrets.token_hex(16)
    }
    
    # Add org_id claim for multi-tenancy
    if org_id:
        payload["org_id"] = org_id
    else:
        # Extract org_id from username if it follows {org_id}_{username} format
        if '_' in username:
            payload["org_id"] = username.split('_')[0]
        else:
            # Legacy single-tenant mode: use 'default' org_id
            payload["org_id"] = "default"
            logger.warning(f"No org_id provided for {username}, using 'default'")
    
    # Add email if provided
    if email:
        payload["email"] = email
    
    # Add username claim for display purposes
    payload["username"] = username
    
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Tuple[Optional[dict], bool, int]:
    """
    Decode and validate JWT token.
    
    Returns:
        Tuple of (payload, needs_refresh, remaining_seconds)
        - payload: Token payload dict or None if invalid
        - needs_refresh: True if token should be refreshed
        - remaining_seconds: Seconds until token expires
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        exp = payload.get("exp", 0)
        now_ts = datetime.now(timezone.utc).timestamp()
        remaining = max(0, int(exp - now_ts))
        needs_refresh = remaining < (settings.JWT_REFRESH_THRESHOLD_MINUTES * 60)
        return payload, needs_refresh, remaining
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None, False, 0
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token: {e}")
        return None, False, 0


def extract_org_id_from_uid(uid: str) -> str:
    """
    Extract organization ID from user ID.
    
    User IDs follow the format: {org_id}_{username}
    Example: "univ123_john" -> "univ123"
    """
    if '_' in uid:
        return uid.split('_')[0]
    return uid  # Return uid itself if no underscore (legacy)
