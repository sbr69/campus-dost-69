"""
Authentication and authorization dependencies for FastAPI.

MULTI-TENANCY: Extracts org_id from JWT tokens and provides role-based access control.
"""
from typing import Optional
from fastapi import Header, Response, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from .services.auth import security, decode_jwt_token, create_jwt_token
from .config import logger


class UserContext(BaseModel):
    """
    User context model for multi-tenancy.
    
    Contains user identification, organization scope, and permissions.
    """
    uid: str
    username: str
    email: Optional[str] = None
    org_id: str
    role: str
    
    @property
    def is_superuser(self) -> bool:
        return self.role == "superuser"
    
    @property
    def can_write(self) -> bool:
        """Check if user has write permissions (superuser or admin)."""
        return self.role in ("superuser", "admin")
    
    @property
    def can_read(self) -> bool:
        """Check if user has read permissions (all roles except analyser)."""
        return self.role in ("superuser", "admin", "viewer")
    
    @property
    def can_manage_users(self) -> bool:
        """Check if user can manage other users (superuser only)."""
        return self.role == "superuser"


def extract_org_id_from_uid(uid: str) -> str:
    """
    Extract organization ID from user ID.
    
    User IDs follow the format: {org_id}_{username}
    Example: "univ123_john" -> "univ123"
    
    If no underscore found, uses uid as org_id (legacy support).
    """
    if '_' in uid:
        return uid.split('_')[0]
    logger.warning(f"Legacy user ID format without org_id: {uid}")
    return uid  # Fallback for legacy users


async def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        authorization: Optional[str] = Header(None),
        response: Response = None
) -> UserContext:
    """
    Extract and validate user from JWT token.
    
    MULTI-TENANCY: Returns UserContext with org_id and uid extracted from token.
    Supports both new format (with org_id/uid claims) and legacy format (extract from sub).
    
    Security Claims Expected:
        - uid: User's unique identifier (preferred)
        - sub: Subject (fallback for uid in legacy tokens)
        - org_id: Organization ID for data isolation
        - role: User's access level
        - email: User's email address (optional)
        - username: Display name (optional, falls back to uid)
    """
    token = credentials.credentials if credentials else (authorization[7:] if authorization and authorization.startswith("Bearer ") else None)
    if not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, 
            "Not authenticated", 
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    payload, refresh, _ = decode_jwt_token(token)
    if not payload:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, 
            "Invalid or expired token", 
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Extract uid: prefer dedicated claim, fallback to sub for legacy tokens
    uid = payload.get("uid") or payload.get("sub")
    role = payload.get("role", "viewer")  # Default to viewer (read-only) for safety
    
    # Extract org_id: prefer explicit claim, fallback to extracting from uid
    org_id = payload.get("org_id")
    if not org_id:
        org_id = extract_org_id_from_uid(uid)
    
    user = UserContext(
        uid=uid,
        username=payload.get("username", uid),
        email=payload.get("email"),
        org_id=org_id,
        role=role
    )
    
    # Auto-refresh token if needed (include uid for new token format)
    if refresh and response:
        new_token = create_jwt_token(
            username=user.username, 
            role=user.role, 
            org_id=user.org_id,
            email=user.email,
            uid=user.uid
        )
        response.headers["X-New-Token"] = new_token
        response.headers["Access-Control-Expose-Headers"] = "X-New-Token"
    
    return user


# Role-based access control dependencies

def require_read_access(user: UserContext = Depends(get_current_user)) -> UserContext:
    """Require at least read permissions (viewer, admin, superuser)."""
    if not user.can_read:
        logger.warning(f"Read access denied for {user.uid} with role {user.role}")
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, 
            "Read access required. Your role does not have permission to view this resource."
        )
    return user


def require_write_access(user: UserContext = Depends(get_current_user)) -> UserContext:
    """Require write permissions (admin, superuser)."""
    if not user.can_write:
        logger.warning(f"Write access denied for {user.uid} with role {user.role}")
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, 
            "Write access required. Your role does not have permission to modify resources."
        )
    return user


def require_admin(user: UserContext = Depends(get_current_user)) -> UserContext:
    """Require admin or superuser role."""
    if user.role not in ("admin", "superuser"):
        logger.warning(f"Admin access denied for {user.uid} with role {user.role}")
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, 
            "Admin access required"
        )
    return user


def require_superuser(user: UserContext = Depends(get_current_user)) -> UserContext:
    """Require superuser role (for org management)."""
    if not user.is_superuser:
        logger.warning(f"Superuser access denied for {user.uid} with role {user.role}")
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, 
            "Superuser access required. This action requires organization administrator privileges."
        )
    return user
