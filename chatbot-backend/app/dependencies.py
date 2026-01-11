"""
FastAPI dependencies for dependency injection.

This module centralizes all FastAPI dependencies for:
- Application state injection
- Rate limiting
- Request validation
- Common utilities

Usage:
    from app.dependencies import get_app_state, validate_request_size
    
    @router.post("/endpoint")
    async def endpoint(state: AppState = Depends(get_app_state)):
        ...
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import get_logger, settings

if TYPE_CHECKING:
    from app.state import AppState

logger = get_logger("dependencies")


# =============================================================================
# Application State
# =============================================================================

async def get_app_state(request: Request) -> "AppState":
    """
    FastAPI dependency to get application state.
    
    This provides access to initialized providers (LLM, embedding, database)
    and shared configuration.
    
    Args:
        request: FastAPI request object
        
    Returns:
        AppState instance
        
    Raises:
        RuntimeError: If application state is not initialized
    """
    if not hasattr(request.app.state, "app_state"):
        logger.error("Application state not initialized")
        raise RuntimeError("Application state not initialized")
    return request.app.state.app_state


# Type alias for dependency injection
AppStateDep = Annotated["AppState", Depends(get_app_state)]


# =============================================================================
# Request Validation
# =============================================================================

async def validate_request_size(request: Request) -> None:
    """
    Validate that request body size is within limits.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If content length exceeds MAX_REQUEST_SIZE
    """
    content_length = request.headers.get("content-length")
    
    if content_length:
        try:
            size = int(content_length)
            if size > settings.MAX_REQUEST_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request body too large. Maximum size: {settings.MAX_REQUEST_SIZE} bytes",
                )
        except ValueError:
            # Invalid content-length header, let FastAPI handle it
            pass


async def validate_content_type(
    content_type: str = Header(default="application/json"),
) -> None:
    """
    Validate that content type is JSON.
    
    Args:
        content_type: Content-Type header value
        
    Raises:
        HTTPException: If content type is not JSON
    """
    if not content_type.startswith("application/json"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be application/json",
        )


# =============================================================================
# Client Information
# =============================================================================

def get_client_ip(request: Request) -> str:
    """
    Get the real client IP address.
    
    Checks X-Forwarded-For and X-Real-IP headers before falling
    back to the direct client IP.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address string
    """
    # Check X-Forwarded-For (can contain multiple IPs)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check X-Real-IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct client IP
    if request.client:
        return request.client.host
    
    return "unknown"


async def get_request_id(
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> str | None:
    """
    Get request ID from header for tracing.
    
    Args:
        x_request_id: X-Request-ID header value
        
    Returns:
        Request ID or None
    """
    return x_request_id


# =============================================================================
# Type Aliases for Common Dependencies
# =============================================================================

# Request size validation dependency
RequestSizeValidator = Annotated[None, Depends(validate_request_size)]

# Content type validation dependency
ContentTypeValidator = Annotated[None, Depends(validate_content_type)]

# Client IP dependency
ClientIP = Annotated[str, Depends(get_client_ip)]

# Request ID dependency
RequestID = Annotated[str | None, Depends(get_request_id)]
