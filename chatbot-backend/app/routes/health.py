"""
Health check endpoints.

This module provides health monitoring endpoints for:
- Liveness probes (ping)
- Readiness probes (ready)
- Deep health checks (health with ?deep=true)

These endpoints follow Kubernetes health check patterns for
container orchestration compatibility.

Usage:
    GET /           - Full health check
    GET /health     - Full health check (alias)
    GET /ping       - Simple liveness probe
    GET /ready      - Readiness probe
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import JSONResponse

from app.config import get_logger
from app.models import HealthResponse, PingResponse, ReadinessResponse
from app.state import AppState, get_app_state

logger = get_logger("routes.health")

router = APIRouter(tags=["Health"])


# =============================================================================
# Health Check Endpoint
# =============================================================================

@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns detailed health status including provider information.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is degraded or unhealthy"},
    },
)
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check (alias)",
    description="Alias for root health check endpoint.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is degraded or unhealthy"},
    },
)
async def health_check(
    state: AppState = Depends(get_app_state),
    deep: bool = Query(
        default=False,
        description="Perform deep health check including database ping",
    ),
) -> HealthResponse | JSONResponse:
    """
    Returns detailed health status including provider information.
    
    The health check reports status as:
    - **healthy**: All services operational
    - **degraded**: Some services unavailable but core functionality works
    - **unhealthy**: Critical services unavailable
    
    Args:
        state: Application state (injected)
        deep: If true, performs actual health check calls to external services
              (slower but more accurate)
    
    Returns:
        HealthResponse with service statuses
    """
    db_available = state.database_provider.is_available()
    db_healthy = db_available
    llm_available = state.llm_provider.is_available()
    embedding_available = state.embedding_provider.is_available()
    
    # Deep health check - actually ping external services
    db_latency_ms: float | None = None
    if deep and db_available:
        start_time = time.perf_counter()
        try:
            if hasattr(state.database_provider, "health_check"):
                db_healthy = await state.database_provider.health_check()
            db_latency_ms = (time.perf_counter() - start_time) * 1000
        except Exception as e:
            logger.warning("Deep health check failed for database: %s", e)
            db_healthy = False
            db_latency_ms = None
    
    # Build services status
    services: dict[str, Any] = {
        "ready": state.is_ready(),
        "llm": {
            "provider": state.llm_provider.get_provider_name(),
            "model": state.llm_provider.get_model_name(),
            "available": llm_available,
            "status": "healthy" if llm_available else "unavailable",
        },
        "embedding": {
            "provider": state.embedding_provider.get_provider_name(),
            "model": state.embedding_provider.get_model_name(),
            "available": embedding_available,
            "status": "healthy" if embedding_available else "unavailable",
        },
        "database": {
            "provider": state.database_provider.get_provider_name(),
            "available": db_available,
            "status": "healthy" if db_healthy else "unavailable",
        },
    }
    
    # Add deep check details if performed
    if deep:
        services["database"]["healthy"] = db_healthy
        if db_latency_ms is not None:
            services["database"]["latency_ms"] = round(db_latency_ms, 2)
    
    # Determine overall status
    if llm_available and db_healthy:
        overall_status = "healthy"
    elif llm_available:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    response = HealthResponse(
        status=overall_status,
        services=services,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="3.0.0",
    )
    
    # Return 503 if unhealthy
    if overall_status == "unhealthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )
    
    return response


# =============================================================================
# Liveness Probe
# =============================================================================

@router.get(
    "/ping",
    response_model=PingResponse,
    summary="Liveness probe",
    description="Simple ping endpoint for keepalive checks. Does not verify service health.",
    responses={
        200: {
            "description": "Service is alive",
            "content": {
                "application/json": {
                    "example": {"status": "ok"}
                }
            },
        },
    },
)
async def ping() -> PingResponse:
    """
    Simple ping endpoint for keepalive checks.
    
    This endpoint always returns 200 as long as the server is running.
    It does not check any external dependencies.
    
    Use this endpoint for:
    - Kubernetes liveness probes
    - Load balancer health checks
    - Simple uptime monitoring
    
    Returns:
        PingResponse with status "ok"
    """
    return PingResponse(status="ok")


# =============================================================================
# Readiness Probe
# =============================================================================

@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Kubernetes-style readiness probe.",
    responses={
        200: {
            "description": "Service is ready to accept requests",
            "content": {
                "application/json": {
                    "example": {
                        "ready": True,
                        "checks": {
                            "llm": True,
                            "database": True
                        }
                    }
                }
            },
        },
        503: {
            "description": "Service is not ready",
            "content": {
                "application/json": {
                    "example": {
                        "ready": False,
                        "checks": {
                            "llm": True,
                            "database": False
                        }
                    }
                }
            },
        },
    },
)
async def readiness_check(
    state: AppState = Depends(get_app_state),
) -> ReadinessResponse | JSONResponse:
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if the service is ready to accept traffic.
    Returns 503 if any critical service is unavailable.
    
    Use this endpoint for:
    - Kubernetes readiness probes
    - Deployment rollout verification
    - Service mesh health checks
    
    Args:
        state: Application state (injected)
    
    Returns:
        ReadinessResponse with ready status and individual check results
    """
    checks = {
        "llm": state.llm_provider.is_available(),
        "database": state.database_provider.is_available(),
        "embedding": state.embedding_provider.is_available(),
    }
    
    # Service is ready if LLM and database are available
    is_ready = checks["llm"] and checks["database"]
    
    response = ReadinessResponse(
        ready=is_ready,
        checks=checks,
    )
    
    if not is_ready:
        logger.warning(
            "Readiness check failed: llm=%s, database=%s",
            checks["llm"],
            checks["database"],
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )
    
    return response
