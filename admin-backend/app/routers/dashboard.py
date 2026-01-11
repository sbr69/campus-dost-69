"""
Dashboard router for metrics and activity.

MULTI-TENANCY: All operations are scoped to the user's organization.
"""
import time
from fastapi import APIRouter, Request, Depends, BackgroundTasks
from ..config import settings, logger
from ..dependencies import require_read_access, UserContext
from ..utils.limiter import limiter
from ..providers.database.metrics import metrics_provider
from ..providers.database.activity import activity_provider

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

@router.get("/stats")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_stats(
    request: Request, 
    background_tasks: BackgroundTasks, 
    user: UserContext = Depends(require_read_access)
):
    """
    Get dashboard statistics.
    
    MULTI-TENANCY: Returns statistics for user's organization only.
    """
    start_time = time.perf_counter()
    # MULTI-TENANCY: Pass org_id to get metrics
    metrics = await metrics_provider.get_metrics(user.org_id)
    
    # Format timestamps
    if metrics.get('last_updated'):
        metrics['last_updated'] = metrics['last_updated'].isoformat() if hasattr(metrics['last_updated'], 'isoformat') else str(metrics['last_updated'])
        
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Dashboard stats retrieved for org={user.org_id} | {elapsed:.1f}ms")
    
    return {"status": "success", "stats": metrics}

@router.get("/activity")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_activity(
    request: Request, 
    limit: int = settings.LOG_DEFAULT_LIMIT, 
    user: UserContext = Depends(require_read_access)
):
    """
    Get activity log.
    
    MULTI-TENANCY: Returns activity for user's organization only.
    """
    start_time = time.perf_counter()
    limit = min(max(1, limit), settings.LOG_MAX_LIMIT)
    # MULTI-TENANCY: Pass org_id to get activity log
    logs = await activity_provider.get_activity_log(user.org_id, limit)
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Activity log retrieved for org={user.org_id}: {len(logs)} items | {elapsed:.1f}ms")
    return {"status": "success", "activity": logs}

@router.get("/weekly")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_weekly_activity(
    request: Request, 
    user: UserContext = Depends(require_read_access)
):
    """
    Get weekly metrics.
    
    MULTI-TENANCY: Returns weekly data for user's organization only.
    """
    start_time = time.perf_counter()
    # MULTI-TENANCY: Pass org_id to get weekly metrics
    data = await metrics_provider.get_weekly_metrics(user.org_id)
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Weekly activity retrieved for org={user.org_id} | {elapsed:.1f}ms")
    return {"status": "success", "weekly": data}
