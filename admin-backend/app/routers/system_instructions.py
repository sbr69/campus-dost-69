"""
System instructions router for chatbot configuration.

MULTI-TENANCY: System instructions are now org-specific.
Each organization has their own system instructions stored in Firestore with LRU cache.
"""
import time
from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks
from ..config import settings, logger
from ..dependencies import require_read_access, require_write_access, UserContext
from ..utils.limiter import limiter
from ..providers.configuration import config_provider
from ..providers.database.metrics import metrics_provider
from ..providers.database.activity import activity_provider

router = APIRouter(prefix="/api/v1/system-instructions", tags=["system-instructions"])

@router.get("")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_sys_ins(
    request: Request, 
    user: UserContext = Depends(require_read_access)
):
    """
    Get system instructions.
    
    MULTI-TENANCY: Returns org-specific instructions from Firestore with LRU cache.
    Cache is mapped by org_id for warm start efficiency and no cross-org conflicts.
    """
    start_time = time.perf_counter()
    # MULTI-TENANCY: Pass org_id to get org-specific instructions
    res = await config_provider.get_instructions(org_id=user.org_id)
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"System instructions retrieved for org={user.org_id} | {elapsed:.1f}ms")
    return {"status": "success", "content": res["content"], "commit": res.get("version"), "org_id": user.org_id}

@router.post("/save")
@limiter.limit(settings.RATE_LIMIT_SYS_INS)
async def save_sys_ins(
    request: Request, 
    background_tasks: BackgroundTasks, 
    user: UserContext = Depends(require_write_access)
):
    """
    Save system instructions.
    
    MULTI-TENANCY: Saves to Firestore under user's organization.
    Backups and activity logs are scoped to org_id.
    Requires write access (admin or superuser role).
    """
    start_time = time.perf_counter()
    req = await request.json()
    content = req.get("content", "")
    message = req.get("message")
    if not content or len(content) > settings.SYS_INS_MAX_CONTENT: raise HTTPException(400, "Invalid content length")
    if message and len(message) > settings.SYS_INS_MAX_MESSAGE: raise HTTPException(400, "Message too long")
    
    # MULTI-TENANCY: Get current instructions for backup
    curr = await config_provider.get_instructions(org_id=user.org_id)
    
    # MULTI-TENANCY: Backup current content before overwriting
    if curr.get("content"): 
        await metrics_provider.backup_system_instructions(user.org_id, curr["content"], user.username)
    
    # MULTI-TENANCY: Save new instructions for this org
    res = await config_provider.save_instructions(
        content, 
        message or f"Update by {user.username}",
        org_id=user.org_id,
        user_id=user.uid
    )
    
    # MULTI-TENANCY: Log activity for this org
    background_tasks.add_task(activity_provider.log_activity, user.org_id, "sys_instructions_updated", user.uid, "sys_ins", "main", {})
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"System instructions saved for org={user.org_id} | {elapsed:.1f}ms")
    return {"status": "success", "message": "System instructions saved", "commit": res.get("version"), "org_id": user.org_id}

@router.get("/history")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_sys_history(
    request: Request, 
    limit: int = 3, 
    user: UserContext = Depends(require_read_access)
):
    """
    Get system instructions history.
    
    MULTI-TENANCY: Returns history for user's organization only.
    """
    start_time = time.perf_counter()
    limit = min(max(1, limit), 50)
    # MULTI-TENANCY: Pass org_id to get history
    hist = await metrics_provider.get_system_instructions_history(user.org_id, limit)
    for h in hist:
        if h.get('backed_up_at'):
            h['backed_up_at'] = h['backed_up_at'].isoformat() if hasattr(h['backed_up_at'], 'isoformat') else str(h['backed_up_at'])
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"System instructions history retrieved for org={user.org_id}: {len(hist)} items | {elapsed:.1f}ms")
    return {"status": "success", "history": hist, "total": len(hist)}

@router.get("/cache-stats")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_cache_stats(
    request: Request, 
    user: UserContext = Depends(require_read_access)
):
    """
    Get system instructions cache statistics (for debugging/monitoring).
    
    Returns cache hit/miss stats, entries count, and TTL info.
    """
    stats = config_provider.get_cache_stats()
    return {"status": "success", "cache_stats": stats}

