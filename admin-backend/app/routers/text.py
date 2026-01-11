"""
Text processing router.

MULTI-TENANCY: All operations are scoped to the user's organization.
"""
import time
from fastapi import APIRouter, Request, Depends, BackgroundTasks
from ..config import settings, logger
from ..dependencies import require_read_access, require_write_access, UserContext
from ..utils.limiter import limiter
from ..utils.validators import validate_text_length, validate_filename
from ..services.processor import text_processor
from ..services.ingestion import pipeline

router = APIRouter(prefix="/api/v1/text", tags=["text"])

@router.post("/process")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def process_text(
    request: Request, 
    user: UserContext = Depends(require_read_access)
):
    """
    Generate a preview of processed text (chunks and embeddings).
    
    Read-only operation - no data is stored.
    """
    start_time = time.perf_counter()
    req = await request.json()
    text = req.get("text", "")
    validate_text_length(text)
    res = await text_processor.generate_preview(text)
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Text preview generated for org={user.org_id} | {elapsed:.1f}ms")
    return res

@router.post("/upload")
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_text(
    request: Request, 
    background_tasks: BackgroundTasks, 
    user: UserContext = Depends(require_write_access)
):
    """
    Upload text content as a new document.
    
    MULTI-TENANCY: Document is stored under the user's organization.
    Requires write access (admin or superuser role).
    """
    start_time = time.perf_counter()
    req = await request.json()
    filename = req.get("filename", "")
    content = req.get("content", "")
    validate_filename(filename)
    validate_text_length(content)
    # MULTI-TENANCY: Pass org_id to pipeline
    res = await pipeline.process_file(
        "virtual_file", 
        filename, 
        content.encode('utf-8'), 
        actor=user.uid, 
        org_id=user.org_id,
        background_tasks=background_tasks
    )
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Text upload processed: {filename} org={user.org_id} | {elapsed:.1f}ms")
    return {
        "status": "success", 
        "document_id": res["document_id"], 
        "filename": filename, 
        "chunks_count": res.get("chunks_count"), 
        "vectors_count": res.get("vectors_count"), 
        "storage_url": res.get("storage_url"), 
        "message": "Text uploaded successfully"
    }
