"""
Upload router for file ingestion.

MULTI-TENANCY: All uploads are scoped to the user's organization.
"""
import time
import os
import tempfile
from pathlib import Path
import aiofiles
from fastapi import APIRouter, Request, BackgroundTasks, UploadFile, File, Depends, HTTPException
from ..config import settings, logger
from ..dependencies import require_write_access, UserContext
from ..utils.limiter import limiter
from ..services.ingestion import pipeline
from ..exceptions import UnsupportedFileTypeError, FileSizeError

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

@router.post("")
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_file(
    request: Request, 
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    user: UserContext = Depends(require_write_access)
):
    """
    Upload and process a file.
    
    MULTI-TENANCY: File is stored under the user's organization.
    Requires write access (admin or superuser role).
    """
    start_time = time.perf_counter()
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"File type {ext} not supported")
    
    # Strictly disallow Archives since we removed ArchiveExtractor
    if ext in [".zip", ".tar", ".gz", ".7z"]:
        raise HTTPException(400, "Archive files are not supported in this configuration.")

    tmp_path = None
    try:
        # Use aiofiles for non-blocking I/O
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
        
        async with aiofiles.open(tmp_path, 'wb') as tmp:
            chunk_size = 1024 * 1024
            total_size = 0
            while True:
                chunk = await file.read(chunk_size)
                if not chunk: break
                total_size += len(chunk)
                if total_size > settings.MAX_FILE_SIZE:
                    raise FileSizeError(f"File exceeds maximum size of {settings.MAX_FILE_SIZE} bytes")
                await tmp.write(chunk)
        
        async with aiofiles.open(tmp_path, "rb") as f:
            content = await f.read()
        
        # MULTI-TENANCY: Pass org_id to pipeline
        res = await pipeline.process_file(
            tmp_path, 
            file.filename, 
            content, 
            actor=user.uid, 
            org_id=user.org_id,
            background_tasks=background_tasks
        )
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Upload endpoint complete: {file.filename} org={user.org_id} | {elapsed:.1f}ms")
        return {**res, "message": "Uploaded successfully"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
