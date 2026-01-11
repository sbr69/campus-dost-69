"""
Chat endpoint with streaming response.

This module provides the main chat API endpoint with:
- Streaming responses (Server-Sent Events compatible)
- RAG (Retrieval Augmented Generation) context
- Request validation
- Rate limiting (configured via main.py)
- Comprehensive OpenAPI documentation
- Background metrics tracking (shared with admin backend)

Usage:
    POST /chat
    {
        "message": "What is machine learning?",
        "history": []
    }
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from fastapi.responses import StreamingResponse

from app.config import get_logger, settings
from app.exceptions import LLMError
from app.models import ChatRequest, ErrorResponse
from app.providers.metrics import metrics_provider
from app.services.chat import build_prompt, generate_chat_stream
from app.services.rag import get_rag_context
from app.state import AppState, get_app_state

if TYPE_CHECKING:
    from app.models import ChatMessage

logger = get_logger("routes.chat")

router = APIRouter(tags=["Chat"])


# =============================================================================
# Response Headers
# =============================================================================

STREAMING_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "X-Accel-Buffering": "no",  # Disable nginx buffering
    "Connection": "keep-alive",
}


# =============================================================================
# Chat Endpoint
# =============================================================================

async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    """
    Process chat request with RAG context and stream response.
    
    This endpoint:
    1. Validates the incoming request
    2. Truncates history if needed (preserves user/model pairs)
    3. Retrieves relevant RAG context
    4. Builds a structured prompt
    5. Streams the LLM response
    
    **Request Body:**
    - `message`: User's message (required, 1-10000 chars)
    - `history`: Conversation history (optional, max 20 messages)
    
    **Response:**
    - Streams plain text chunks as they are generated
    - Uses `text/plain` media type for broad compatibility
    
    **Headers:**
    - `X-History-Truncated`: Present if history was truncated
    - `Cache-Control: no-cache`: Prevents caching of streamed content
    
    **Rate Limiting:**
    - Configured via RATE_LIMIT environment variable (default: 100/minute)
    
    Args:
        request: FastAPI request object
        chat_request: Validated chat request payload
        state: Application state with providers
    
    Returns:
        StreamingResponse with generated text chunks
    
    Raises:
        LLMError: If the LLM provider is unavailable
        ValidationError: If request validation fails
    """
    start_time = time.perf_counter()
    
    # Get configuration
    max_history = settings.MAX_HISTORY_MESSAGES
    
    # Prepare history (truncate if needed while preserving pairs)
    history = list(chat_request.history)
    history_truncated = False
    original_history_len = len(history)

    if len(history) > max_history:
        # Truncate to keep most recent messages while preserving pairs
        truncate_to = max_history - (max_history % 2)
        logger.warning(
            "History truncated from %d to %d messages - context may be lost",
            len(history),
            truncate_to,
        )
        history = history[-truncate_to:]
        history_truncated = True

    logger.info(
        "Chat request | message_len=%d | history_len=%d (truncated=%s) | provider=%s",
        len(chat_request.message),
        len(history),
        history_truncated,
        state.llm_provider.get_provider_name(),
    )

    # Validate LLM provider availability
    if not state.llm_provider.is_available():
        provider_name = state.llm_provider.get_provider_name()
        logger.error("LLM provider not available: %s", provider_name)
        raise LLMError(
            message="Chat service temporarily unavailable",
            details=f"Provider {provider_name} is not available",
        )

    # Get RAG context
    rag_start = time.perf_counter()
    rag_results = await get_rag_context(chat_request.message, state, history)
    rag_elapsed_ms = (time.perf_counter() - rag_start) * 1000

    logger.info(
        "RAG complete | results=%d | elapsed_ms=%.1f",
        len(rag_results),
        rag_elapsed_ms,
    )

    # Build prompt with RAG context
    prompt = build_prompt(chat_request.message, rag_results)
    logger.debug("Prompt constructed | length=%d", len(prompt))

    async def stream_response() -> AsyncIterator[str]:
        """
        Stream wrapper that logs timing metrics.
        
        Yields:
            Text chunks from the LLM
        """
        chunk_count = 0
        bytes_sent = 0
        gen_start = time.perf_counter()
        success = False

        try:
            async for chunk in generate_chat_stream(
                prompt=prompt,
                history=history,
                system_instruction=state.system_instruction,
                llm_provider=state.llm_provider,
            ):
                chunk_count += 1
                bytes_sent += len(chunk.encode("utf-8"))
                yield chunk
            
            # Mark as successful if we completed without exception
            success = True

        finally:
            gen_elapsed_ms = (time.perf_counter() - gen_start) * 1000
            total_elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Stream complete | chunks=%d | bytes=%d | gen_ms=%.1f | total_ms=%.1f | success=%s",
                chunk_count,
                bytes_sent,
                gen_elapsed_ms,
                total_elapsed_ms,
                success,
            )
            
            # Track successful requests in metrics (fire-and-forget, non-blocking)
            # Uses asyncio.create_task to schedule without blocking the response
            # This writes to the same Firestore collection read by admin dashboard
            if success and chunk_count > 0:
                asyncio.create_task(metrics_provider.increment_daily_hit())

    # Build response headers
    headers = dict(STREAMING_HEADERS)
    
    # Add truncation info header if applicable
    if history_truncated:
        headers["X-History-Truncated"] = (
            f"true;original={original_history_len};kept={len(history)}"
        )

    return StreamingResponse(
        content=stream_response(),
        media_type="text/plain",
        headers=headers,
    )


# =============================================================================
# OpenAPI Documentation
# =============================================================================

# Note: The actual route is added in main.py with rate limiting.
# This file provides the endpoint logic and is imported there.

# Re-export dependencies for main.py
__all__ = [
    "chat_endpoint",
    "ChatRequest",
    "Depends",
    "get_app_state",
]
