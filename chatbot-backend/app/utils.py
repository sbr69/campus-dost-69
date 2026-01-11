"""
Utility functions for the chatbot backend.

Provides common functionality for:
- Input sanitization
- Retry logic with exponential backoff
- Text processing utilities
"""
from __future__ import annotations

import asyncio
import re
import unicodedata
from functools import wraps
from typing import Any, Callable, TypeVar, ParamSpec

from app.config import settings, get_logger

logger = get_logger("utils")

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# Input Sanitization
# =============================================================================

# Patterns for potentially dangerous content
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
_EXCESSIVE_WHITESPACE = re.compile(r'\s{10,}')
_NULL_BYTES = re.compile(r'\x00')


def sanitize_text(text: str, max_length: int | None = None) -> str:
    """
    Sanitize user input text for safe processing.
    
    - Removes control characters
    - Normalizes Unicode (NFC form)
    - Removes null bytes
    - Collapses excessive whitespace
    - Strips leading/trailing whitespace
    - Optionally truncates to max_length
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (None for no limit)
        
    Returns:
        Sanitized text string
    """
    if not text:
        return ""
    
    # Normalize Unicode to NFC form
    text = unicodedata.normalize("NFC", text)
    
    # Remove null bytes
    text = _NULL_BYTES.sub("", text)
    
    # Remove control characters (except newlines and tabs)
    text = _CONTROL_CHARS.sub("", text)
    
    # Collapse excessive whitespace (more than 10 consecutive)
    text = _EXCESSIVE_WHITESPACE.sub(" ", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]
        logger.debug("Text truncated to %d characters", max_length)
    
    return text


def sanitize_for_embedding(text: str) -> str:
    """
    Sanitize text specifically for embedding generation.
    
    Additional processing:
    - Removes excessive newlines
    - Normalizes quotes and dashes
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text suitable for embedding
    """
    text = sanitize_text(text, max_length=settings.MAX_MESSAGE_LENGTH)
    
    if not text:
        return ""
    
    # Normalize various quote styles to standard ASCII
    text = re.sub(r'[""„]', '"', text)
    text = re.sub(r"[''`]", "'", text)
    
    # Normalize various dash styles
    text = re.sub(r'[–—]', '-', text)
    
    # Collapse multiple newlines to double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


# =============================================================================
# Retry Logic with Exponential Backoff
# =============================================================================

class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(self, message: str, last_exception: Exception | None = None):
        self.message = message
        self.last_exception = last_exception
        super().__init__(message)


def is_retryable_error(exc: Exception) -> bool:
    """
    Determine if an exception is retryable.
    
    Retryable errors include:
    - Timeout errors
    - Connection errors
    - Rate limit errors (429)
    - Server errors (5xx)
    """
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    
    # Timeout errors
    if "timeout" in exc_str or "timeout" in exc_type:
        return True
    
    # Connection errors
    if any(term in exc_str for term in ["connection", "connect", "network", "socket"]):
        return True
    
    # Rate limit (429) or server errors (5xx)
    if "429" in exc_str or "rate limit" in exc_str:
        return True
    if any(f"{code}" in exc_str for code in range(500, 600)):
        return True
    
    # Specific exception types
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    
    return False


async def retry_async(
    func: Callable[P, T],
    *args: P.args,
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    **kwargs: P.kwargs,
) -> T:
    """
    Execute an async function with retry logic and exponential backoff.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        max_retries: Maximum retry attempts (default from settings)
        base_delay: Base delay in seconds (default from settings)
        max_delay: Maximum delay in seconds (default from settings)
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call
        
    Raises:
        RetryError: If all retries are exhausted
    """
    max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
    base_delay = base_delay if base_delay is not None else settings.RETRY_BASE_DELAY
    max_delay = max_delay if max_delay is not None else settings.RETRY_MAX_DELAY
    
    last_exception: Exception | None = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if not is_retryable_error(e):
                # Non-retryable error, raise immediately
                raise
            
            if attempt >= max_retries:
                # All retries exhausted
                logger.error(
                    "All %d retry attempts exhausted for %s: %s",
                    max_retries,
                    func.__name__,
                    e,
                )
                raise RetryError(
                    f"Operation failed after {max_retries + 1} attempts",
                    last_exception=e,
                )
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            # Add 10% jitter
            import random
            delay *= (0.9 + random.random() * 0.2)
            
            logger.warning(
                "Attempt %d/%d failed for %s (%s), retrying in %.2fs",
                attempt + 1,
                max_retries + 1,
                func.__name__,
                type(e).__name__,
                delay,
            )
            
            await asyncio.sleep(delay)
    
    # Should not reach here, but just in case
    raise RetryError("Unexpected retry loop exit", last_exception=last_exception)


def with_retry(
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
):
    """
    Decorator to add retry logic to async functions.
    
    Usage:
        @with_retry(max_retries=3)
        async def my_function():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await retry_async(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                **kwargs,
            )
        return wrapper
    return decorator


# =============================================================================
# Text Processing Utilities
# =============================================================================

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to max_length, adding suffix if truncated.
    
    Attempts to truncate at word boundaries when possible.
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    target_length = max_length - len(suffix)
    if target_length <= 0:
        return suffix[:max_length]
    
    # Try to truncate at word boundary
    truncated = text[:target_length]
    last_space = truncated.rfind(" ")
    
    if last_space > target_length * 0.7:  # Only if not too far back
        truncated = truncated[:last_space]
    
    return truncated.rstrip() + suffix


def normalize_query(query: str) -> str:
    """
    Normalize a query string for consistent comparison.
    
    - Lowercase
    - Normalize Unicode
    - Remove extra whitespace
    - Remove punctuation
    """
    query = sanitize_text(query)
    query = query.lower()
    query = re.sub(r'[^\w\s]', ' ', query)  # Remove punctuation
    query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
    return query.strip()
