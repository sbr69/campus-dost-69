"""
Custom exceptions for the chatbot backend.

This module provides a consistent exception hierarchy for error handling
across all providers and services.

Exception Hierarchy:
    ChatbotException (base)
    ├── ConfigurationError (500)
    ├── ValidationError (400)
    ├── RateLimitError (429)
    ├── LLMError (503)
    ├── EmbeddingError (503)
    ├── DatabaseError (503)
    └── RAGError (503)

Usage:
    from app.exceptions import LLMError
    
    raise LLMError("Generation failed", details="API key invalid")
"""
from __future__ import annotations

from typing import Any


class ChatbotException(Exception):
    """
    Base exception for all chatbot errors.
    
    All custom exceptions inherit from this class, enabling
    consistent error handling at the API layer.
    
    Attributes:
        message: Human-readable error message
        status_code: HTTP status code for API response
        details: Additional error details (optional)
        error_code: Machine-readable error code (optional)
    """
    
    default_message: str = "An error occurred"
    default_status_code: int = 500
    
    def __init__(
        self,
        message: str | None = None,
        status_code: int | None = None,
        details: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Error message (uses default if not provided)
            status_code: HTTP status code (uses default if not provided)
            details: Additional details about the error
            error_code: Machine-readable error code
        """
        self.message = message or self.default_message
        self.status_code = status_code or self.default_status_code
        self.details = details
        self.error_code = error_code or self.__class__.__name__.upper()
        super().__init__(self.message)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for API response.
        
        Returns:
            Dictionary with error details
        """
        result = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code})"


# =============================================================================
# Configuration Errors (500)
# =============================================================================

class ConfigurationError(ChatbotException):
    """
    Raised when configuration is invalid or missing.
    
    Examples:
        - Missing required environment variables
        - Invalid configuration values
        - Failed to load configuration files
    """
    
    default_message = "Configuration error"
    default_status_code = 500


# =============================================================================
# Client Errors (4xx)
# =============================================================================

class ValidationError(ChatbotException):
    """
    Raised when input validation fails.
    
    Examples:
        - Invalid request body
        - Message too long
        - Invalid history format
    """
    
    default_message = "Validation error"
    default_status_code = 400


class RateLimitError(ChatbotException):
    """
    Raised when rate limits are exceeded.
    
    Examples:
        - Too many requests per minute
        - API quota exhausted
    """
    
    default_message = "Rate limit exceeded"
    default_status_code = 429


# =============================================================================
# Service Errors (503)
# =============================================================================

class LLMError(ChatbotException):
    """
    Raised when LLM operations fail.
    
    Examples:
        - API call failure
        - Generation timeout
        - Invalid model response
    """
    
    default_message = "LLM service error"
    default_status_code = 503


class EmbeddingError(ChatbotException):
    """
    Raised when embedding generation fails.
    
    Examples:
        - Embedding API failure
        - Invalid input text
        - Rate limit on embedding service
    """
    
    default_message = "Embedding service error"
    default_status_code = 503


class DatabaseError(ChatbotException):
    """
    Raised when database operations fail.
    
    Examples:
        - Connection failure
        - Query timeout
        - Authentication error
    """
    
    default_message = "Database service error"
    default_status_code = 503


class RAGError(ChatbotException):
    """
    Raised when RAG operations fail.
    
    Examples:
        - Vector search failure
        - Context retrieval error
        - Embedding + search combined failure
    """
    
    default_message = "RAG service error"
    default_status_code = 503


# =============================================================================
# Utility Functions
# =============================================================================

def is_retryable_exception(exc: Exception) -> bool:
    """
    Check if an exception is potentially retryable.
    
    Args:
        exc: Exception to check
        
    Returns:
        True if the exception might succeed on retry
    """
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, (LLMError, EmbeddingError, DatabaseError)):
        # Check if it's a transient error
        error_str = str(exc).lower()
        transient_indicators = [
            "timeout", "connection", "network",
            "temporarily", "unavailable", "retry",
            "429", "503", "504",
        ]
        return any(indicator in error_str for indicator in transient_indicators)
    return False
