"""
Centralized configuration using Pydantic BaseSettings.

This module follows industry best practices for FastAPI configuration:
- Pydantic BaseSettings for type-safe environment variable loading
- Validation at startup (fail-fast)
- Thread-safe API key rotation
- Comprehensive documentation
- Environment file support (.env)

Configuration Philosophy:
    - .env: Only sensitive data (API keys, credentials)
    - config.py: All application settings with sensible defaults
    
Usage:
    from app.config import settings, get_logger
    
    print(settings.HOST)  # Type-safe access
"""
from __future__ import annotations

import logging
import re
import threading
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings are validated at startup. Required variables
    will raise a validation error if not set.
    
    Thread-safe round-robin key rotation is implemented using locks.
    
    Configuration Sources:
        1. Environment variables
        2. .env file (if present)
        3. Default values (defined below)
    
    Design: .env contains ONLY sensitive data (API keys, credentials).
            All other config has sensible defaults here.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
        validate_default=True,
    )
    
    # =========================================================================
    # Server Configuration
    # =========================================================================
    # Note: These rarely change and have sensible defaults for most deployments
    
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host (0.0.0.0 for external access, 127.0.0.1 for local only)",
    )
    PORT: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Server port number",
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode - enables detailed error messages (NEVER use in production)",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    
    # =========================================================================
    # LLM Provider Configuration
    # =========================================================================
    # Sensitive: API keys loaded from .env
    # Configuration: Model IDs can change as new models are released
    
    # Provider Selection
    LLM_PROVIDER: Literal["groq", "gemini"] = Field(
        default="groq",
        description="Active LLM provider (groq recommended for speed)",
    )
    
    # Groq Configuration (https://console.groq.com)
    GROQ_API_KEY: str | None = Field(
        default=None,
        description="Single Groq API key (alternative to CSV)",
    )
    GROQ_API_KEYS_CSV: str = Field(
        default="",
        description="Comma-separated Groq API keys for automatic load balancing",
    )
    GROQ_MODEL_ID: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Groq model identifier (changes as models update)",
    )
    
    # Gemini Configuration (https://makersuite.google.com/app/apikey)
    GEMINI_API_KEYS_CSV: str = Field(
        default="",
        description="Comma-separated Gemini API keys for automatic load balancing",
    )
    GEMINI_MODEL_ID: str = Field(
        default="gemini-2.0-flash-lite",
        description="Gemini chat model identifier (changes as models update)",
    )
    
    # =========================================================================
    # Embedding Provider Configuration
    # =========================================================================
    # Sensitive: API keys loaded from .env
    # Note: Embedding dimensions must match the model (768 for gemini-embedding-001)
    
    EMBEDDING_PROVIDER: Literal["gemini"] = Field(
        default="gemini",
        description="Embedding provider (currently only Gemini supported)",
    )
    EMBEDDING_API_KEY: str | None = Field(
        default=None,
        description="Dedicated embedding API key (optional, falls back to GEMINI_API_KEYS_CSV)",
    )
    EMBEDDING_MODEL_ID: str = Field(
        default="gemini-embedding-001",
        description="Embedding model (official Google recommendation for 768-dim vectors)",
    )
    
    # Embedding Configuration (application settings, not sensitive)
    EMBEDDING_DIMENSIONS: int = Field(
        default=768,
        ge=1,
        description="Vector dimensions (must match model: 768 for gemini-embedding-001)",
    )
    EMBEDDING_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0,
        description="Maximum time to wait for embedding generation",
    )
    
    # =========================================================================
    # Database / Vector Store Configuration
    # =========================================================================
    # Sensitive: Firebase credentials loaded from .env
    # Note: Collection and field names are application constants with defaults
    
    DATABASE_PROVIDER: Literal["firestore"] = Field(
        default="firestore",
        description="Vector database provider (currently only Firestore supported)",
    )
    FIREBASE_CREDS_BASE64: str | None = Field(
        default=None,
        description="Base64-encoded Firebase service account JSON (alternative to GOOGLE_APPLICATION_CREDENTIALS)",
    )
    
    # Firestore Collection Configuration (application constants)
    FIRESTORE_VECTOR_COLLECTION: str = Field(
        default="vector_store",
        description="Firestore collection name for vector embeddings and documents",
    )
    FIRESTORE_VECTOR_FIELD: str = Field(
        default="embedding",
        description="Document field name containing the embedding vector",
    )
    FIRESTORE_QUERY_TIMEOUT_SECONDS: float = Field(
        default=8.0,
        gt=0,
        description="Maximum time to wait for Firestore queries",
    )
    
    # =========================================================================
    # Metrics Configuration
    # =========================================================================
    # Shared collection with admin backend for unified analytics
    
    FIRESTORE_WEEKLY_METRICS_COLLECTION: str = Field(
        default="weekly_metrics",
        description="Firestore collection for daily/weekly usage metrics (shared with admin backend)",
    )
    
    # =========================================================================
    # RAG (Retrieval-Augmented Generation) Configuration
    # =========================================================================
    # Tuning parameters for semantic search and context retrieval
    
    RAG_TOP_K: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of most similar documents to retrieve for context",
    )
    RAG_SIMILARITY_THRESHOLD: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score (0-1) to include a document",
    )
    
    # =========================================================================
    # Chat Configuration
    # =========================================================================
    # LLM generation and conversation management settings
    
    GENERATION_TEMPERATURE: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0=deterministic, 2=very creative)",
    )
    MAX_COMPLETION_TOKENS: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens in LLM response (prevents runaway generation)",
    )
    CHAT_COMPLETION_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        gt=0,
        description="Maximum time to wait for LLM response before timeout",
    )
    MAX_HISTORY_MESSAGES: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum conversation history messages to retain (prevents context overflow)",
    )
    MAX_MESSAGE_LENGTH: int = Field(
        default=10000,
        ge=1,
        description="Maximum user message length in characters",
    )
    MAX_REQUEST_SIZE: int = Field(
        default=1_000_000,
        ge=1,
        description="Maximum HTTP request body size in bytes (1MB default)",
    )
    STREAM_ERROR_MESSAGE: str = Field(
        default="\n\n[Error: Failed to complete response. Please try again.]",
        description="User-facing error message for stream failures",
    )
    
    # =========================================================================
    # Retry Configuration
    # =========================================================================
    # Exponential backoff for transient failures (network issues, rate limits, etc.)
    
    MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests",
    )
    RETRY_BASE_DELAY: float = Field(
        default=0.5,
        ge=0,
        description="Initial delay between retries in seconds (grows exponentially)",
    )
    RETRY_MAX_DELAY: float = Field(
        default=10.0,
        ge=0,
        description="Maximum delay between retries (caps exponential growth)",
    )
    
    # =========================================================================
    # Rate Limiting
    # =========================================================================
    # Protects API from abuse and ensures fair resource allocation
    
    RATE_LIMIT: str = Field(
        default="100/minute",
        pattern=r"^\d+/(second|minute|hour|day)$",
        description="Rate limit for chat endpoint (format: 'count/period')",
    )
    
    # =========================================================================
    # CORS Configuration
    # =========================================================================
    # Cross-Origin Resource Sharing for frontend integration
    
    CORS_ORIGINS: str = Field(
        default="*",
        description="Comma-separated allowed origins ('*' for all, restrict in production)",
    )
    
    # =========================================================================
    # Internal State (not from environment)
    # =========================================================================
    
    # Thread-safe locks for key rotation (ClassVar = not included in model)
    _groq_lock: ClassVar[threading.Lock] = threading.Lock()
    _gemini_lock: ClassVar[threading.Lock] = threading.Lock()
    
    # Key indices (instance state, managed via property)
    _groq_key_index: int = 0
    _gemini_key_index: int = 0
    
    # =========================================================================
    # Validators
    # =========================================================================
    
    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def uppercase_log_level(cls, v: str) -> str:
        """Ensure LOG_LEVEL is uppercase."""
        return v.upper() if isinstance(v, str) else v
    
    @field_validator("LLM_PROVIDER", "EMBEDDING_PROVIDER", "DATABASE_PROVIDER", mode="before")
    @classmethod
    def lowercase_provider(cls, v: str) -> str:
        """Ensure provider names are lowercase."""
        return v.lower() if isinstance(v, str) else v
    
    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        """Validate that required API keys are present for the selected provider."""
        # Validate LLM provider keys
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEYS:
            raise ValueError(
                "Groq provider requires GROQ_API_KEY or GROQ_API_KEYS_CSV"
            )
        if self.LLM_PROVIDER == "gemini" and not self.GEMINI_API_KEYS:
            raise ValueError(
                "Gemini LLM provider requires GEMINI_API_KEYS_CSV"
            )
        
        # Validate embedding provider keys
        if self.EMBEDDING_PROVIDER == "gemini":
            if not self.EMBEDDING_API_KEY and not self.GEMINI_API_KEYS:
                raise ValueError(
                    "Gemini embedding provider requires EMBEDDING_API_KEY or GEMINI_API_KEYS_CSV"
                )
        
        return self
    
    # =========================================================================
    # Computed Properties
    # =========================================================================
    
    @cached_property
    def GROQ_API_KEYS(self) -> list[str]:
        """Get list of Groq API keys (deduplicated)."""
        keys: list[str] = []
        if self.GROQ_API_KEYS_CSV:
            keys.extend(k.strip() for k in self.GROQ_API_KEYS_CSV.split(",") if k.strip())
        if self.GROQ_API_KEY and self.GROQ_API_KEY not in keys:
            keys.append(self.GROQ_API_KEY)
        return list(dict.fromkeys(keys))  # Preserve order, remove duplicates
    
    @cached_property
    def GEMINI_API_KEYS(self) -> list[str]:
        """Get list of Gemini API keys (deduplicated)."""
        if not self.GEMINI_API_KEYS_CSV:
            return []
        keys = [k.strip() for k in self.GEMINI_API_KEYS_CSV.split(",") if k.strip()]
        return list(dict.fromkeys(keys))
    
    @cached_property
    def CORS_ORIGINS_LIST(self) -> list[str]:
        """Get list of CORS origins."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
    
    @cached_property
    def BASE_DIR(self) -> Path:
        """Get base directory (project root)."""
        return Path(__file__).parent.parent
    
    @cached_property
    def SYSTEM_INSTRUCTION_PATH(self) -> Path:
        """Get system instruction file path."""
        return self.BASE_DIR / "system_instruction.txt"
    
    # =========================================================================
    # Thread-Safe Round-Robin Key Rotation (for LLM Providers)
    # =========================================================================
    # Note: Embedding provider handles its own key rotation with multiple clients
    
    def get_groq_api_key(self) -> str | None:
        """Get next Groq API key using thread-safe round-robin."""
        keys = self.GROQ_API_KEYS
        if not keys:
            return None
        with self._groq_lock:
            key = keys[self._groq_key_index % len(keys)]
            self._groq_key_index = (self._groq_key_index + 1) % len(keys)
        return key
    
    def get_gemini_api_key(self) -> str | None:
        """Get next Gemini API key using thread-safe round-robin."""
        keys = self.GEMINI_API_KEYS
        if not keys:
            return None
        with self._gemini_lock:
            key = keys[self._gemini_key_index % len(keys)]
            self._gemini_key_index = (self._gemini_key_index + 1) % len(keys)
        return key


# =============================================================================
# Settings Factory with Caching
# =============================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache for singleton behavior while allowing
    cache invalidation in tests.
    """
    return Settings()


# Convenience alias for direct access
settings = get_settings()


# =============================================================================
# Logging Configuration
# =============================================================================

class SanitizingFormatter(logging.Formatter):
    """
    Logging formatter that redacts sensitive information.
    
    Automatically redacts:
    - Bearer tokens
    - API keys
    - Tokens
    - Passwords
    """
    
    SENSITIVE_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r'(Bearer\s+)[^\s]+', re.I), r'\1[REDACTED]'),
        (re.compile(r'(api[_-]?key["\']?\s*[:=]\s*["\']?)[^"\'\s]+', re.I), r'\1[REDACTED]'),
        (re.compile(r'(token["\']?\s*[:=]\s*["\']?)[^"\'\s]+', re.I), r'\1[REDACTED]'),
        (re.compile(r'(password["\']?\s*[:=]\s*["\']?)[^"\'\s]+', re.I), r'\1[REDACTED]'),
    ]
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sensitive data redaction."""
        message = super().format(record)
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            message = pattern.sub(replacement, message)
        return message


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    handler = logging.StreamHandler()
    handler.setFormatter(SanitizingFormatter(log_format))
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[handler],
        force=True,  # Override any existing configuration
    )
    
    # Suppress noisy third-party loggers
    for logger_name in ("httpx", "httpcore", "google", "urllib3", "grpc"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured logging.Logger instance
    """
    return logging.getLogger(name)


# Initialize logging on module load
configure_logging(settings.LOG_LEVEL)
