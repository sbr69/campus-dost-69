"""
Application state management using provider-based architecture.

This module provides centralized state management that leverages
the hot-swappable provider pattern for all external services.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request

from app.config import settings, get_logger
from app.providers.llm import llm_provider, LLMProviderInterface
from app.providers.embeddings import embedding_provider, EmbeddingProviderInterface
from app.providers.database import database_provider, DatabaseProviderInterface

logger = get_logger("state")


@dataclass
class AppState:
    """
    Central container for shared application resources.
    
    Uses the provider pattern for all external services, enabling
    hot-swapping of implementations without code changes.
    """
    llm_provider: LLMProviderInterface
    embedding_provider: EmbeddingProviderInterface
    database_provider: DatabaseProviderInterface
    system_instruction: str = field(default="You are a helpful AI assistant.")
    
    @classmethod
    async def create(cls) -> "AppState":
        """
        Create and initialize application state.
        
        Returns:
            Initialized AppState instance
        
        Raises:
            RuntimeError: If critical providers fail to initialize.
        """
        # Initialize database provider
        db_initialized = await database_provider.initialize()
        if not db_initialized:
            logger.critical("Database provider failed to initialize. Aborting startup.")
            raise RuntimeError("Critical Dependency Failed: Database Provider could not be initialized.")
        
        # Load system instruction
        system_instruction = cls._load_system_instruction()
        
        # Log provider status
        logger.info(
            "Providers: LLM=%s (%s) | Embedding=%s (%s) | Database=%s (%s)",
            llm_provider.get_provider_name(),
            "OK" if llm_provider.is_available() else "UNAVAILABLE",
            embedding_provider.get_provider_name(),
            "OK" if embedding_provider.is_available() else "UNAVAILABLE",
            database_provider.get_provider_name(),
            "OK" if database_provider.is_available() else "UNAVAILABLE",
        )
        
        return cls(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            database_provider=database_provider,
            system_instruction=system_instruction,
        )
    
    @staticmethod
    def _load_system_instruction() -> str:
        """Load system instruction from file."""
        try:
            if settings.SYSTEM_INSTRUCTION_PATH.exists():
                content = settings.SYSTEM_INSTRUCTION_PATH.read_text(encoding="utf-8")
                logger.info("System instruction loaded from %s", settings.SYSTEM_INSTRUCTION_PATH)
                return content
            logger.warning("system_instruction.txt not found, using default")
        except Exception as e:
            logger.error("Failed to load system instruction: %s", e)
        
        return "You are a helpful AI assistant."
    
    def is_ready(self) -> bool:
        """Check if the application is ready to handle requests."""
        return (
            self.llm_provider.is_available() and
            self.database_provider.is_available()
        )


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency to get application state."""
    if not hasattr(request.app.state, "app_state"):
        raise RuntimeError("Application state not initialized")
    return request.app.state.app_state
