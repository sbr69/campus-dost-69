"""
Embedding Provider - Factory module for embedding generation.

Selects the appropriate embedding implementation based on configuration.
"""

from app.config import settings, get_logger
from .interface import EmbeddingProviderInterface

logger = get_logger("embeddings.provider")

# Factory pattern - select implementation based on configuration
if settings.EMBEDDING_PROVIDER == "gemini":
    from .gemini_impl import GeminiEmbeddingProvider
    embedding_provider: EmbeddingProviderInterface = GeminiEmbeddingProvider()
    logger.info("Embedding Provider: Gemini (model: %s)", settings.EMBEDDING_MODEL_ID)
else:
    raise ValueError(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}. Supported: gemini")

__all__ = ["embedding_provider", "EmbeddingProviderInterface"]
