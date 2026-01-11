"""
LLM Provider - Factory module for chat completion providers.

Selects the appropriate LLM implementation based on configuration.
"""

from app.config import settings, get_logger
from .interface import LLMProviderInterface

logger = get_logger("llm.provider")

# Factory pattern - select implementation based on configuration
if settings.LLM_PROVIDER == "groq":
    from .groq_impl import GroqLLMProvider
    llm_provider: LLMProviderInterface = GroqLLMProvider()
    logger.info("LLM Provider: Groq (model: %s)", settings.GROQ_MODEL_ID)
elif settings.LLM_PROVIDER == "gemini":
    from .gemini_impl import GeminiLLMProvider
    llm_provider: LLMProviderInterface = GeminiLLMProvider()
    logger.info("LLM Provider: Gemini (model: %s)", settings.GEMINI_MODEL_ID)
else:
    raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}. Supported: groq, gemini")

__all__ = ["llm_provider", "LLMProviderInterface"]
