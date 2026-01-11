"""
Gemini Embedding Provider implementation.

Uses Google's Gemini API for text embedding generation.
Includes retry logic for transient failures and round-robin key rotation.
"""
from __future__ import annotations

import asyncio
import threading
from typing import List, Optional

from google import genai
from google.genai import types

from ...config import settings, get_logger
from ...exceptions import EmbeddingError
from ...utils import sanitize_for_embedding, is_retryable_error
from .interface import EmbeddingProviderInterface

logger = get_logger("embeddings.gemini")


class GeminiEmbeddingProvider(EmbeddingProviderInterface):
    """
    Gemini embedding provider with thread-safe round-robin API key rotation.
    
    Supports configurable model and dimensions with multiple API keys for
    load balancing and failover. Includes input sanitization and retry logic.
    """
    
    def __init__(self):
        self._clients: List[genai.Client] = []
        self._client_index = 0
        self._lock = threading.Lock()
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize Gemini clients from configured embedding API keys."""
        # Get all available keys (either dedicated EMBEDDING_API_KEY or shared GEMINI_API_KEYS)
        keys: List[str] = []
        
        if settings.EMBEDDING_API_KEY:
            keys.append(settings.EMBEDDING_API_KEY)
        
        # Add Gemini API keys from CSV if available
        if settings.GEMINI_API_KEYS:
            for key in settings.GEMINI_API_KEYS:
                if key not in keys:  # Avoid duplicates
                    keys.append(key)
        
        if not keys:
            logger.warning("No embedding API keys configured")
            return
        
        # Initialize a client for each key
        for key in keys:
            try:
                client = genai.Client(api_key=key)
                self._clients.append(client)
            except Exception as e:
                logger.error("Failed to init embedding client for key ...%s: %s", key[-4:], e)
        
        if self._clients:
            logger.info("Initialized %d embedding client(s) (model: %s, dims: %d)",
                       len(self._clients), settings.EMBEDDING_MODEL_ID, 
                       settings.EMBEDDING_DIMENSIONS)
        else:
            logger.warning("No embedding clients initialized")
    
    def _get_client(self) -> genai.Client:
        """Get next client using thread-safe round-robin."""
        if not self._clients:
            raise EmbeddingError("No embedding clients available", "API keys not configured")
        
        with self._lock:
            client = self._clients[self._client_index]
            self._client_index = (self._client_index + 1) % len(self._clients)
        return client
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text with retry logic and key rotation."""
        if not self._clients:
            logger.warning("No embedding clients available")
            return None
        
        # Sanitize input
        text = sanitize_for_embedding(text)
        if not text:
            return None
        
        last_error: Exception | None = None
        
        for attempt in range(settings.MAX_RETRIES + 1):
            try:
                # Get next client using round-robin
                client = self._get_client()
                
                response = await asyncio.wait_for(
                    client.aio.models.embed_content(
                        model=settings.EMBEDDING_MODEL_ID,
                        contents=text,
                        config=types.EmbedContentConfig(
                            output_dimensionality=settings.EMBEDDING_DIMENSIONS
                        ),
                    ),
                    timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
                )
                
                if response.embeddings:
                    return list(response.embeddings[0].values)
                return None
                
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    "Embedding generation timed out (attempt %d/%d)",
                    attempt + 1,
                    settings.MAX_RETRIES + 1,
                )
                if attempt >= settings.MAX_RETRIES:
                    logger.error("Embedding generation timed out after all retries")
                    return None
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for retryable errors
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    last_error = e
                    logger.warning(
                        "Embedding rate limit/quota hit (attempt %d/%d): %s",
                        attempt + 1,
                        settings.MAX_RETRIES + 1,
                        e,
                    )
                    if attempt >= settings.MAX_RETRIES:
                        logger.error("Embedding generation failed: rate limit exceeded")
                        return None
                        
                elif is_retryable_error(e):
                    last_error = e
                    logger.warning(
                        "Retryable embedding error (attempt %d/%d): %s",
                        attempt + 1,
                        settings.MAX_RETRIES + 1,
                        e,
                    )
                    if attempt >= settings.MAX_RETRIES:
                        logger.error("Embedding generation failed after all retries: %s", e)
                        return None
                else:
                    # Non-retryable error
                    logger.error("Embedding generation failed: %s", e)
                    return None
            
            # Wait before retry with exponential backoff
            if attempt < settings.MAX_RETRIES:
                delay = min(
                    settings.RETRY_BASE_DELAY * (2 ** attempt),
                    settings.RETRY_MAX_DELAY,
                )
                await asyncio.sleep(delay)
        
        return None
    
    async def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts concurrently."""
        if not texts:
            return []
        
        # Sanitize all inputs first
        sanitized = [sanitize_for_embedding(text) for text in texts]
        
        async def _get_embedding_or_none(text: str) -> Optional[List[float]]:
            if not text:
                return None
            return await self.generate_embedding(text)
        
        tasks = [_get_embedding_or_none(text) for text in sanitized]
        results = await asyncio.gather(*tasks)
        return list(results)
    
    def get_dimensions(self) -> int:
        """Get the embedding vector dimensions."""
        return settings.EMBEDDING_DIMENSIONS
    
    def get_model_name(self) -> str:
        """Get the embedding model identifier."""
        return settings.EMBEDDING_MODEL_ID
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "gemini"
    
    def is_available(self) -> bool:
        """Check if the embedding provider is available."""
        return len(self._clients) > 0
