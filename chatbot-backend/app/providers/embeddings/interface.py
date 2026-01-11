"""
Abstract interface for embedding generation providers.

All embedding providers must implement this interface to ensure
consistent behavior and easy hot-swapping.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProviderInterface(ABC):
    """
    Abstract interface for embedding generation providers.
    
    All implementations must provide:
    - Single text embedding generation
    - Batch embedding generation
    - Model and dimension information
    """
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector,
            or None if generation failed
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (or None for failed items)
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the embedding vector dimensions."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the embedding model identifier."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name (e.g., 'gemini', 'openai')."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available (has valid API keys)."""
        pass
