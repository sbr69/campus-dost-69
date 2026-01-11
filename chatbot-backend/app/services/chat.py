"""
Chat generation service.

This module provides chat completion functionality using the configured
LLM provider. It handles:
- Prompt construction with RAG context
- History format conversion
- Streaming response generation
- Error handling and graceful degradation

Usage:
    from app.services.chat import ChatService, generate_chat_stream
    
    async for chunk in generate_chat_stream(prompt, history, system, provider):
        print(chunk, end="")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, AsyncIterator, Sequence

from app.config import get_logger, settings
from app.exceptions import LLMError
from app.models import ChatMessage
from app.services.rag import RAGResult
from app.utils import sanitize_text

if TYPE_CHECKING:
    from app.providers.llm.interface import (
        ChatMessage as ProviderChatMessage,
        LLMProviderInterface,
    )

logger = get_logger("services.chat")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class PromptPayload:
    """
    Structured prompt payload sent to the LLM.
    
    Contains metadata, RAG context, and the user query in a JSON format
    optimized for LLM consumption.
    """
    query: str
    context: list[dict] | None = None
    metadata: dict = field(default_factory=dict)
    
    def to_json(self, compact: bool = True) -> str:
        """
        Serialize to JSON string.
        
        Args:
            compact: Use compact JSON format (no extra whitespace)
            
        Returns:
            JSON string representation
        """
        payload = {
            "meta": self.metadata,
            "context": self.context,
            "query": self.query,
        }
        
        if compact:
            return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        return json.dumps(payload, indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for chat generation."""
    temperature: float = field(default_factory=lambda: settings.GENERATION_TEMPERATURE)
    max_tokens: int = field(default_factory=lambda: settings.MAX_COMPLETION_TOKENS)
    max_message_length: int = field(default_factory=lambda: settings.MAX_MESSAGE_LENGTH)


# =============================================================================
# Chat Service Class
# =============================================================================

class ChatService:
    """
    Service for chat generation using LLM providers.
    
    This class encapsulates all chat-related logic including:
    - Prompt construction with RAG context
    - Message format conversion
    - Streaming generation
    - Error handling
    
    Example:
        >>> service = ChatService(llm_provider)
        >>> async for chunk in service.generate(prompt, history, system):
        ...     print(chunk, end="")
    """
    
    def __init__(
        self,
        llm_provider: "LLMProviderInterface",
        config: GenerationConfig | None = None,
    ) -> None:
        """
        Initialize chat service.
        
        Args:
            llm_provider: LLM provider for generation
            config: Optional custom configuration
        """
        self._provider = llm_provider
        self._config = config or GenerationConfig()
    
    @staticmethod
    def build_prompt(
        user_message: str,
        rag_results: Sequence[RAGResult],
        max_length: int | None = None,
    ) -> str:
        """
        Build a structured JSON prompt with RAG context and metadata.
        
        Args:
            user_message: User's query
            rag_results: Retrieved RAG context (sorted by similarity, highest first)
            max_length: Maximum message length (default from settings)
            
        Returns:
            JSON-formatted prompt string with individual chunks and metadata
        """
        max_length = max_length or settings.MAX_MESSAGE_LENGTH
        sanitized_message = sanitize_text(user_message, max_length=max_length)
        
        # Build structured context with individual chunks and similarity scores
        # Results are already sorted by similarity (highest to lowest)
        context_chunks = None
        metadata = {
            "chunks_count": len(rag_results),
            "has_context": len(rag_results) > 0
        }
        
        if rag_results:
            context_chunks = [
                {
                    "text": result.text,
                    "similarity": round(result.score, 3)  # Keep 3 decimal places
                }
                for result in rag_results
            ]
            # Add similarity range to metadata
            metadata["similarity_range"] = {
                "max": round(rag_results[0].score, 3),
                "min": round(rag_results[-1].score, 3)
            }
        
        # Create structured payload
        payload = {
            "metadata": metadata,
            "context": context_chunks,
            "query": sanitized_message
        }
        
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    
    @staticmethod
    def _extract_clean_message(content: str) -> str:
        """
        Extract clean message from content, removing RAG context if present.
        
        If the content is a JSON prompt with RAG context, extract just the query.
        Otherwise, return the content as-is.
        
        Args:
            content: Message content (may contain RAG JSON)
            
        Returns:
            Clean message text without RAG context
        """
        # Check if content looks like RAG JSON structure
        content = content.strip()
        if content.startswith("{") and '"query"' in content:
            try:
                data = json.loads(content)
                # If it has our RAG structure, extract just the query
                if isinstance(data, dict) and "query" in data:
                    return data["query"]
            except json.JSONDecodeError:
                # Not valid JSON, return as-is
                pass
        return content
    
    @staticmethod
    def convert_history_to_provider_format(
        history: Sequence[ChatMessage],
        system_instruction: str,
        prompt: str,
    ) -> list["ProviderChatMessage"]:
        """
        Convert internal chat history to provider message format.
        
        Strips RAG context from historical messages to avoid token waste.
        Only the current prompt includes RAG context.
        
        Args:
            history: Conversation history
            system_instruction: System prompt
            prompt: Current user prompt (with RAG context)
            
        Returns:
            List of provider ChatMessage objects with RAG stripped from history
        """
        # Import here to avoid circular imports
        from app.providers.llm.interface import ChatMessage as ProviderMessage
        
        messages: list[ProviderMessage] = [
            ProviderMessage(role="system", content=system_instruction)
        ]
        
        # Convert history - strip RAG context from historical messages
        for msg in history:
            role = "user" if msg.role == "user" else "assistant"
            content = "".join(msg.parts)
            
            # For user messages, extract clean query (remove RAG if present)
            if role == "user":
                content = ChatService._extract_clean_message(content)
            
            messages.append(ProviderMessage(role=role, content=content))
        
        # Add current prompt (ONLY this one has RAG context)
        messages.append(ProviderMessage(role="user", content=prompt))
        
        return messages
    
    async def generate_stream(
        self,
        prompt: str,
        history: Sequence[ChatMessage],
        system_instruction: str,
    ) -> AsyncIterator[str]:
        """
        Generate streaming chat response.
        
        Args:
            prompt: User prompt (typically JSON-formatted from build_prompt)
            history: Conversation history
            system_instruction: System prompt
            
        Yields:
            Content chunks from the LLM
            
        Note:
            On error, yields an error message chunk instead of raising.
            This allows the stream to complete gracefully.
        """
        messages = self.convert_history_to_provider_format(
            history, system_instruction, prompt
        )
        
        has_yielded = False
        
        try:
            async for chunk in self._provider.generate_stream(
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            ):
                has_yielded = True
                yield chunk
                
        except LLMError as e:
            logger.error(
                "LLM error during stream: %s (details: %s)",
                e.message,
                e.details,
            )
            # Only yield error message if we haven't started streaming
            if not has_yielded:
                yield f"[Error: {e.message}. Please try again.]"
            else:
                logger.warning("Stream interrupted after content was sent")
                
        except Exception as e:
            logger.error("Unexpected stream generation error: %s", e)
            if not has_yielded:
                yield settings.STREAM_ERROR_MESSAGE


# =============================================================================
# Convenience Functions (for backward compatibility)
# =============================================================================

def build_prompt(
    user_message: str,
    rag_results: Sequence[RAGResult],
) -> str:
    """
    Build structured JSON prompt with RAG context.
    
    This is a convenience function. For repeated calls or custom config,
    consider using ChatService.build_prompt() directly.
    
    Args:
        user_message: User's query
        rag_results: Retrieved RAG context
        
    Returns:
        JSON-formatted prompt string
    """
    return ChatService.build_prompt(user_message, rag_results)


async def generate_chat_stream(
    prompt: str,
    history: Sequence[ChatMessage],
    system_instruction: str,
    llm_provider: "LLMProviderInterface",
) -> AsyncIterator[str]:
    """
    Generate streaming response using the configured LLM provider.
    
    This is a convenience function that creates a ChatService instance.
    For repeated calls, consider using ChatService directly.
    
    Args:
        prompt: User prompt (typically JSON-formatted from build_prompt)
        history: Conversation history
        system_instruction: System prompt
        llm_provider: LLM provider for generation
        
    Yields:
        Content chunks from the LLM
    """
    service = ChatService(llm_provider)
    async for chunk in service.generate_stream(prompt, history, system_instruction):
        yield chunk


# =============================================================================
# Internal Helper Functions (kept for backward compatibility)
# =============================================================================

def _convert_history_to_provider_format(
    history: Sequence[ChatMessage],
    system_instruction: str,
    prompt: str,
) -> list["ProviderChatMessage"]:
    """
    Convert internal chat history to provider message format.
    
    Deprecated: Use ChatService.convert_history_to_provider_format() instead.
    """
    return ChatService.convert_history_to_provider_format(
        history, system_instruction, prompt
    )
