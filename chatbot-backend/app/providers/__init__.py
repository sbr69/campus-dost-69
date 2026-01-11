"""
Provider layer for swappable implementations.

This module follows the service-as-architecture pattern where each provider
type has an abstract interface that concrete implementations must satisfy.

Providers are selected at runtime based on configuration, allowing hot-swapping
of components without code changes.

Directory Structure:
    providers/
    ├── __init__.py           # This file - exports provider instances
    ├── llm/                   # LLM chat completion providers
    │   ├── __init__.py        # Factory - selects provider based on config
    │   ├── interface.py       # Abstract interface all LLMs must implement
    │   ├── groq_impl.py       # Groq implementation
    │   └── gemini_impl.py     # Gemini implementation
    ├── embeddings/            # Embedding generation providers
    │   ├── __init__.py        # Factory - selects provider
    │   ├── interface.py       # Abstract interface
    │   └── gemini_impl.py     # Gemini embedding implementation
    └── database/              # Vector store / database providers
        ├── __init__.py        # Factory - selects provider
        ├── interface.py       # Abstract interface
        └── firestore_impl.py  # Firestore implementation
"""

# Lazy imports to avoid circular dependencies
# Providers are instantiated by their respective __init__.py files

__all__ = []
