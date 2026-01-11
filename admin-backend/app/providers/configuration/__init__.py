"""
Configuration provider - supports both GitHub and Firestore implementations.

MULTI-TENANCY: Uses Firestore implementation with org-specific LRU cache.
"""
from .interface import ConfigProviderInterface
from .firestore_impl import firestore_config_provider, FirestoreConfigProvider

# Use Firestore implementation for multi-tenant org-specific instructions
config_provider: FirestoreConfigProvider = firestore_config_provider

__all__ = ['config_provider', 'ConfigProviderInterface', 'FirestoreConfigProvider']
