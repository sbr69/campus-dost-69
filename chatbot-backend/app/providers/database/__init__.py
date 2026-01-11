"""
Database Provider - Factory module for vector store operations.

Selects the appropriate database implementation based on configuration.
"""

from app.config import settings, get_logger
from .interface import DatabaseProviderInterface

logger = get_logger("database.provider")

# Factory pattern - select implementation based on configuration
if settings.DATABASE_PROVIDER == "firestore":
    from .firestore_impl import FirestoreDatabaseProvider
    database_provider: DatabaseProviderInterface = FirestoreDatabaseProvider()
    logger.info("Database Provider: Firestore (collection: %s)", 
                settings.FIRESTORE_VECTOR_COLLECTION)
else:
    raise ValueError(f"Unknown database provider: {settings.DATABASE_PROVIDER}. Supported: firestore")

__all__ = ["database_provider", "DatabaseProviderInterface"]
