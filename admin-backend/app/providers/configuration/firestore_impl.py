"""
Firestore-based system instructions storage with org-specific LRU cache.

MULTI-TENANCY: Each organization has their own system instructions stored in Firestore.
LRU cache is mapped by org_id to prevent cross-org data leakage and reduce DB hits.
"""
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from collections import OrderedDict
from ...config import settings, logger
from ...exceptions import DatabaseError
from ..database.firestore_init import get_db
from .interface import ConfigProviderInterface


class OrgLRUCache:
    """
    LRU Cache mapped by org_id for system instructions.
    
    Features:
    - Per-org cache entries prevent cross-org data conflicts
    - TTL-based expiration for freshness
    - Max entries limit to bound memory usage
    - Thread-safe for async operations
    """
    
    def __init__(self, max_entries: int = 100, ttl_seconds: int = 300):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
    
    def get(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get cached instructions for org_id if not expired."""
        if org_id not in self._cache:
            return None
        
        entry = self._cache[org_id]
        
        # Check TTL
        if time.time() - entry['cached_at'] > self._ttl_seconds:
            # Expired, remove from cache
            del self._cache[org_id]
            logger.debug(f"Cache expired for org={org_id}")
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(org_id)
        logger.debug(f"Cache hit for org={org_id}")
        return entry['data']
    
    def set(self, org_id: str, data: Dict[str, Any]) -> None:
        """Cache instructions for org_id."""
        # Remove oldest if at capacity
        while len(self._cache) >= self._max_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted oldest entry for org={oldest_key}")
        
        self._cache[org_id] = {
            'data': data,
            'cached_at': time.time()
        }
        # Move to end (most recently used)
        self._cache.move_to_end(org_id)
        logger.debug(f"Cache set for org={org_id}")
    
    def invalidate(self, org_id: str) -> None:
        """Invalidate cache for specific org_id."""
        if org_id in self._cache:
            del self._cache[org_id]
            logger.debug(f"Cache invalidated for org={org_id}")
    
    def invalidate_all(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        logger.info("Cache cleared for all orgs")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'entries': len(self._cache),
            'max_entries': self._max_entries,
            'ttl_seconds': self._ttl_seconds,
            'orgs_cached': list(self._cache.keys())
        }


class FirestoreConfigProvider(ConfigProviderInterface):
    """
    Firestore-based system instructions storage with org-specific caching.
    
    Storage Structure:
    - Collection: system_instructions
    - Document ID: {org_id}
    - Fields: content, updated_at, updated_by, version
    
    Cache Strategy:
    - LRU cache mapped by org_id
    - 5-minute TTL for warm start efficiency
    - Cache invalidated on save
    """
    
    # Class-level cache shared across all instances (singleton pattern)
    _cache = OrgLRUCache(
        max_entries=100,  # Support up to 100 orgs in memory
        ttl_seconds=300   # 5 minutes TTL
    )
    
    COLLECTION_NAME = "system_instructions"
    
    def __init__(self):
        self._db = None
    
    @property
    def db(self):
        """Lazy initialization of Firestore client."""
        if self._db is None:
            self._db = get_db()
        return self._db
    
    async def get_instructions(self, org_id: str = "default") -> dict:
        """
        Get system instructions for an organization.
        
        Uses LRU cache for fast warm starts and reduced DB hits.
        Falls back to Firestore on cache miss.
        
        Args:
            org_id: Organization identifier
            
        Returns:
            dict with 'content' and 'version' keys
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cached = self._cache.get(org_id)
        if cached is not None:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"System instructions cache hit for org={org_id} | {elapsed:.2f}ms")
            return cached
        
        # Cache miss - fetch from Firestore
        try:
            doc_ref = self.db.collection(self.COLLECTION_NAME).document(org_id)
            doc = await doc_ref.get()
            
            if not doc.exists:
                # Return empty content for new orgs
                result = {"content": "", "version": None, "org_id": org_id}
            else:
                data = doc.to_dict()
                result = {
                    "content": data.get("content", ""),
                    "version": data.get("version"),
                    "org_id": org_id,
                    "updated_at": data.get("updated_at").isoformat() if data.get("updated_at") else None,
                    "updated_by": data.get("updated_by")
                }
            
            # Cache the result
            self._cache.set(org_id, result)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"System instructions fetched from Firestore for org={org_id} | {elapsed:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get instructions for org={org_id}: {e}")
            raise DatabaseError(f"Failed to fetch system instructions: {e}")
    
    async def save_instructions(self, content: str, message: str, org_id: str = "default", user_id: str = "system") -> dict:
        """
        Save system instructions for an organization.
        
        Invalidates cache after successful save to ensure consistency.
        
        Args:
            content: The system instructions content
            message: Update message/reason
            org_id: Organization identifier
            user_id: User making the update
            
        Returns:
            dict with 'success', 'version', and 'org_id' keys
        """
        start_time = time.perf_counter()
        
        try:
            doc_ref = self.db.collection(self.COLLECTION_NAME).document(org_id)
            
            # Get current version for increment
            current_doc = await doc_ref.get()
            current_version = 0
            if current_doc.exists:
                current_version = current_doc.to_dict().get("version", 0)
            
            new_version = current_version + 1
            now = datetime.now(timezone.utc)
            
            # Save to Firestore
            await doc_ref.set({
                "content": content,
                "version": new_version,
                "updated_at": now,
                "updated_by": user_id,
                "update_message": message,
                "org_id": org_id
            })
            
            # Invalidate cache for this org
            self._cache.invalidate(org_id)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"System instructions saved for org={org_id} v{new_version} | {elapsed:.2f}ms")
            
            return {
                "success": True,
                "version": new_version,
                "org_id": org_id,
                "commit": str(new_version)  # Compatibility with GitHub provider
            }
            
        except Exception as e:
            logger.error(f"Failed to save instructions for org={org_id}: {e}")
            raise DatabaseError(f"Failed to save system instructions: {e}")
    
    async def get_history(self, org_id: str = "default", limit: int = 10) -> list:
        """
        Get instruction history from backup collection.
        
        Note: History is stored in a separate collection by the metrics provider.
        This method queries that collection.
        
        Args:
            org_id: Organization identifier
            limit: Maximum number of history items
            
        Returns:
            List of historical instruction versions
        """
        try:
            # Query the system_instructions_history collection
            history_ref = self.db.collection("system_instructions_history")
            query = history_ref.where("org_id", "==", org_id).order_by(
                "backed_up_at", direction="DESCENDING"
            ).limit(limit)
            
            docs = await query.get()
            history = []
            
            for doc in docs:
                data = doc.to_dict()
                history.append({
                    "id": doc.id,
                    "content": data.get("content", ""),
                    "backed_up_at": data.get("backed_up_at").isoformat() if data.get("backed_up_at") else None,
                    "backed_up_by": data.get("backed_up_by"),
                    "org_id": data.get("org_id", org_id)
                })
            
            return history
            
        except Exception as e:
            logger.warning(f"Failed to get instruction history for org={org_id}: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return self._cache.stats()
    
    def invalidate_cache(self, org_id: Optional[str] = None) -> None:
        """
        Manually invalidate cache.
        
        Args:
            org_id: Specific org to invalidate, or None for all
        """
        if org_id:
            self._cache.invalidate(org_id)
        else:
            self._cache.invalidate_all()


# Singleton instance
firestore_config_provider = FirestoreConfigProvider()
