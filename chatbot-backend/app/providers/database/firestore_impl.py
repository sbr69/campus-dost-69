"""
Firestore Database Provider implementation.

Uses Google Cloud Firestore for vector similarity search.
Includes retry logic for transient failures.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import List, Optional

from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from google.oauth2 import service_account

from ...config import settings, get_logger
from ...exceptions import DatabaseError
from ...utils import is_retryable_error
from .interface import DatabaseProviderInterface, VectorSearchResult

logger = get_logger("database.firestore")


class FirestoreDatabaseProvider(DatabaseProviderInterface):
    """
    Firestore database provider for vector similarity search.
    
    Supports async operations, configurable collection/field names,
    and retry logic for transient failures.
    """
    
    def __init__(self):
        self._client: Optional[firestore.AsyncClient] = None
        self._initialized = False
        self._credentials = None
    
    async def initialize(self) -> bool:
        """Initialize Firestore connection with retry logic."""
        if self._initialized and self._client:
            return True
        
        last_error: Exception | None = None
        
        for attempt in range(settings.MAX_RETRIES + 1):
            try:
                if settings.FIREBASE_CREDS_BASE64:
                    # Decode base64 credentials
                    padded = settings.FIREBASE_CREDS_BASE64 + "=" * (
                        (4 - len(settings.FIREBASE_CREDS_BASE64) % 4) % 4
                    )
                    cred_json = base64.b64decode(padded).decode("utf-8")
                    info = json.loads(cred_json)
                    self._credentials = service_account.Credentials.from_service_account_info(info)
                    self._client = firestore.AsyncClient(credentials=self._credentials)
                else:
                    # Use default credentials (ADC)
                    self._client = firestore.AsyncClient()
                
                self._initialized = True
                logger.info("Firestore AsyncClient initialized successfully")
                return True
                
            except Exception as e:
                last_error = e
                if is_retryable_error(e) and attempt < settings.MAX_RETRIES:
                    delay = min(
                        settings.RETRY_BASE_DELAY * (2 ** attempt),
                        settings.RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "Firestore initialization failed (attempt %d/%d): %s, retrying in %.2fs",
                        attempt + 1,
                        settings.MAX_RETRIES + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Firestore initialization failed: %s", e)
                    return False
        
        logger.error("Firestore initialization failed after all retries: %s", last_error)
        return False
    
    async def search_similar(
        self,
        embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[VectorSearchResult]:
        """Search for similar documents using Firestore vector search with retry logic."""
        if not self._client:
            logger.warning("Firestore client not initialized")
            return []
        
        last_error: Exception | None = None
        
        for attempt in range(settings.MAX_RETRIES + 1):
            try:
                vector = Vector(embedding)
                coll_ref = self._client.collection(settings.FIRESTORE_VECTOR_COLLECTION)
                
                query_task = coll_ref.find_nearest(
                    vector_field=settings.FIRESTORE_VECTOR_FIELD,
                    query_vector=vector,
                    distance_measure=DistanceMeasure.COSINE,
                    limit=top_k,
                    distance_result_field="distance",
                ).get()
                
                results = await asyncio.wait_for(
                    query_task,
                    timeout=settings.FIRESTORE_QUERY_TIMEOUT_SECONDS
                )
                
                search_results: List[VectorSearchResult] = []
                for doc in results:
                    doc_dict = doc.to_dict()
                    text = doc_dict.get("text", "")
                    distance = doc_dict.get("distance", 1.0)
                    
                    # Convert cosine distance to similarity: similarity = 1 - distance
                    similarity = max(0.0, 1.0 - distance)
                    
                    if text and similarity >= similarity_threshold:
                        # Extract metadata (excluding internal fields)
                        metadata = {
                            k: v for k, v in doc_dict.items()
                            if k not in ("text", "embedding", "distance", settings.FIRESTORE_VECTOR_FIELD)
                        }
                        
                        search_results.append(VectorSearchResult(
                            text=text,
                            score=similarity,
                            metadata=metadata if metadata else None,
                        ))
                
                # Sort by score (descending)
                search_results.sort(key=lambda x: x.score, reverse=True)
                return search_results
                
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    "Vector store query timed out (attempt %d/%d)",
                    attempt + 1,
                    settings.MAX_RETRIES + 1,
                )
                if attempt >= settings.MAX_RETRIES:
                    return []
                    
            except Exception as e:
                last_error = e
                if is_retryable_error(e) and attempt < settings.MAX_RETRIES:
                    delay = min(
                        settings.RETRY_BASE_DELAY * (2 ** attempt),
                        settings.RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "Vector store query failed (attempt %d/%d): %s, retrying in %.2fs",
                        attempt + 1,
                        settings.MAX_RETRIES + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Vector store query failed: %s", e)
                    return []
        
        return []
    
    async def health_check(self) -> bool:
        """Perform a health check by querying the collection."""
        if not self._client:
            return False
        
        try:
            coll_ref = self._client.collection(settings.FIRESTORE_VECTOR_COLLECTION)
            # Just check if we can access the collection (limit 1 for speed)
            await asyncio.wait_for(
                coll_ref.limit(1).get(),
                timeout=5.0,
            )
            return True
        except Exception as e:
            logger.warning("Firestore health check failed: %s", e)
            return False
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "firestore"
    
    def is_available(self) -> bool:
        """Check if Firestore is available."""
        return self._initialized and self._client is not None
    
    async def close(self) -> None:
        """Close the Firestore connection properly."""
        if self._client:
            try:
                # Close the underlying gRPC channel
                self._client._client._transport.grpc_channel.close()
                logger.info("Firestore gRPC channel closed")
            except Exception as e:
                logger.warning("Error closing Firestore gRPC channel: %s", e)
            finally:
                self._client = None
                self._initialized = False
                self._credentials = None
                logger.info("Firestore connection closed")
