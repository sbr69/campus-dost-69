"""
Firestore Metrics Provider implementation.

Provides lightweight metrics tracking for the chatbot backend.
Writes to the same collections as the admin backend so the
admin dashboard can display chatbot usage statistics.

Features:
- Atomic increment using Firestore Increment sentinel
- Non-blocking with native async Firestore client
- Retry logic for transient failures
- Lazy initialization (doesn't block startup)
"""
from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore
from google.oauth2 import service_account

from ...config import settings, get_logger
from ...utils import is_retryable_error

logger = get_logger("providers.metrics")


class FirestoreMetricsProvider:
    """
    Firestore metrics provider for tracking API usage.
    
    This provider writes to the weekly_metrics collection, which is
    shared with the admin backend. The admin dashboard reads from
    this collection to display chatbot usage statistics.
    
    Uses native async Firestore client for non-blocking I/O.
    
    Collection: weekly_metrics (configurable via FIRESTORE_WEEKLY_METRICS_COLLECTION)
    Document ID: YYYY-MM-DD (date string)
    Document structure: { "hits": <int> }
    """
    
    def __init__(self):
        self._client: Optional[firestore.AsyncClient] = None
        self._initialized = False
        self._credentials = None
    
    async def _ensure_initialized(self) -> bool:
        """
        Ensure Firestore client is initialized.
        
        Lazily initializes the client on first use to avoid blocking
        during application startup.
        
        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized and self._client:
            return True
        
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
            logger.debug("Metrics Firestore client initialized")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize metrics Firestore client: %s", e)
            return False
    
    async def increment_daily_hit(self, date_str: str | None = None) -> None:
        """
        Increment the hit counter for a specific date (non-blocking).
        
        Uses Firestore's atomic Increment sentinel for thread-safe counting.
        This is fully async and non-blocking - safe to call from anywhere.
        
        Args:
            date_str: Date string in YYYY-MM-DD format (defaults to today UTC)
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if not await self._ensure_initialized():
            logger.warning("Metrics provider not initialized, skipping increment")
            return
        
        for attempt in range(settings.MAX_RETRIES + 1):
            try:
                # Use Firestore's atomic Increment sentinel
                # This is non-blocking with AsyncClient
                doc_ref = self._client.collection(
                    settings.FIRESTORE_WEEKLY_METRICS_COLLECTION
                ).document(date_str)
                
                await doc_ref.set(
                    {"hits": firestore.Increment(1)},
                    merge=True
                )
                
                logger.debug("Incremented daily hit for %s", date_str)
                return
                
            except Exception as e:
                if is_retryable_error(e) and attempt < settings.MAX_RETRIES:
                    delay = min(
                        settings.RETRY_BASE_DELAY * (2 ** attempt),
                        settings.RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "Metrics increment failed (attempt %d/%d): %s, retrying in %.2fs",
                        attempt + 1,
                        settings.MAX_RETRIES + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    # Log and swallow - metrics should never break the main flow
                    logger.error("Failed to increment daily hit (giving up): %s", e)
                    return
    
    async def close(self) -> None:
        """Close the Firestore client."""
        if self._client:
            self._client.close()
            self._initialized = False
            logger.debug("Metrics Firestore client closed")
