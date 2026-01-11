"""
Metrics providers for tracking API usage.

This module provides metrics tracking functionality that writes to
the same Firestore collections as the admin backend, allowing the
admin dashboard to display chatbot usage statistics.
"""
from __future__ import annotations

from .firestore_impl import FirestoreMetricsProvider

# Singleton instance
metrics_provider = FirestoreMetricsProvider()

__all__ = ["metrics_provider", "FirestoreMetricsProvider"]
