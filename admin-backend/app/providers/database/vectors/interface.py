from abc import ABC, abstractmethod
from typing import List, Optional

class VectorStorageInterface(ABC):
    """
    Abstract interface for vector storage providers.
    
    MULTI-TENANCY: All methods require org_id as first parameter
    for data isolation between organizations.
    """
    
    @abstractmethod
    async def store_vectors(self, org_id: str, doc_id: str, vector_data: List[dict]) -> int:
        """Store vector embeddings for a document within an organization."""
        pass
    
    @abstractmethod
    async def get_vectors(self, org_id: str, doc_id: str, archived: bool = False) -> Optional[dict]:
        """Get all vectors for a document within an organization."""
        pass
    
    @abstractmethod
    async def delete_vectors(self, org_id: str, doc_id: str, archived: bool = False) -> bool:
        """Delete all vectors for a document within an organization."""
        pass
    
    @abstractmethod
    async def archive_vectors(self, org_id: str, doc_id: str) -> int:
        """Archive vectors for a document within an organization."""
        pass
    
    @abstractmethod
    async def restore_vectors(self, org_id: str, doc_id: str) -> int:
        """Restore vectors for a document within an organization."""
        pass
