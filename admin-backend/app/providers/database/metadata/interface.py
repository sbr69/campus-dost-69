from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

class MetadataProviderInterface(ABC):
    """
    Abstract interface for document metadata storage providers.
    
    MULTI-TENANCY: All methods require org_id as first parameter
    for data isolation between organizations.
    """
    
    @abstractmethod
    def generate_id(self) -> str:
        """Generate a unique document ID."""
        pass
    
    @abstractmethod
    async def create_document(self, org_id: str, doc_id: str, data: dict, archived: bool = False) -> str:
        """Create a document metadata record for an organization."""
        pass
    
    @abstractmethod
    async def get_document(self, org_id: str, doc_id: str, archived: bool = False) -> Optional[dict]:
        """Get document metadata by ID within an organization."""
        pass
    
    @abstractmethod
    async def list_documents(self, org_id: str, limit: int, archived: bool = False) -> List[dict]:
        """List documents within an organization with optional limit."""
        pass
    
    @abstractmethod
    async def delete_document(self, org_id: str, doc_id: str, archived: bool = False) -> bool:
        """Delete document metadata within an organization."""
        pass
    
    @abstractmethod
    async def update_document(self, org_id: str, doc_id: str, updates: dict) -> bool:
        """Update specific fields in a document within an organization."""
        pass
    
    @abstractmethod
    async def get_document_count(self, org_id: str) -> dict:
        """Get counts of active and archived documents for an organization."""
        pass
    
    @abstractmethod
    async def get_expired_archives(self, org_id: str, cutoff_date: datetime) -> List[dict]:
        """Get archived documents older than cutoff date for an organization."""
        pass
    
    @abstractmethod
    async def cleanup_old_archives(self, org_id: str, days: int) -> dict:
        """
        Cleanup archived documents older than specified days for an organization.
        
        Returns:
            dict with 'deleted_count' (int) and 'documents' (list of deleted doc info)
        """
        pass
