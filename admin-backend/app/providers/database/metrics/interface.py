from abc import ABC, abstractmethod
from typing import List

class MetricsProviderInterface(ABC):
    """
    Abstract interface for metrics storage providers.
    
    MULTI-TENANCY: All methods require org_id as first parameter
    for data isolation between organizations.
    """
    
    @abstractmethod
    async def get_metrics(self, org_id: str) -> dict:
        """Get dashboard metrics for an organization."""
        pass
    
    @abstractmethod
    async def update_metrics(self, org_id: str, updates: dict = None) -> bool:
        """Update dashboard metrics for an organization."""
        pass
    
    @abstractmethod
    async def increment_daily_hit(self, org_id: str, date_str: str = None) -> None:
        """Increment hit counter for a specific date for an organization."""
        pass
    
    @abstractmethod
    async def get_weekly_metrics(self, org_id: str, days: int = 7) -> List[dict]:
        """Get weekly metrics for the last N days for an organization."""
        pass
    
    @abstractmethod
    async def calculate_total_size(self, org_id: str) -> int:
        """Calculate total size of all documents for an organization."""
        pass
    
    @abstractmethod
    async def update_total_size(self, org_id: str, size_delta: int = None) -> None:
        """Update total size in metrics for an organization."""
        pass
    
    @abstractmethod
    async def backup_system_instructions(self, org_id: str, content: str, backed_up_by: str) -> str:
        """Backup system instructions to history for an organization."""
        pass
    
    @abstractmethod
    async def get_system_instructions_history(self, org_id: str, limit: int = 50) -> List[dict]:
        """Get system instructions history for an organization."""
        pass
