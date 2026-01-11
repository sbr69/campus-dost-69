from abc import ABC, abstractmethod
from typing import List

class ActivityProviderInterface(ABC):
    """
    Abstract interface for activity logging providers.
    
    MULTI-TENANCY: All methods require org_id as first parameter
    for data isolation between organizations.
    """
    
    @abstractmethod
    async def log_activity(self, org_id: str, action: str, actor: str, resource_type: str = None,
                          resource_id: str = None, meta: dict = None) -> str:
        """Log an activity event for an organization."""
        pass
    
    @abstractmethod
    async def get_activity_log(self, org_id: str, limit: int) -> List[dict]:
        """Get recent activity logs for an organization."""
        pass
