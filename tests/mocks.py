from typing import Optional
from pydantic import BaseModel

class UserContext(BaseModel):
    user_id: str
    tenant_id: Optional[str] = None
    downstream_token: Optional[str] = None
