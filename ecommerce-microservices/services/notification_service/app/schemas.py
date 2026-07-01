from datetime import datetime

from pydantic import BaseModel


class NotificationResponse(BaseModel):
    id: int
    user_id: int
    order_id: int
    recipient_email: str
    event_type: str
    channel: str
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}
