"""Pydantic schemas for prediction history retrieval and pagination."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.common import PageMeta


class PredictionHistoryItem(BaseModel):
    """A single prediction record retrieved from database history."""

    id: int = Field(..., description="Unique record identifier.")
    created_at: datetime = Field(..., description="UTC creation timestamp.")
    model_id: str = Field(..., description="Model identifier used for prediction.")
    prediction: str = Field(..., description="Class prediction: 'spam' or 'ham'.")
    is_spam: bool = Field(..., description="True if spam, False if ham.")
    message_length: int = Field(..., description="Character count of evaluated message.")
    inference_ms: float = Field(..., description="Measured inference latency in milliseconds.")
    score_type: Optional[str] = Field(default=None, description="Score type ('probability' or 'decision_score').")
    spam_score: Optional[float] = Field(default=None, description="Spam score value.")
    message_text: Optional[str] = Field(
        default=None,
        description="Raw message text if persistence was enabled; null otherwise.",
    )


class PaginatedHistoryResponse(BaseModel):
    """Response envelope for paginated prediction history queries."""

    items: List[PredictionHistoryItem] = Field(..., description="List of prediction history records.")
    pagination: PageMeta = Field(..., description="Pagination metadata.")
