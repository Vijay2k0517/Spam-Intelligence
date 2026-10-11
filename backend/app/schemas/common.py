"""Common Pydantic schemas for API metadata, errors, and pagination."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Specific error item description."""

    code: str = Field(..., description="Error code or error category.")
    message: str = Field(..., description="Human-readable description of error.")
    field: Optional[str] = Field(default=None, description="Request field that caused the error.")


class ErrorResponse(BaseModel):
    """Standardized error envelope returned by the API."""

    error: str = Field(..., description="High-level error classification.")
    message: str = Field(..., description="Summary message describing the failure.")
    details: Optional[List[ErrorDetail]] = Field(default=None, description="Detailed validation or operational issues.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of error.")


class PageMeta(BaseModel):
    """Pagination metadata."""

    page: int = Field(..., ge=1, description="Current page index.")
    page_size: int = Field(..., ge=1, le=100, description="Items per page.")
    total_items: int = Field(..., ge=0, description="Total number of matching items.")
    total_pages: int = Field(..., ge=0, description="Total pages available.")
