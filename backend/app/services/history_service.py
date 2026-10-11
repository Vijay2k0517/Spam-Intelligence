"""Service layer for querying and managing prediction history records."""

from datetime import datetime
import math
from typing import Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import PredictionRecord
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.common import PageMeta
from app.schemas.history import PaginatedHistoryResponse, PredictionHistoryItem

logger = get_logger(__name__)


class HistoryService:
    """Coordinates prediction history filtering, pagination, and deletion."""

    def __init__(self, db: Session):
        self.repository = PredictionRepository(db)

    def _to_schema(self, record: PredictionRecord) -> PredictionHistoryItem:
        """Convert SQLAlchemy entity to Pydantic schema, applying privacy masking if configured."""
        text_val = record.message_text if settings.save_raw_messages else None
        return PredictionHistoryItem(
            id=record.id,
            created_at=record.created_at,
            model_id=record.model_id,
            prediction=record.prediction,
            is_spam=record.is_spam,
            message_length=record.message_length,
            inference_ms=round(record.inference_ms, 3),
            score_type=record.score_type,
            spam_score=round(record.spam_score, 4) if record.spam_score is not None else None,
            message_text=text_val,
        )

    def get_history(
        self,
        page: int = 1,
        page_size: int = 20,
        model_id: Optional[str] = None,
        prediction: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PaginatedHistoryResponse:
        """Query paginated prediction history records."""
        safe_page = max(1, page)
        safe_page_size = min(max(1, page_size), 100)

        records, total_count = self.repository.list_records(
            page=safe_page,
            page_size=safe_page_size,
            model_id=model_id,
            prediction=prediction,
            start_date=start_date,
            end_date=end_date,
        )

        total_pages = math.ceil(total_count / safe_page_size) if total_count > 0 else 0

        items = [self._to_schema(r) for r in records]
        return PaginatedHistoryResponse(
            items=items,
            pagination=PageMeta(
                page=safe_page,
                page_size=safe_page_size,
                total_items=total_count,
                total_pages=total_pages,
            ),
        )

    def get_record(self, record_id: int) -> Optional[PredictionHistoryItem]:
        """Fetch a specific history record by ID."""
        record = self.repository.get_by_id(record_id)
        if not record:
            return None
        return self._to_schema(record)

    def delete_record(self, record_id: int) -> bool:
        """Delete a history record by ID."""
        return self.repository.delete_by_id(record_id)
