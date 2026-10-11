"""Repository layer for PredictionRecord persistence and queries."""

from datetime import datetime
from typing import List, Optional, Tuple
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import PredictionRecord

logger = get_logger(__name__)


class PredictionRepository:
    """Data access object for PredictionRecord persistence in SQLite."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, record: PredictionRecord) -> PredictionRecord:
        """Insert a single prediction record with transaction management."""
        try:
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
            return record
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to insert PredictionRecord: %s", type(e).__name__)
            raise

    def bulk_create(self, records: List[PredictionRecord]) -> List[PredictionRecord]:
        """Insert multiple prediction records in a single transaction."""
        if not records:
            return []
        try:
            self.db.add_all(records)
            self.db.commit()
            for r in records:
                self.db.refresh(r)
            return records
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to bulk insert PredictionRecords: %s", type(e).__name__)
            raise

    def get_by_id(self, record_id: int) -> Optional[PredictionRecord]:
        """Retrieve a single prediction record by ID."""
        stmt = select(PredictionRecord).where(PredictionRecord.id == record_id)
        return self.db.scalars(stmt).first()

    def list_records(
        self,
        page: int = 1,
        page_size: int = 20,
        model_id: Optional[str] = None,
        prediction: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[List[PredictionRecord], int]:
        """Query paginated prediction records with filtering."""
        stmt = select(PredictionRecord)
        count_stmt = select(func.count(PredictionRecord.id))

        # Apply filters
        if model_id:
            stmt = stmt.where(PredictionRecord.model_id == model_id)
            count_stmt = count_stmt.where(PredictionRecord.model_id == model_id)
        if prediction:
            stmt = stmt.where(PredictionRecord.prediction == prediction.lower())
            count_stmt = count_stmt.where(PredictionRecord.prediction == prediction.lower())
        if start_date:
            stmt = stmt.where(PredictionRecord.created_at >= start_date)
            count_stmt = count_stmt.where(PredictionRecord.created_at >= start_date)
        if end_date:
            stmt = stmt.where(PredictionRecord.created_at <= end_date)
            count_stmt = count_stmt.where(PredictionRecord.created_at <= end_date)

        total_count = self.db.scalar(count_stmt) or 0

        # Stable ordering: newest records first
        stmt = (
            stmt.order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        records = list(self.db.scalars(stmt).all())
        return records, total_count

    def delete_by_id(self, record_id: int) -> bool:
        """Delete a prediction record by ID. Returns True if deleted, False if not found."""
        record = self.get_by_id(record_id)
        if not record:
            return False
        try:
            self.db.delete(record)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to delete PredictionRecord %d: %s", record_id, type(e).__name__)
            raise
