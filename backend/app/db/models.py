"""SQLAlchemy database models for persistence."""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PredictionRecord(Base):
    """Database model for storing successful SMS prediction history."""

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    prediction: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    is_spam: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    message_length: Mapped[int] = mapped_column(Integer, nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    score_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    spam_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    message_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<PredictionRecord(id={self.id}, model_id='{self.model_id}', "
            f"prediction='{self.prediction}', created_at='{self.created_at}')>"
        )
