"""Automated tests for SQLite database layer and repository operations."""

from datetime import datetime, timezone
import pytest
from sqlalchemy import select

from app.core.config import settings
from app.db.models import PredictionRecord
from app.repositories.prediction_repository import PredictionRepository
from app.services.history_service import HistoryService


def test_prediction_repository_crud(db_session):
    """Verify PredictionRepository create, get, list, and delete operations."""
    repo = PredictionRepository(db_session)

    # 1. Create
    rec = PredictionRecord(
        model_id="hybrid_mnb",
        prediction="spam",
        is_spam=True,
        message_length=45,
        inference_ms=1.23,
        score_type="probability",
        spam_score=0.985,
        message_text=None,
    )
    created = repo.create(rec)
    assert created.id is not None
    assert created.id > 0
    assert created.created_at is not None

    # 2. Get by ID
    fetched = repo.get_by_id(created.id)
    assert fetched is not None
    assert fetched.model_id == "hybrid_mnb"
    assert fetched.is_spam is True

    # 3. List
    records, count = repo.list_records(page=1, page_size=10)
    assert count == 1
    assert len(records) == 1

    # 4. Delete
    deleted = repo.delete_by_id(created.id)
    assert deleted is True
    assert repo.get_by_id(created.id) is None


def test_prediction_repository_bulk_create(db_session):
    """Verify PredictionRepository bulk creation."""
    repo = PredictionRepository(db_session)
    recs = [
        PredictionRecord(
            model_id="hybrid_mnb",
            prediction="ham",
            is_spam=False,
            message_length=20 + i,
            inference_ms=0.95,
            score_type="probability",
            spam_score=0.01,
            message_text=None,
        )
        for i in range(5)
    ]
    created = repo.bulk_create(recs)
    assert len(created) == 5
    for r in created:
        assert r.id is not None


def test_privacy_setting_raw_message_omission(db_session, monkeypatch):
    """Verify raw message text is excluded from history when save_raw_messages is disabled."""
    monkeypatch.setattr(settings, "save_raw_messages", False)
    repo = PredictionRepository(db_session)

    # Insert a record with text directly
    rec = PredictionRecord(
        model_id="hybrid_mnb",
        prediction="spam",
        is_spam=True,
        message_length=30,
        inference_ms=1.1,
        score_type="probability",
        spam_score=0.99,
        message_text="Confidential message text here",
    )
    repo.create(rec)

    # When accessed through HistoryService, message_text must be masked to None
    history_service = HistoryService(db_session)
    item = history_service.get_record(rec.id)
    assert item is not None
    assert item.message_text is None
