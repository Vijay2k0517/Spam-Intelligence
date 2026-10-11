"""Automated tests for batch message classification endpoint."""

from fastapi import status


def test_classify_batch_success(client):
    """Verify batch classification processes multiple messages preserving input order."""
    messages = [
        "Hey, are we having lunch today at 12?",
        "URGENT: Claim your free £1000 gift! Call 08452810075 now.",
        "Can you send me the lecture notes from yesterday?",
    ]
    payload = {
        "messages": messages,
        "model_id": "hybrid_mnb",
        "save_history": True,
    }
    response = client.post("/api/v1/classify/batch", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["total_messages"] == 3
    assert data["total_latency_ms"] > 0.0
    assert data["model_id"] == "hybrid_mnb"
    assert len(data["predictions"]) == 3

    # Check input order preservation and individual measurements
    for idx, item in enumerate(data["predictions"]):
        assert item["index"] == idx
        assert item["prediction"] in ("spam", "ham")
        assert item["inference_ms"] > 0.0
        assert item["score_type"] == "probability"
        assert item["history_id"] is not None


def test_classify_batch_empty_list_rejected(client):
    """Verify empty messages list is rejected with 422."""
    response = client.post("/api/v1/classify/batch", json={"messages": []})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_classify_batch_exceeds_maximum_size_rejected(client):
    """Verify batch with > 100 messages is rejected with 422."""
    messages = ["Message test"] * 101
    response = client.post("/api/v1/classify/batch", json={"messages": messages})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_classify_batch_with_invalid_inner_item_rejected(client):
    """Verify batch containing an empty whitespace string is rejected with 422."""
    messages = ["Valid message 1", "   ", "Valid message 3"]
    response = client.post("/api/v1/classify/batch", json={"messages": messages})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_classify_batch_oversized_item_rejected(client):
    """Verify batch containing an item > 10,000 characters is rejected with 422."""
    messages = ["Valid message", "B" * 10001]
    response = client.post("/api/v1/classify/batch", json={"messages": messages})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_classify_batch_without_history(client, db_session):
    """Verify batch classification with save_history=False does not create records."""
    from app.db.models import PredictionRecord
    from sqlalchemy import func, select

    initial_count = db_session.scalar(select(func.count(PredictionRecord.id)))

    payload = {
        "messages": ["Sample message A", "Sample message B"],
        "save_history": False,
    }
    response = client.post("/api/v1/classify/batch", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    for item in data["predictions"]:
        assert item["history_id"] is None

    final_count = db_session.scalar(select(func.count(PredictionRecord.id)))
    assert final_count == initial_count
