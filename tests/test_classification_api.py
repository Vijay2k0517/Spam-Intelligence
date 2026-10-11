"""Automated tests for single-message classification endpoint."""

from fastapi import status


def test_classify_single_default_model(client):
    """Verify single message prediction using default model (hybrid_mnb)."""
    payload = {
        "message": "URGENT! Call 08452810075 now to claim your £1000 cash reward!",
        "save_history": True,
    }
    response = client.post("/api/v1/classify", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_id"] == "hybrid_mnb"
    assert data["prediction"] in ("spam", "ham")
    assert isinstance(data["is_spam"], bool)
    assert data["inference_ms"] > 0.0
    assert data["score_type"] == "probability"
    assert isinstance(data["score"], float)
    assert data["probability"] == data["score"]
    assert data["history_id"] is not None
    assert isinstance(data["history_id"], int)


def test_classify_single_svm_decision_score_integrity(client):
    """Verify SVM estimator returns decision_score without misrepresenting it as probability."""
    payload = {
        "message": "Are we still meeting for lunch at 12?",
        "model_id": "hybrid_svm_scaled_minmax",
        "save_history": True,
        "include_metadata": True,
    }
    response = client.post("/api/v1/classify", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_id"] == "hybrid_svm_scaled_minmax"
    assert data["score_type"] == "decision_score"
    assert isinstance(data["score"], float)
    # Never claim SVM decision margin as probability
    assert data["probability"] is None
    assert data["metadata_features"] is not None
    assert "length" in data["metadata_features"]
    assert "has_mobile_number" in data["metadata_features"]


def test_classify_single_without_history(client, db_session):
    """Verify prediction when save_history is False leaves database untouched and returns null history_id."""
    from app.db.models import PredictionRecord
    from sqlalchemy import func, select

    initial_count = db_session.scalar(select(func.count(PredictionRecord.id)))

    payload = {
        "message": "Hello friend, see you at the library.",
        "save_history": False,
    }
    response = client.post("/api/v1/classify", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["history_id"] is None

    final_count = db_session.scalar(select(func.count(PredictionRecord.id)))
    assert final_count == initial_count


def test_classify_unknown_model_rejected(client):
    """Verify requesting an unrecognized model identifier returns 400 Bad Request."""
    payload = {
        "message": "Normal message",
        "model_id": "nonexistent_custom_model_xyz",
    }
    response = client.post("/api/v1/classify", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "Unrecognized model identifier" in data["message"]


def test_classify_empty_message_rejected(client):
    """Verify empty or whitespace message is rejected with 422."""
    for empty_msg in ["", "   ", "\t\n"]:
        response = client.post("/api/v1/classify", json={"message": empty_msg})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_classify_oversized_message_rejected(client):
    """Verify messages exceeding 10,000 characters are rejected with 422."""
    oversized = "A" * 10001
    response = client.post("/api/v1/classify", json={"message": oversized})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_classify_missing_artifact_returns_503(client, monkeypatch):
    """Verify that a model with missing artifact file fails with 503 instead of faking predictions."""
    from app.services.model_registry_service import ModelRegistryService

    def mock_get_pipeline(self, model_id):
        raise FileNotFoundError(f"Mock missing artifact for {model_id}")

    monkeypatch.setattr(ModelRegistryService, "get_pipeline", mock_get_pipeline)

    payload = {
        "message": "Test message",
        "model_id": "hybrid_mnb",
    }
    response = client.post("/api/v1/classify", json=payload)
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "currently unavailable" in response.json()["message"]
