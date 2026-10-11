"""Automated tests for model discovery and metadata endpoints."""

from fastapi import status


def test_list_models_returns_all_registered_configurations(client):
    """Verify GET /api/v1/models lists all 18 registered models from manifest/registry."""
    response = client.get("/api/v1/models")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["total_models"] == 18
    assert data["default_model"] == "hybrid_mnb"
    assert len(data["models"]) == 18

    # Find hybrid_mnb and verify its attributes
    mnb_item = next(m for m in data["models"] if m["model_id"] == "hybrid_mnb")
    assert mnb_item["is_default"] is True
    assert mnb_item["is_available"] is True
    assert mnb_item["supports_probability"] is True
    assert mnb_item["supports_decision_score"] is False
    assert mnb_item["score_type"] == "probability"
    assert "5-Fold Stratified Cross-Validation" in mnb_item["selection_basis"]

    # Find hybrid_svm_scaled_minmax and verify score type
    svm_item = next(m for m in data["models"] if m["model_id"] == "hybrid_svm_scaled_minmax")
    assert svm_item["supports_probability"] is False
    assert svm_item["supports_decision_score"] is True
    assert svm_item["score_type"] == "decision_score"


def test_get_model_details_success(client):
    """Verify GET /api/v1/models/{model_id} returns detailed parameters and exploratory metrics."""
    response = client.get("/api/v1/models/hybrid_mnb")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["model_id"] == "hybrid_mnb"
    assert data["experiment_id"] == "E5_mnb"
    assert data["feature_space"] == "hybrid"
    assert data["feature_count"] == 7731
    assert data["classifier_type"] == "MultinomialNB"


def test_get_model_unknown_returns_404(client):
    """Verify unknown model ID returns 404 with structured error."""
    response = client.get("/api/v1/models/unrecognized_model_123")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "was not found in the approved model registry" in response.json()["message"]
