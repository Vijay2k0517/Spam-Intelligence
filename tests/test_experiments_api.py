"""Automated tests for Phase 1 experiment report retrieval endpoints."""

from fastapi import status


def test_list_experiments_success(client):
    """Verify GET /api/v1/experiments returns all 18 evaluated experiments and selection bias notice."""
    response = client.get("/api/v1/experiments")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["total_experiments"] == 18
    assert len(data["experiments"]) == 18
    assert "selection_bias_disclosure" in data
    assert "selection bias" in data["selection_bias_disclosure"].lower()


    # Check presence of key baseline and remediation models
    exp_ids = [e["experiment_id"] for e in data["experiments"]]
    assert "E1" in exp_ids
    assert "E6" in exp_ids
    assert "E5_mnb" in exp_ids
    assert "E6_scaled_minmax" in exp_ids


def test_get_experiment_detail_success(client):
    """Verify GET /api/v1/experiments/{experiment_id} returns metrics, confusion matrix, and CV results."""
    response = client.get("/api/v1/experiments/E5_mnb")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["experiment_id"] == "E5_mnb"
    assert data["model_name"] == "hybrid_mnb"
    assert data["classifier"] == "MultinomialNB"
    assert data["test_accuracy"] == 0.9884
    assert data["spam_f1"] == 0.9512

    # Verify confusion matrix counts
    cm = data["confusion_matrix"]
    assert cm["tn"] == 903
    assert cm["fp"] == 1
    assert cm["fn"] == 11
    assert cm["tp"] == 117
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 1032

    # Verify training cross-validation metrics
    cv = data["cross_validation_metrics"]
    assert cv is not None
    assert cv["mean_cv_accuracy"] == 0.9881
    assert cv["mean_cv_f1"] == 0.9501


def test_get_experiment_unknown_returns_404(client):
    """Verify requesting an unknown experiment ID returns 404."""
    response = client.get("/api/v1/experiments/E99_nonexistent")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found in benchmark reports" in response.json()["message"]
