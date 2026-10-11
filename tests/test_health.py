"""Automated tests for health and readiness endpoints."""

from fastapi import status
from sqlalchemy import text


def test_root_endpoint(client):
    """Verify root endpoint returns API metadata and documentation links."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "docs_url" in data
    assert data["docs_url"] == "/docs"


def test_health_liveness_endpoint(client):
    """Verify health endpoint returns healthy status and API version."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["api_version"] == "v1"
    assert "timestamp" in data


def test_health_readiness_endpoint(client):
    """Verify readiness endpoint validates SQLite and default model availability."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["database"] == "connected"
    assert data["checks"]["default_model"] == "hybrid_mnb"
    assert data["checks"]["model_status"] == "ready"


def test_health_readiness_database_failure(client, monkeypatch):
    """Verify readiness probe returns 503 if database check fails."""
    def fail_execute(*args, **kwargs):
        raise RuntimeError("Simulated SQLite connection failure")

    from app.db.session import SessionLocal
    monkeypatch.setattr("sqlalchemy.orm.Session.execute", fail_execute)

    response = client.get("/api/v1/health/ready")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"]["database"] == "disconnected"
