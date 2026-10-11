"""Health check and readiness endpoints."""

from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_model_registry_service
from app.core.config import settings
from app.core.logging import get_logger
from app.services.model_registry_service import ModelRegistryService

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    summary="Application Liveness Probe",
    description="Returns high-level service status, API version, and timestamp.",
    response_model=Dict[str, Any],
)
def get_health() -> Dict[str, Any]:
    """Basic liveness probe."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "api_version": "v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/ready",
    summary="Application Readiness Probe",
    description="Verifies SQLite connectivity and default model availability.",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Service is fully ready to classify messages."},
        503: {"description": "Service is live but database or default model is unavailable."},
    },
)
def get_readiness(
    db: Session = Depends(get_db),
    registry: ModelRegistryService = Depends(get_model_registry_service),
) -> JSONResponse:
    """Readiness probe checking SQLite connectivity and default model."""
    checks: Dict[str, Any] = {}
    is_ready = True

    # 1. Check SQLite connectivity
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "connected"
    except Exception as e:
        logger.error("Readiness check failed for database: %s", type(e).__name__)
        checks["database"] = "disconnected"
        is_ready = False

    # 2. Check default model artifact readiness
    default_model = settings.default_model
    try:
        is_avail = registry.is_model_available(default_model)
        if is_avail:
            # Verify loadability into memory
            registry.get_pipeline(default_model)
            checks["default_model"] = default_model
            checks["model_status"] = "ready"
        else:
            checks["default_model"] = default_model
            checks["model_status"] = "artifact_missing"
            is_ready = False
    except Exception as e:
        logger.error("Readiness check failed for model %s: %s", default_model, type(e).__name__)
        checks["default_model"] = default_model
        checks["model_status"] = f"load_error: {type(e).__name__}"
        is_ready = False

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    response_data = {
        "status": "ready" if is_ready else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
    return JSONResponse(status_code=status_code, content=response_data)
