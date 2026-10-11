"""Endpoints for discovering registered classifiers and artifact metadata."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_model_registry_service
from app.core.config import settings
from app.schemas.model import ModelDetail, ModelListResponse
from app.services.model_registry_service import ModelRegistryService

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List All Registered Models",
    description="Returns the full registry of available classifiers, feature spaces, and artifact statuses.",
)
def list_models(
    service: ModelRegistryService = Depends(get_model_registry_service),
) -> ModelListResponse:
    """Retrieve all registered models."""
    models = service.list_models()
    return ModelListResponse(
        models=models,
        total_models=len(models),
        default_model=settings.default_model,
    )


@router.get(
    "/{model_id}",
    response_model=ModelDetail,
    summary="Get Model Details",
    description="Returns detailed metadata, hyperparameters, and exploratory metrics for a specific registered model.",
    responses={
        200: {"description": "Model details retrieved successfully."},
        404: {"description": "Model identifier not found in registry."},
    },
)
def get_model(
    model_id: str,
    service: ModelRegistryService = Depends(get_model_registry_service),
) -> ModelDetail:
    """Retrieve detailed metadata for a registered model."""
    detail = service.get_model_detail(model_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model identifier '{model_id}' was not found in the approved model registry.",
        )
    return detail
