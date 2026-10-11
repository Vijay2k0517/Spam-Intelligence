"""Endpoints for retrieving Phase 1 experimental records and metrics."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_experiment_service
from app.schemas.experiment import ExperimentDetail, ExperimentListResponse
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get(
    "",
    response_model=ExperimentListResponse,
    summary="List Phase 1 Experiments",
    description="Returns recorded experimental comparison results for all 18 evaluated baseline and remediation models.",
)
def list_experiments(
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentListResponse:
    """List all evaluated experiment configurations."""
    experiments = service.list_experiments()
    return ExperimentListResponse(
        experiments=experiments,
        total_experiments=len(experiments),
        selection_bias_disclosure=service.get_selection_bias_disclosure(),
    )


@router.get(
    "/{experiment_id}",
    response_model=ExperimentDetail,
    summary="Get Experiment Details",
    description="Returns detailed metrics, confusion matrix, ROC-AUC, and CV ranking for an experiment configuration.",
    responses={
        200: {"description": "Experiment details retrieved."},
        404: {"description": "Experiment identifier not found."},
    },
)
def get_experiment(
    experiment_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentDetail:
    """Retrieve full detail for a specific experiment identifier."""
    detail = service.get_experiment_detail(experiment_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found in benchmark reports.",
        )
    return detail
