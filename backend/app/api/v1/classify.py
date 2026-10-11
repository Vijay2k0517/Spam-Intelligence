"""Endpoints for single and batch SMS spam classification."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_classification_service
from app.core.logging import get_logger
from app.schemas.prediction import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
)
from app.services.classification_service import ClassificationService

logger = get_logger(__name__)

router = APIRouter(prefix="/classify", tags=["Classification"])


@router.post(
    "",
    response_model=ClassifyResponse,
    summary="Classify a Single SMS Message",
    description=(
        "Evaluates a single raw SMS message string using a specified or default classifier model. "
        "Measures high-resolution prediction latency and optionally records genuine successful predictions "
        "in SQLite history."
    ),
    responses={
        200: {"description": "Message classified successfully."},
        400: {"description": "Invalid input message or configuration."},
        404: {"description": "Requested model identifier not recognized in registry."},
        503: {"description": "Model artifact is not available on disk."},
    },
)
def classify_single_message(
    request: ClassifyRequest,
    service: ClassificationService = Depends(get_classification_service),
) -> ClassifyResponse:
    """Classify a single SMS text message."""
    try:
        return service.classify_single(request)
    except ValueError as e:
        logger.warning("Classification validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except FileNotFoundError as e:
        logger.error("Requested model artifact missing: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Selected model artifact is currently unavailable: {e}",
        )
    except Exception as e:
        logger.error("Unexpected classification failure: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred during message classification.",
        )


@router.post(
    "/batch",
    response_model=BatchClassifyResponse,
    summary="Classify a Batch of SMS Messages",
    description=(
        "Evaluates an ordered list of 1 to 100 SMS messages using the selected model. "
        "Preserves input ordering, measures per-message and total batch latency, and optionally "
        "persists successful history records."
    ),
    responses={
        200: {"description": "Batch processed successfully."},
        400: {"description": "Invalid batch payload or individual message validation failure."},
        404: {"description": "Requested model identifier not found."},
        503: {"description": "Selected model artifact is unavailable."},
    },
)
def classify_batch_messages(
    request: BatchClassifyRequest,
    service: ClassificationService = Depends(get_classification_service),
) -> BatchClassifyResponse:
    """Classify multiple SMS messages in batch."""
    try:
        return service.classify_batch(request)
    except ValueError as e:
        logger.warning("Batch classification validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except FileNotFoundError as e:
        logger.error("Batch classification artifact missing: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Selected model artifact is currently unavailable: {e}",
        )
    except Exception as e:
        logger.error("Unexpected batch classification error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred during batch classification.",
        )
