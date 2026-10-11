"""Endpoints for querying and managing SQLite prediction history."""

from datetime import datetime
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_history_service
from app.schemas.history import PaginatedHistoryResponse, PredictionHistoryItem
from app.services.history_service import HistoryService

router = APIRouter(prefix="/history", tags=["History"])


@router.get(
    "",
    response_model=PaginatedHistoryResponse,
    summary="Query Prediction History",
    description="Returns paginated prediction history records with optional filters for model, class, and date range.",
)
def get_history(
    page: int = Query(1, ge=1, description="Page number starting at 1."),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)."),
    model_id: Optional[str] = Query(None, description="Filter by registered model identifier."),
    prediction: Optional[str] = Query(None, description="Filter by predicted class ('spam' or 'ham')."),
    start_date: Optional[datetime] = Query(None, description="Filter records created on or after this UTC timestamp."),
    end_date: Optional[datetime] = Query(None, description="Filter records created on or before this UTC timestamp."),
    service: HistoryService = Depends(get_history_service),
) -> PaginatedHistoryResponse:
    """Retrieve filtered, paginated prediction records."""
    if prediction and prediction.lower() not in ("spam", "ham"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prediction filter must be either 'spam' or 'ham'.",
        )
    return service.get_history(
        page=page,
        page_size=page_size,
        model_id=model_id,
        prediction=prediction,
        start_date=start_date,
        end_date=end_date,
    )


@router.get(
    "/{record_id}",
    response_model=PredictionHistoryItem,
    summary="Get Single History Record",
    description="Retrieves a specific prediction history record by ID.",
    responses={
        200: {"description": "History record found."},
        404: {"description": "History record not found."},
    },
)
def get_history_record(
    record_id: int,
    service: HistoryService = Depends(get_history_service),
) -> PredictionHistoryItem:
    """Retrieve single prediction record by ID."""
    record = service.get_record(record_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record #{record_id} was not found.",
        )
    return record


@router.delete(
    "/{record_id}",
    summary="Delete a History Record",
    description="Deletes a specific prediction history record from the SQLite database.",
    responses={
        200: {"description": "Record deleted successfully."},
        404: {"description": "Record not found."},
    },
)
def delete_history_record(
    record_id: int,
    service: HistoryService = Depends(get_history_service),
) -> Dict[str, Any]:
    """Delete a prediction record by ID."""
    deleted = service.delete_record(record_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record #{record_id} does not exist.",
        )
    return {"deleted": True, "record_id": record_id}
