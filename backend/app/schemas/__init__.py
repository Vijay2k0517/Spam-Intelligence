"""Pydantic schemas package."""

from app.schemas.common import ErrorDetail, ErrorResponse, PageMeta
from app.schemas.experiment import (
    ConfusionMatrixSchema,
    ExperimentDetail,
    ExperimentListResponse,
    ExperimentSummary,
)
from app.schemas.history import PaginatedHistoryResponse, PredictionHistoryItem
from app.schemas.model import ModelDetail, ModelListResponse, ModelSummary
from app.schemas.prediction import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    BatchItemResult,
    BatchMessageInput,
    BatchPredictionOutput,
    ClassifyRequest,
    ClassifyResponse,
    MessageInput,
    PredictionOutput,
)

__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "PageMeta",
    "ConfusionMatrixSchema",
    "ExperimentDetail",
    "ExperimentListResponse",
    "ExperimentSummary",
    "PaginatedHistoryResponse",
    "PredictionHistoryItem",
    "ModelDetail",
    "ModelListResponse",
    "ModelSummary",
    "BatchClassifyRequest",
    "BatchClassifyResponse",
    "BatchItemResult",
    "BatchMessageInput",
    "BatchPredictionOutput",
    "ClassifyRequest",
    "ClassifyResponse",
    "MessageInput",
    "PredictionOutput",
]
