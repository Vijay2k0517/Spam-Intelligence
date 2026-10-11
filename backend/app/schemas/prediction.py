"""Pydantic schemas for single and batch predictions."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MessageInput(BaseModel):
    """Input payload for single message prediction."""

    message: str = Field(..., description="Raw SMS text to be classified.")
    model_name: str = Field(
        default="hybrid_svm",
        description="Model to use (text_nb, text_svm, metadata_nb, metadata_svm, hybrid_nb, hybrid_svm).",
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include extracted metadata features in response.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Message must be a string.")
        if not v.strip():
            raise ValueError("Message cannot be empty or pure whitespace.")
        return v


class BatchMessageInput(BaseModel):
    """Input payload for batch message prediction."""

    messages: List[str] = Field(..., min_length=1, description="List of raw SMS texts.")
    model_name: str = Field(
        default="hybrid_svm",
        description="Model to use (text_nb, text_svm, metadata_nb, metadata_svm, hybrid_nb, hybrid_svm).",
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include extracted metadata features in response.",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Messages list cannot be empty.")
        for idx, msg in enumerate(v):
            if not isinstance(msg, str) or not msg.strip():
                raise ValueError(f"Message at index {idx} cannot be empty or non-string.")
        return v


class PredictionOutput(BaseModel):
    """Output prediction result for an SMS message."""

    prediction: str = Field(..., description="Canonical predicted class: 'ham' or 'spam'.")
    is_spam: bool = Field(..., description="True if predicted as spam, False if ham.")
    model_name: str = Field(..., description="Identifier of the model used.")
    experiment_id: str = Field(..., description="Experiment ID (E1 - E6).")
    decision_score: Optional[float] = Field(
        default=None,
        description="SVM signed margin distance (uncalibrated score, not probability).",
    )
    probability: Optional[float] = Field(
        default=None,
        description="Calibrated probability of spam class (for Naive Bayes models).",
    )
    inference_latency_ms: float = Field(..., description="Elapsed prediction time in milliseconds.")
    metadata_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Extracted metadata feature dictionary if requested.",
    )


class BatchPredictionOutput(BaseModel):
    """Output result for batch message prediction."""

    predictions: List[PredictionOutput] = Field(..., description="List of prediction results.")
    total_messages: int = Field(..., description="Number of messages processed.")
    total_latency_ms: float = Field(..., description="Total elapsed prediction time in milliseconds.")
    model_name: str = Field(..., description="Identifier of the model used.")


# ==============================================================================
# Phase 2 FastAPI Endpoints Request & Response Schemas
# ==============================================================================

class ClassifyRequest(BaseModel):
    """Request payload for single-message classification."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw SMS message string to classify.",
        examples=["URGENT: Call 08452810075 now to claim your £1000 cash reward!"],
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional registered model ID (defaults to application default 'hybrid_mnb').",
        examples=["hybrid_mnb"],
    )
    save_history: bool = Field(
        default=True,
        description="Whether to persist successful prediction in SQLite history.",
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include extracted handcrafted metadata in response.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Message cannot be empty or pure whitespace.")
        return v


class ClassifyResponse(BaseModel):
    """Response payload for single-message classification."""

    prediction: str = Field(..., description="Predicted class: 'spam' or 'ham'.")
    is_spam: bool = Field(..., description="Boolean flag: True if spam, False if ham.")
    model_id: str = Field(..., description="Model identifier used for inference.")
    inference_ms: float = Field(..., description="High-resolution measured inference latency in milliseconds.")
    score: Optional[float] = Field(
        default=None,
        description="Estimator score: probability for Naïve Bayes, margin distance for SVM.",
    )
    score_type: Optional[str] = Field(
        default=None,
        description="Type of score: 'probability' (calibrated [0,1]) or 'decision_score' (uncalibrated distance).",
    )
    probability: Optional[float] = Field(
        default=None,
        description="Calibrated probability if provided by estimator (e.g. Naïve Bayes).",
    )
    history_id: Optional[int] = Field(
        default=None,
        description="Primary key in SQLite prediction history if saved, null otherwise.",
    )
    metadata_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Handcrafted metadata features if requested.",
    )


class BatchClassifyRequest(BaseModel):
    """Request payload for batch message classification."""

    messages: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of raw SMS texts to classify (1 to 100 items).",
        examples=[[
            "Are we still having lunch at 12?",
            "WINNER! Free tickets to the concert. Text CLAIM to 80085."
        ]],
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional registered model identifier (defaults to 'hybrid_mnb').",
    )
    save_history: bool = Field(
        default=True,
        description="Whether to persist successful predictions in SQLite history.",
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include extracted handcrafted metadata in response.",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Messages list cannot be empty.")
        if len(v) > 100:
            raise ValueError("Batch size exceeds maximum limit of 100 messages.")
        for idx, msg in enumerate(v):
            if not isinstance(msg, str) or not msg.strip():
                raise ValueError(f"Message at index {idx} cannot be empty or pure whitespace.")
            if len(msg) > 10000:
                raise ValueError(f"Message at index {idx} exceeds maximum length of 10000 characters.")
        return v


class BatchItemResult(BaseModel):
    """Prediction result for a single item in a batch request."""

    index: int = Field(..., description="Zero-based index matching input order.")
    prediction: str = Field(..., description="Predicted class: 'spam' or 'ham'.")
    is_spam: bool = Field(..., description="True if spam, False if ham.")
    inference_ms: float = Field(..., description="Measured per-message inference latency in milliseconds.")
    score: Optional[float] = Field(default=None, description="Estimator score.")
    score_type: Optional[str] = Field(default=None, description="Score type: 'probability' or 'decision_score'.")
    probability: Optional[float] = Field(default=None, description="Probability if supported.")
    history_id: Optional[int] = Field(default=None, description="Database history ID if saved.")
    metadata_features: Optional[Dict[str, float]] = Field(default=None, description="Metadata dictionary if requested.")


class BatchClassifyResponse(BaseModel):
    """Response payload for batch message classification."""

    predictions: List[BatchItemResult] = Field(..., description="Ordered list of prediction results.")
    total_messages: int = Field(..., description="Total count of messages processed.")
    total_latency_ms: float = Field(..., description="Total elapsed batch processing time in milliseconds.")
    model_id: str = Field(..., description="Model identifier used for inference.")

