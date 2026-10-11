"""Pydantic schemas for model discovery and metadata inspection."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    """Summary information for a registered classifier model."""

    model_id: str = Field(..., description="Unique model identifier matching registry.")
    display_name: str = Field(..., description="Human-readable model name.")
    classifier_type: str = Field(..., description="Classifier architecture (BernoulliNB, MultinomialNB, SVC, etc.).")
    feature_space: str = Field(..., description="Feature space representation (textual, metadata, hybrid).")
    feature_count: int = Field(..., description="Total input feature dimension.")
    is_default: bool = Field(..., description="True if configured as the application default model.")
    is_available: bool = Field(..., description="True if model artifact exists and can be loaded.")
    supports_probability: bool = Field(..., description="True if model outputs class probabilities.")
    supports_decision_score: bool = Field(..., description="True if model outputs signed decision margin.")
    score_type: Optional[str] = Field(default=None, description="Score type: 'probability' or 'decision_score'.")
    selection_basis: str = Field(..., description="Basis of selection or role (e.g. CV-selected top candidate, baseline, etc.).")


class ModelDetail(ModelSummary):
    """Detailed model metadata and serialized artifact attributes."""

    experiment_id: str = Field(..., description="Associated experiment identifier (E1 - E6, etc.).")
    scaling: str = Field(..., description="Feature scaling configuration applied to continuous features.")
    feature_mode: str = Field(..., description="Schema mode: 'paper_core' (19) or 'paper_extended' (22).")
    preprocessing_details: Optional[str] = Field(default=None, description="Text preprocessing configuration.")
    model_parameters: Optional[str] = Field(default=None, description="Estimator hyperparameter settings.")
    artifact_file: Optional[str] = Field(default=None, description="Filename of serialized model binary.")
    test_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Exploratory test metrics if available.")


class ModelListResponse(BaseModel):
    """Response envelope for listing registered models."""

    models: List[ModelSummary] = Field(..., description="List of registered model summaries.")
    total_models: int = Field(..., description="Total count of registered models.")
    default_model: str = Field(..., description="Identifier of the configured default model.")
