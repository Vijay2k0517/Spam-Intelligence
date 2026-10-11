"""Pydantic schemas for Phase 1 experiment report retrieval."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConfusionMatrixSchema(BaseModel):
    """Contingency table counts for binary evaluation."""

    tn: int = Field(..., description="True Negatives (Ham correctly classified).")
    fp: int = Field(..., description="False Positives (Ham incorrectly classified as Spam).")
    fn: int = Field(..., description="False Negatives (Spam missed as Ham).")
    tp: int = Field(..., description="True Positives (Spam correctly caught).")


class ExperimentSummary(BaseModel):
    """Summary of an evaluated experiment configuration."""

    experiment_id: str = Field(..., description="Experiment identifier (E1 - E6, etc.).")
    model_name: str = Field(..., description="Internal model name.")
    feature_space: str = Field(..., description="Feature space (textual, metadata, hybrid).")
    classifier: str = Field(..., description="Estimator type.")
    scaling: str = Field(..., description="Continuous feature scaling method.")
    evaluation_protocol: str = Field(
        ...,
        description="Protocol applied: '80:20 Exploratory Benchmark' or '5-Fold Stratified Cross-Validation'.",
    )
    test_accuracy: Optional[float] = Field(default=None, description="Exploratory test accuracy.")
    spam_precision: Optional[float] = Field(default=None, description="Spam class precision.")
    spam_recall: Optional[float] = Field(default=None, description="Spam class recall.")
    spam_f1: Optional[float] = Field(default=None, description="Spam class F1-score.")


class ExperimentDetail(ExperimentSummary):
    """Full experimental details including confusion matrices, parameters, and CV data."""

    feature_count: int = Field(..., description="Total feature dimension.")
    feature_mode: str = Field(..., description="Schema mode: 'paper_core' (19) or 'paper_extended' (22).")
    preprocessing: Optional[str] = Field(default=None, description="Preprocessing transformations.")
    model_parameters: Optional[str] = Field(default=None, description="Hyperparameter settings.")
    macro_f1: Optional[float] = Field(default=None, description="Unweighted macro average F1.")
    weighted_f1: Optional[float] = Field(default=None, description="Support-weighted F1.")
    roc_auc: Optional[float] = Field(default=None, description="Area under ROC curve.")
    pr_auc: Optional[float] = Field(default=None, description="Area under Precision-Recall curve.")
    confusion_matrix: Optional[ConfusionMatrixSchema] = Field(default=None, description="Confusion matrix breakdown.")
    cross_validation_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="5-Fold Stratified Cross-Validation metrics evaluated strictly on training partition.",
    )
    selection_bias_disclosure: str = Field(
        default=(
            "The 80:20 held-out test split results represent an exploratory benchmark. "
            "Candidate selection was determined strictly via 5-Fold Stratified Cross-Validation "
            "on the training partition (4,128 samples). For unbiased final performance estimation, "
            "a genuinely independent external test set should be evaluated."
        ),
        description="Scientific disclosure on evaluation protocol and model selection bias.",
    )


class ExperimentListResponse(BaseModel):
    """Response envelope for listing experimental records."""

    experiments: List[ExperimentSummary] = Field(..., description="List of experiment summaries.")
    total_experiments: int = Field(..., description="Total count of recorded experiments.")
    selection_bias_disclosure: str = Field(
        ...,
        description="Notice explaining evaluation protocol and exploratory holdout designation.",
    )
