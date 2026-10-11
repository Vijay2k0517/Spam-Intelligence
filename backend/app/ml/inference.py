"""Reusable inference module with artifact caching, input validation, and latency tracking.

Designed for direct integration into FastAPI backend in subsequent phases.
"""

import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Union
import numpy as np

from app.core.config import settings
from app.ml.features import extract_metadata_dict
from app.ml.models import SpamClassifierPipeline
from app.schemas.prediction import (
    BatchPredictionOutput,
    MessageInput,
    PredictionOutput,
)

logger = logging.getLogger(__name__)

# In-memory artifact cache to avoid deserializing on every inference call
_MODEL_CACHE: Dict[str, SpamClassifierPipeline] = {}


def clear_model_cache() -> None:
    """Clear in-memory model cache."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.debug("Model inference cache cleared.")


def get_pipeline(
    model_name: str = "hybrid_svm",
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> SpamClassifierPipeline:
    """Load or retrieve cached pipeline artifact by model name.

    Args:
        model_name: One of 'text_nb', 'text_svm', 'metadata_nb', 'metadata_svm', 'hybrid_nb', 'hybrid_svm'.
        artifacts_dir: Directory containing saved artifacts. Defaults to configured artifacts dir.

    Returns:
        Loaded SpamClassifierPipeline instance.

    Raises:
        ValueError: If model_name is unrecognized.
        FileNotFoundError: If artifact file does not exist on disk.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if model_name not in settings.experiment_registry:
        available = list(settings.experiment_registry.keys())
        raise ValueError(
            f"Unknown model name '{model_name}'. Available models: {available}"
        )

    meta = settings.experiment_registry[model_name]
    exp_id = meta["id"].lower()
    base_dir = Path(artifacts_dir) if artifacts_dir else settings.default_artifacts_dir
    candidate_1 = base_dir / f"{exp_id}_{model_name}.joblib"
    candidate_2 = base_dir / meta.get("artifact_file", "")
    if candidate_1.is_file():
        artifact_path = candidate_1
    elif candidate_2.is_file():
        artifact_path = candidate_2
    else:
        artifact_path = candidate_1

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact for '{model_name}' not found at {artifact_path}. "
            f"Please run the training pipeline first using: python -m app.ml.train"
        )

    logger.info("Loading model artifact from %s...", artifact_path)
    pipeline = SpamClassifierPipeline.load(artifact_path)
    _MODEL_CACHE[model_name] = pipeline
    return pipeline


def predict_message(
    message: str,
    model_name: str = "hybrid_svm",
    artifacts_dir: Optional[Union[str, Path]] = None,
    include_metadata: bool = False,
) -> PredictionOutput:
    """Predict whether a single SMS message is spam or ham.

    Args:
        message: Raw input SMS text.
        model_name: Target model identifier (default: 'hybrid_svm').
        artifacts_dir: Optional path to artifacts directory.
        include_metadata: Whether to return extracted metadata feature dictionary.

    Returns:
        PredictionOutput schema instance.

    Raises:
        ValueError: On empty or invalid message string, or invalid model name.
        FileNotFoundError: If model artifact is missing.
    """
    # 1. Validation using Pydantic schema
    validated_input = MessageInput(
        message=message,
        model_name=model_name,
        include_metadata=include_metadata,
    )

    # 2. Retrieve cached pipeline
    pipeline = get_pipeline(model_name=validated_input.model_name, artifacts_dir=artifacts_dir)

    # 3. Time prediction latency
    t0 = time.perf_counter()
    preds, decision_scores, probas = pipeline.predict_detailed([validated_input.message])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_id = int(preds[0])
    label_str = "spam" if pred_id == 1 else "ham"
    is_spam_bool = pred_id == 1

    dec_score = float(decision_scores[0]) if decision_scores is not None else None
    prob_val = float(probas[0]) if probas is not None else None

    # Optional metadata features extraction
    meta_dict = None
    if include_metadata:
        inc_sentiment = pipeline.metadata_extractor.include_sentiment if pipeline.metadata_extractor else False
        meta_dict = extract_metadata_dict(validated_input.message, include_sentiment=inc_sentiment)

    return PredictionOutput(
        prediction=label_str,
        is_spam=is_spam_bool,
        model_name=pipeline.model_name,
        experiment_id=pipeline.experiment_id,
        decision_score=round(dec_score, 4) if dec_score is not None else None,
        probability=round(prob_val, 4) if prob_val is not None else None,
        inference_latency_ms=round(latency_ms, 3),
        metadata_features=meta_dict,
    )


def predict_batch(
    messages: List[str],
    model_name: str = "hybrid_svm",
    artifacts_dir: Optional[Union[str, Path]] = None,
    include_metadata: bool = False,
) -> BatchPredictionOutput:
    """Predict classes for a batch of SMS messages in a single efficient call.

    Args:
        messages: List of raw input SMS texts.
        model_name: Target model identifier.
        artifacts_dir: Optional path to artifacts directory.
        include_metadata: Whether to return extracted metadata features.

    Returns:
        BatchPredictionOutput schema containing individual predictions and timing.
    """
    if not messages:
        raise ValueError("Input messages list cannot be empty.")

    pipeline = get_pipeline(model_name=model_name, artifacts_dir=artifacts_dir)

    t0 = time.perf_counter()
    preds, decision_scores, probas = pipeline.predict_detailed(messages)
    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    per_item_latency = total_latency_ms / len(messages)

    outputs: List[PredictionOutput] = []
    inc_sentiment = pipeline.metadata_extractor.include_sentiment if pipeline.metadata_extractor else False

    for idx, (p, msg) in enumerate(zip(preds, messages)):
        pred_id = int(p)
        label_str = "spam" if pred_id == 1 else "ham"
        dec_score = float(decision_scores[idx]) if decision_scores is not None else None
        prob_val = float(probas[idx]) if probas is not None else None

        meta_dict = None
        if include_metadata:
            meta_dict = extract_metadata_dict(msg, include_sentiment=inc_sentiment)

        outputs.append(
            PredictionOutput(
                prediction=label_str,
                is_spam=(pred_id == 1),
                model_name=pipeline.model_name,
                experiment_id=pipeline.experiment_id,
                decision_score=round(dec_score, 4) if dec_score is not None else None,
                probability=round(prob_val, 4) if prob_val is not None else None,
                inference_latency_ms=round(per_item_latency, 3),
                metadata_features=meta_dict,
            )
        )

    return BatchPredictionOutput(
        predictions=outputs,
        total_messages=len(messages),
        total_latency_ms=round(total_latency_ms, 3),
        model_name=model_name,
    )
