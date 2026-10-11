"""Service for coordinating ML inference, latency measurements, and prediction persistence."""

import time
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import PredictionRecord
from app.ml.features import extract_metadata_dict
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.prediction import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    BatchItemResult,
    ClassifyRequest,
    ClassifyResponse,
)
from app.services.model_registry_service import ModelRegistryService

logger = get_logger(__name__)


class ClassificationService:
    """Orchestrates message validation, cached ML model inference, and optional history storage."""

    def __init__(self, db: Optional[Session] = None, registry_service: Optional[ModelRegistryService] = None):
        self.db = db
        self.registry = registry_service or ModelRegistryService()
        self.repository = PredictionRepository(db) if db else None

    def _resolve_model_id(self, requested_model_id: Optional[str]) -> str:
        """Resolve requested or default model ID, verifying existence."""
        model_id = (requested_model_id or settings.default_model).strip()
        if model_id not in settings.experiment_registry:
            valid_models = list(settings.experiment_registry.keys())
            raise ValueError(f"Unrecognized model identifier '{model_id}'. Available models: {valid_models}")
        return model_id

    def classify_single(self, request: ClassifyRequest) -> ClassifyResponse:
        """Execute single-message classification."""
        model_id = self._resolve_model_id(request.model_id)

        # Retrieve cached model pipeline
        pipeline = self.registry.get_pipeline(model_id)

        # Measure high-resolution monotonic inference time
        t0 = time.perf_counter()
        preds, decision_scores, probas = pipeline.predict_detailed([request.message])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_spam_bool = bool(preds[0] == 1)
        label_str = "spam" if is_spam_bool else "ham"

        # Determine exact score type: Never label an SVM decision score as probability
        prob_val = float(probas[0]) if probas is not None else None
        dec_score = float(decision_scores[0]) if decision_scores is not None else None

        if prob_val is not None:
            score = round(prob_val, 4)
            score_type = "probability"
        elif dec_score is not None:
            score = round(dec_score, 4)
            score_type = "decision_score"
        else:
            score = None
            score_type = None

        # Optional metadata extraction
        meta_dict = None
        if request.include_metadata:
            inc_sentiment = pipeline.metadata_extractor.include_sentiment if pipeline.metadata_extractor else False
            meta_dict = extract_metadata_dict(request.message, include_sentiment=inc_sentiment)

        # Optional SQLite prediction history persistence
        history_id = None
        if request.save_history and self.repository:
            try:
                # Privacy check: raw message text only stored if explicitly enabled
                stored_text = request.message if settings.save_raw_messages else None
                record = PredictionRecord(
                    model_id=model_id,
                    prediction=label_str,
                    is_spam=is_spam_bool,
                    message_length=len(request.message),
                    inference_ms=latency_ms,
                    score_type=score_type,
                    spam_score=score,
                    message_text=stored_text,
                )
                saved_record = self.repository.create(record)
                history_id = saved_record.id
            except Exception as e:
                logger.error("Failed to persist prediction history for model %s: %s", model_id, type(e).__name__)
                history_id = None

        return ClassifyResponse(
            prediction=label_str,
            is_spam=is_spam_bool,
            model_id=model_id,
            inference_ms=round(latency_ms, 3),
            score=score,
            score_type=score_type,
            probability=round(prob_val, 4) if prob_val is not None else None,
            history_id=history_id,
            metadata_features=meta_dict,
        )

    def classify_batch(self, request: BatchClassifyRequest) -> BatchClassifyResponse:
        """Execute batch message classification preserving input order."""
        if not request.messages:
            raise ValueError("Batch request cannot be empty.")
        if len(request.messages) > settings.max_batch_size:
            raise ValueError(f"Batch size {len(request.messages)} exceeds maximum of {settings.max_batch_size}.")

        model_id = self._resolve_model_id(request.model_id)
        pipeline = self.registry.get_pipeline(model_id)

        batch_t0 = time.perf_counter()
        results: List[BatchItemResult] = []
        history_records: List[PredictionRecord] = []

        inc_sentiment = pipeline.metadata_extractor.include_sentiment if pipeline.metadata_extractor else False

        # Measure individual message inference precisely
        for idx, msg in enumerate(request.messages):
            t_msg = time.perf_counter()
            preds, decision_scores, probas = pipeline.predict_detailed([msg])
            msg_latency_ms = (time.perf_counter() - t_msg) * 1000.0

            is_spam_bool = bool(preds[0] == 1)
            label_str = "spam" if is_spam_bool else "ham"

            prob_val = float(probas[0]) if probas is not None else None
            dec_score = float(decision_scores[0]) if decision_scores is not None else None

            if prob_val is not None:
                score = round(prob_val, 4)
                score_type = "probability"
            elif dec_score is not None:
                score = round(dec_score, 4)
                score_type = "decision_score"
            else:
                score = None
                score_type = None

            meta_dict = None
            if request.include_metadata:
                meta_dict = extract_metadata_dict(msg, include_sentiment=inc_sentiment)

            if request.save_history and self.repository:
                stored_text = msg if settings.save_raw_messages else None
                history_records.append(
                    PredictionRecord(
                        model_id=model_id,
                        prediction=label_str,
                        is_spam=is_spam_bool,
                        message_length=len(msg),
                        inference_ms=msg_latency_ms,
                        score_type=score_type,
                        spam_score=score,
                        message_text=stored_text,
                    )
                )

            results.append(
                BatchItemResult(
                    index=idx,
                    prediction=label_str,
                    is_spam=is_spam_bool,
                    inference_ms=round(msg_latency_ms, 3),
                    score=score,
                    score_type=score_type,
                    probability=round(prob_val, 4) if prob_val is not None else None,
                    history_id=None,
                    metadata_features=meta_dict,
                )
            )

        total_batch_latency_ms = (time.perf_counter() - batch_t0) * 1000.0

        # Bulk persist history if requested
        if request.save_history and self.repository and history_records:
            try:
                saved_records = self.repository.bulk_create(history_records)
                for item_res, saved_rec in zip(results, saved_records):
                    item_res.history_id = saved_rec.id
            except Exception as e:
                logger.error("Failed to bulk persist history: %s", type(e).__name__)

        return BatchClassifyResponse(
            predictions=results,
            total_messages=len(request.messages),
            total_latency_ms=round(total_batch_latency_ms, 3),
            model_id=model_id,
        )
