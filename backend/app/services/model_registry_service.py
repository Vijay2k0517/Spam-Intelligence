"""Service for model discovery, artifact verification, and metadata registry."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.inference import get_pipeline
from app.ml.models import SpamClassifierPipeline
from app.schemas.model import ModelDetail, ModelSummary

logger = get_logger(__name__)


class ModelRegistryService:
    """Manages model registry metadata, artifact verification, and cached pipeline loading."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or settings.default_artifacts_dir
        self._manifest_cache: Optional[Dict[str, Any]] = None

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest.json if present on disk."""
        if self._manifest_cache is not None:
            return self._manifest_cache

        manifest_file = self.artifacts_dir / "manifest.json"
        if manifest_file.is_file():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    self._manifest_cache = json.load(f)
                    return self._manifest_cache
            except Exception as e:
                logger.warning("Failed to parse manifest.json: %s", e)
        self._manifest_cache = {}
        return self._manifest_cache

    def resolve_artifact_path(self, model_id: str) -> Optional[Path]:
        """Find the real artifact path for a registered model ID, or None if missing."""
        if model_id not in settings.experiment_registry:
            return None
        meta = settings.experiment_registry[model_id]
        exp_id = meta["id"].lower()

        candidate_1 = self.artifacts_dir / f"{exp_id}_{model_id}.joblib"
        candidate_2 = self.artifacts_dir / meta.get("artifact_file", "")

        if candidate_1.is_file():
            return candidate_1
        if candidate_2.is_file():
            return candidate_2
        return candidate_1

    def is_model_available(self, model_id: str) -> bool:
        """Check whether the model's artifact file actually exists on disk."""
        path = self.resolve_artifact_path(model_id)
        return path is not None and path.is_file()

    def get_selection_basis(self, model_id: str) -> str:
        """Return the documented selection basis or research role for a model."""
        if model_id == "hybrid_mnb":
            return (
                "Leading candidate selected via 5-Fold Stratified Cross-Validation strictly on training data "
                "(Mean CV Accuracy: 98.81%, F1: 0.9501, Recall: 91.25%)."
            )
        elif model_id.startswith("hybrid_svm_scaled"):
            return "Engineering remediation resolving RBF kernel distance distortion via feature scaling."
        elif model_id == "hybrid_svm_linear":
            return "Engineering remediation benchmarking linear hyperplane on unscaled hybrid features."
        elif model_id.startswith("text_") and "stemmed" in model_id:
            return "Preprocessing comparison evaluating Porter stemming on textual BoW representation."
        elif "ext22" in model_id:
            return "Extended 22-feature representation evaluating VADER sentiment polarity features."
        elif model_id.startswith("e") or model_id in ["text_nb", "text_svm", "metadata_nb", "metadata_svm", "hybrid_nb", "hybrid_svm"]:
            return "Reference paper baseline configuration (Core 19 handcrafted features, unscaled)."
        return "Registered experiment configuration."

    def get_model_summary(self, model_id: str) -> Optional[ModelSummary]:
        """Build ModelSummary schema from registered configuration and manifest."""
        if model_id not in settings.experiment_registry:
            return None

        reg_entry = settings.experiment_registry[model_id]
        manifest = self._load_manifest().get("models", {}).get(model_id, {})

        clf_type = reg_entry["classifier"]
        is_nb = "nb" in clf_type.lower() or "naive" in clf_type.lower()
        supports_prob = is_nb
        supports_score = not is_nb

        score_type = "probability" if supports_prob else "decision_score"
        is_avail = self.is_model_available(model_id)
        feat_count = manifest.get("feature_count", 0)

        # Fallback feature count estimation if manifest is empty
        if feat_count == 0:
            f_space = reg_entry["feature_space"]
            if f_space == "metadata":
                feat_count = 22 if "22" in model_id else 19
            elif f_space == "textual":
                feat_count = 7279 if "stemmed" in model_id else 7712
            else:
                base_text = 7279 if "stemmed" in model_id else 7712
                base_meta = 22 if "22" in model_id else 19
                feat_count = base_text + base_meta

        return ModelSummary(
            model_id=model_id,
            display_name=reg_entry["name"],
            classifier_type=clf_type,
            feature_space=reg_entry["feature_space"],
            feature_count=feat_count,
            is_default=(model_id == settings.default_model),
            is_available=is_avail,
            supports_probability=supports_prob,
            supports_decision_score=supports_score,
            score_type=score_type,
            selection_basis=self.get_selection_basis(model_id),
        )

    def get_model_detail(self, model_id: str) -> Optional[ModelDetail]:
        """Build ModelDetail schema with artifact parameters and test metrics."""
        summary = self.get_model_summary(model_id)
        if summary is None:
            return None

        reg_entry = settings.experiment_registry[model_id]
        manifest = self._load_manifest().get("models", {}).get(model_id, {})

        f_mode = manifest.get("feature_mode", "paper_extended" if "22" in model_id else "paper_core")
        scaling = manifest.get("scaling", "minmax_continuous" if "scaled" in model_id else "none")

        return ModelDetail(
            **summary.model_dump(),
            experiment_id=reg_entry["id"],
            scaling=scaling,
            feature_mode=f_mode,
            preprocessing_details="Porter stemming + stopwords" if "stemmed" in model_id else "Standard BoW dual stream",
            model_parameters=f"Estimator={reg_entry['classifier']}",
            artifact_file=reg_entry.get("artifact_file"),
            test_metrics={
                "test_accuracy": manifest.get("test_accuracy"),
                "spam_precision": manifest.get("spam_precision"),
                "spam_recall": manifest.get("spam_recall"),
                "spam_f1": manifest.get("spam_f1"),
                "confusion_matrix": manifest.get("confusion_matrix"),
            } if manifest else None,
        )

    def list_models(self) -> List[ModelSummary]:
        """List all models registered in experiment registry."""
        results: List[ModelSummary] = []
        for model_id in settings.experiment_registry.keys():
            summary = self.get_model_summary(model_id)
            if summary is not None:
                results.append(summary)
        return results

    def get_pipeline(self, model_id: str) -> SpamClassifierPipeline:
        """Retrieve or load cached pipeline artifact."""
        if not self.is_model_available(model_id):
            path = self.resolve_artifact_path(model_id)
            raise FileNotFoundError(f"Model artifact for '{model_id}' is not available at {path}")
        return get_pipeline(model_name=model_id, artifacts_dir=self.artifacts_dir)
