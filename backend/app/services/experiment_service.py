"""Service for reading, validating, and presenting Phase 1 experiment reports."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.experiment import (
    ConfusionMatrixSchema,
    ExperimentDetail,
    ExperimentSummary,
)

logger = get_logger(__name__)


class ExperimentService:
    """Reads stored research benchmark reports without modifying files or rerunning training."""

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or settings.default_reports_dir

    def _load_metrics_json(self) -> Dict[str, Any]:
        """Read serialized metrics and normalize to dict mapping experiment_id -> data."""
        metrics_file = self.reports_dir / "experiment_metrics.json"
        if not metrics_file.is_file():
            logger.warning("Reports file not found: %s", metrics_file)
            return {}
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return {item.get("experiment_id", str(idx)): item for idx, item in enumerate(raw)}
            elif isinstance(raw, dict):
                return raw
            return {}
        except Exception as e:
            logger.error("Failed to read %s: %s", metrics_file, e)
            return {}


    def _load_cv_json(self) -> Dict[str, Any]:
        """Read 5-Fold Stratified Cross-Validation report."""
        cv_file = self.reports_dir / "model_selection_cv.json"
        if not cv_file.is_file():
            return {}
        try:
            with open(cv_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read %s: %s", cv_file, e)
            return {}

    def _load_comparison_csv(self) -> Optional[pd.DataFrame]:
        """Read tabular comparison CSV."""
        csv_file = self.reports_dir / "experiment_comparison.csv"
        if not csv_file.is_file():
            return None
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            logger.error("Failed to read %s: %s", csv_file, e)
            return None

    def get_selection_bias_disclosure(self) -> str:
        """Return explicit scientific disclosure regarding model selection bias."""
        cv_data = self._load_cv_json()
        if "selection_bias_disclosure" in cv_data:
            return cv_data["selection_bias_disclosure"]
        return (
            "The 80:20 held-out test split results represent an exploratory benchmark. "
            "Repeated model selection on that split introduces selection bias. Candidate selection "
            "was determined strictly via 5-Fold Stratified Cross-Validation on the training partition "
            "(4,128 messages) without consulting the final test split."
        )

    def list_experiments(self) -> List[ExperimentSummary]:
        """Return summary list of all evaluated experiment configurations."""
        metrics_dict = self._load_metrics_json()
        results: List[ExperimentSummary] = []

        if metrics_dict:
            for exp_id, data in metrics_dict.items():
                results.append(
                    ExperimentSummary(
                        experiment_id=exp_id,
                        model_name=data.get("model_name", ""),
                        feature_space=data.get("feature_space", ""),
                        classifier=data.get("classifier") or data.get("classifier_name", ""),
                        scaling=data.get("scaling", "none"),
                        evaluation_protocol="80:20 Exploratory Benchmark",
                        test_accuracy=data.get("accuracy"),
                        spam_precision=data.get("spam_precision"),
                        spam_recall=data.get("spam_recall"),
                        spam_f1=data.get("spam_f1"),
                    )
                )
        else:
            # Fallback to CSV if json is unavailable
            df = self._load_comparison_csv()
            if df is not None:
                for _, row in df.iterrows():
                    results.append(
                        ExperimentSummary(
                            experiment_id=str(row["Experiment"]),
                            model_name=str(row["Model Name"]),
                            feature_space=str(row["Feature Space"]),
                            classifier=str(row["Classifier"]),
                            scaling=str(row["Scaling"]),
                            evaluation_protocol="80:20 Exploratory Benchmark",
                            test_accuracy=float(row["Test Accuracy"]) if pd.notna(row["Test Accuracy"]) else None,
                            spam_precision=float(row["Spam Precision"]) if pd.notna(row["Spam Precision"]) else None,
                            spam_recall=float(row["Spam Recall"]) if pd.notna(row["Spam Recall"]) else None,
                            spam_f1=float(row["Spam F1"]) if pd.notna(row["Spam F1"]) else None,
                        )
                    )

        return results

    def get_experiment_detail(self, experiment_id: str) -> Optional[ExperimentDetail]:
        """Return full detail record for an experiment by ID."""
        metrics_dict = self._load_metrics_json()
        target_key = None
        for key in metrics_dict.keys():
            if key.lower() == experiment_id.lower():
                target_key = key
                break

        if not target_key:
            return None

        data = metrics_dict[target_key]
        cm_data = data.get("confusion_matrix")
        cm_schema = None
        if isinstance(cm_data, dict):
            cm_schema = ConfusionMatrixSchema(
                tn=int(cm_data.get("tn", 0)),
                fp=int(cm_data.get("fp", 0)),
                fn=int(cm_data.get("fn", 0)),
                tp=int(cm_data.get("tp", 0)),
            )

        # Cross-validation record lookup if available
        cv_data = self._load_cv_json()
        cv_metric = None
        for cand in cv_data.get("ranked_candidates", []):
            if cand.get("experiment_id", "").lower() == experiment_id.lower():
                cv_metric = cand
                break

        return ExperimentDetail(
            experiment_id=target_key,
            model_name=data.get("model_name", ""),
            feature_space=data.get("feature_space", ""),
            classifier=data.get("classifier") or data.get("classifier_name", ""),
            scaling=data.get("scaling", "none"),
            evaluation_protocol="80:20 Exploratory Benchmark",
            test_accuracy=data.get("accuracy"),
            spam_precision=data.get("spam_precision"),
            spam_recall=data.get("spam_recall"),
            spam_f1=data.get("spam_f1"),
            feature_count=int(data.get("feature_count", 0)),
            feature_mode=data.get("feature_mode", "paper_core"),
            preprocessing=data.get("preprocessing_details"),
            model_parameters=data.get("model_parameters"),
            macro_f1=data.get("macro_f1"),
            weighted_f1=data.get("weighted_f1"),
            roc_auc=data.get("roc_auc"),
            pr_auc=data.get("pr_auc"),
            confusion_matrix=cm_schema,
            cross_validation_metrics=cv_metric,
            selection_bias_disclosure=self.get_selection_bias_disclosure(),
        )
