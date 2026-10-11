"""Comprehensive evaluation module for held-out test predictions and reporting."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Held-out evaluation metrics container."""

    experiment_id: str
    model_name: str
    feature_space: str
    classifier: str
    total_samples: int
    ham_support: int
    spam_support: int
    accuracy: float
    spam_precision: float
    spam_recall: float
    spam_f1: float
    macro_f1: float
    weighted_f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: Dict[str, int]  # tp, tn, fp, fn
    per_class_metrics: Dict[str, Dict[str, float]]
    feature_count: Optional[int] = None
    feature_mode: Optional[str] = None
    scaling: Optional[str] = None
    preprocessing_details: Optional[str] = None
    model_parameters: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_scores: Optional[Sequence[float]] = None,
    experiment_id: str = "E?",
    model_name: str = "unknown",
    feature_space: str = "unknown",
    classifier_name: str = "unknown",
    feature_count: Optional[int] = None,
    feature_mode: Optional[str] = None,
    scaling: Optional[str] = None,
    preprocessing_details: Optional[str] = None,
    model_parameters: Optional[str] = None,
) -> EvaluationMetrics:
    """Compute held-out evaluation metrics with spam as the positive class (1).

    Handles undefined metric cases safely with zero_division=0.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
    rec = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
    f1_spam = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    # cm layout:
    # row 0 (actual ham): cm[0,0] = TN, cm[0,1] = FP
    # row 1 (actual spam): cm[1,0] = FN, cm[1,1] = TP
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    ham_sup = int(np.sum(y_t == 0))
    spam_sup = int(np.sum(y_t == 1))

    # ROC-AUC and PR-AUC if valid scores/probabilities are present
    roc_auc_val: Optional[float] = None
    pr_auc_val: Optional[float] = None
    if y_scores is not None:
        try:
            s = np.array(y_scores, dtype=float)
            if len(np.unique(y_t)) > 1 and not np.isnan(s).any():
                roc_auc_val = float(roc_auc_score(y_t, s))
                pr_auc_val = float(average_precision_score(y_t, s))
        except Exception as e:
            logger.debug("Could not compute AUC scores: %s", e)
            roc_auc_val = None
            pr_auc_val = None

    # Detailed per-class report
    clf_dict = classification_report(
        y_t,
        y_p,
        labels=[0, 1],
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        "ham": {
            "precision": float(clf_dict["ham"]["precision"]),
            "recall": float(clf_dict["ham"]["recall"]),
            "f1_score": float(clf_dict["ham"]["f1-score"]),
            "support": int(clf_dict["ham"]["support"]),
        },
        "spam": {
            "precision": float(clf_dict["spam"]["precision"]),
            "recall": float(clf_dict["spam"]["recall"]),
            "f1_score": float(clf_dict["spam"]["f1-score"]),
            "support": int(clf_dict["spam"]["support"]),
        },
    }

    return EvaluationMetrics(
        experiment_id=experiment_id,
        model_name=model_name,
        feature_space=feature_space,
        classifier=classifier_name,
        total_samples=len(y_t),
        ham_support=ham_sup,
        spam_support=spam_sup,
        accuracy=round(acc, 4),
        spam_precision=round(prec, 4),
        spam_recall=round(rec, 4),
        spam_f1=round(f1_spam, 4),
        macro_f1=round(f1_macro, 4),
        weighted_f1=round(f1_weighted, 4),
        roc_auc=round(roc_auc_val, 4) if roc_auc_val is not None else None,
        pr_auc=round(pr_auc_val, 4) if pr_auc_val is not None else None,
        confusion_matrix={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        per_class_metrics=per_class,
        feature_count=feature_count,
        feature_mode=feature_mode,
        scaling=scaling,
        preprocessing_details=preprocessing_details,
        model_parameters=model_parameters,
    )


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    title: str,
    output_path: Union[str, Path],
) -> None:
    """Generate and save confusion matrix figure using Matplotlib.

    Generated strictly from actual evaluated predictions.
    """
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham (0)", "Spam (1)"])
    ax.set_yticklabels(["Ham (0)", "Spam (1)"])
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, pad=12, fontweight="bold")

    # Cell values with counts
    labels = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix figure saved to %s", out_file)


def export_reports(
    results: List[EvaluationMetrics],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Export all required reports in JSON and CSV formats."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. experiment_metrics.json
    metrics_path = out / "experiment_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # 2. experiment_comparison.csv
    rows = []
    for r in results:
        rows.append({
            "Experiment": r.experiment_id,
            "Model Name": r.model_name,
            "Feature Space": r.feature_space,
            "Feature Count": r.feature_count if r.feature_count is not None else "N/A",
            "Feature Mode": r.feature_mode if r.feature_mode is not None else "N/A",
            "Scaling": r.scaling if r.scaling is not None else "N/A",
            "Preprocessing": r.preprocessing_details if r.preprocessing_details is not None else "N/A",
            "Classifier": r.classifier,
            "Model Parameters": r.model_parameters if r.model_parameters is not None else "N/A",
            "Test Accuracy": r.accuracy,
            "Spam Precision": r.spam_precision,
            "Spam Recall": r.spam_recall,
            "Spam F1": r.spam_f1,
            "Macro F1": r.macro_f1,
            "Weighted F1": r.weighted_f1,
            "ROC-AUC": r.roc_auc if r.roc_auc is not None else "N/A",
            "PR-AUC": r.pr_auc if r.pr_auc is not None else "N/A",
            "True Positives (TP)": r.confusion_matrix["tp"],
            "True Negatives (TN)": r.confusion_matrix["tn"],
            "False Positives (FP)": r.confusion_matrix["fp"],
            "False Negatives (FN)": r.confusion_matrix["fn"],
            "Total Test Samples": r.total_samples,
        })
    df_comp = pd.DataFrame(rows)
    comp_path = out / "experiment_comparison.csv"
    df_comp.to_csv(comp_path, index=False)

    # 3. classification_reports.json
    clf_reports = {
        f"{r.experiment_id}_{r.model_name}": {
            "experiment_id": r.experiment_id,
            "model_name": r.model_name,
            "feature_space": r.feature_space,
            "feature_count": r.feature_count,
            "scaling": r.scaling,
            "per_class": r.per_class_metrics,
            "confusion_matrix": r.confusion_matrix,
            "accuracy": r.accuracy,
        }
        for r in results
    }
    clf_path = out / "classification_reports.json"
    with open(clf_path, "w", encoding="utf-8") as f:
        json.dump(clf_reports, f, indent=2)

    return {
        "metrics_json": metrics_path,
        "comparison_csv": comp_path,
        "classification_reports_json": clf_path,
    }
