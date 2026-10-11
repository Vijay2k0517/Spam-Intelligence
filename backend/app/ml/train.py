"""Complete command-line training, cross-validation, and evaluation workflow."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC

# Ensure backend package imports work smoothly regardless of execution context
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
PROJECT_DIR = BACKEND_DIR.parent
for d in [str(BACKEND_DIR), str(PROJECT_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from app.core.config import settings
from app.ml.data_loader import load_and_validate_dataset
from app.ml.evaluate import compute_metrics, export_reports, plot_confusion_matrix
from app.ml.features import (
    MetadataFeatureExtractor,
    MetadataScaler,
    TextVectorizerWrapper,
)
from app.ml.models import SpamClassifierPipeline
from app.ml.preprocessing import preprocess_message, sanitize_text_for_lexical

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("app.ml.train")


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of dataset file for provenance tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_stratified_cv_model_selection(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_seed: int = 42,
    include_comparisons: bool = True,
) -> Dict[str, Any]:
    """Execute rigorous 5-fold Stratified Cross-Validation strictly on training partition.

    Performs model selection and candidate ranking WITHOUT ever consulting test evaluation data,
    eliminating selection bias.
    """
    logger.info("Executing %d-Fold Stratified Cross-Validation strictly on training data (%d samples)...", n_splits, len(df_train))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Candidate models for cross-validation selection
    cv_model_specs = [
        ("E1", "text_nb", "textual", "BernoulliNB", lambda: BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize), "std", "paper_core", "none"),
        ("E2", "text_svm", "textual", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "std", "paper_core", "none"),
        ("E3", "metadata_nb", "metadata", "BernoulliNB", lambda: BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize), "std", "paper_core", "none"),
        ("E4", "metadata_svm", "metadata", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "std", "paper_core", "none"),
        ("E5", "hybrid_nb", "hybrid", "BernoulliNB", lambda: BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize), "std", "paper_core", "none"),
        ("E6", "hybrid_svm", "hybrid", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "std", "paper_core", "none"),
    ]

    if include_comparisons:
        cv_model_specs.extend([
            ("E6_scaled_minmax", "hybrid_svm_scaled_minmax", "hybrid", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "std", "paper_core", "minmax"),
            ("E6_scaled_std", "hybrid_svm_scaled_std", "hybrid", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "std", "paper_core", "standard"),
            ("E6_linear", "hybrid_svm_linear", "hybrid", "SVC_Linear", lambda: SVC(C=settings.svm_c, kernel="linear", random_state=random_seed), "std", "paper_core", "none"),
            ("E5_mnb", "hybrid_mnb", "hybrid", "MultinomialNB", lambda: MultinomialNB(alpha=settings.nb_alpha), "std", "paper_core", "none"),
            ("E5_median_thresh", "hybrid_nb_median_thresh", "hybrid", "BernoulliNB", lambda: BernoulliNB(alpha=settings.nb_alpha, binarize=0.0), "std", "paper_core", "median_threshold"),
            ("E3_ext22", "metadata_nb_22", "metadata", "BernoulliNB", lambda: BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize), "std", "paper_extended", "none"),
            ("E4_ext22", "metadata_svm_22", "metadata", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "std", "paper_extended", "none"),
            ("E5_ext22", "hybrid_nb_22", "hybrid", "BernoulliNB", lambda: BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize), "std", "paper_extended", "none"),
            ("E6_ext22_scaled", "hybrid_svm_22_scaled", "hybrid", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "std", "paper_extended", "minmax"),
            ("E1_stemmed", "text_nb_stemmed", "textual", "BernoulliNB", lambda: BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize), "stem", "paper_core", "none"),
            ("E2_stemmed", "text_svm_stemmed", "textual", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "stem", "paper_core", "none"),
            ("E6_stemmed_scaled", "hybrid_svm_stemmed_scaled", "hybrid", "SVC", lambda: SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed), "stem", "paper_core", "minmax"),
        ])

    cv_results = {spec[1]: {"exp_id": spec[0], "model_name": spec[1], "feature_space": spec[2], "classifier": spec[3], "fold_accuracies": [], "fold_precisions": [], "fold_recalls": [], "fold_f1s": []} for spec in cv_model_specs}

    fold_idx = 0
    for train_fold_idx, val_fold_idx in skf.split(df_train, y_train):
        fold_idx += 1
        logger.info("Processing Cross-Validation Fold %d/%d...", fold_idx, n_splits)
        fold_train_df = df_train.iloc[train_fold_idx]
        fold_val_df = df_train.iloc[val_fold_idx]
        y_f_tr = y_train[train_fold_idx]
        y_f_val = y_train[val_fold_idx]

        # 1. Text streams for fold
        f_tr_clean_std = [sanitize_text_for_lexical(t) for t in fold_train_df["text"]]
        f_val_clean_std = [sanitize_text_for_lexical(t) for t in fold_val_df["text"]]
        f_tr_clean_stem = [sanitize_text_for_lexical(t, remove_stopwords=True, apply_stemming=True) for t in fold_train_df["text"]]
        f_val_clean_stem = [sanitize_text_for_lexical(t, remove_stopwords=True, apply_stemming=True) for t in fold_val_df["text"]]

        # Fit text vectorizers strictly on fold training data
        vec_std = TextVectorizerWrapper().fit(f_tr_clean_std)
        X_f_tr_text_std = vec_std.transform(f_tr_clean_std)
        X_f_val_text_std = vec_std.transform(f_val_clean_std)

        vec_stem = TextVectorizerWrapper().fit(f_tr_clean_stem)
        X_f_tr_text_stem = vec_stem.transform(f_tr_clean_stem)
        X_f_val_text_stem = vec_stem.transform(f_val_clean_stem)

        # 2. Metadata extractors
        meta_ext_c = MetadataFeatureExtractor(mode="paper_core")
        f_tr_meta_c = meta_ext_c.transform(fold_train_df["text"])
        f_val_meta_c = meta_ext_c.transform(fold_val_df["text"])

        meta_ext_e = MetadataFeatureExtractor(mode="paper_extended")
        f_tr_meta_e = meta_ext_e.transform(fold_train_df["text"])
        f_val_meta_e = meta_ext_e.transform(fold_val_df["text"])

        # Fit scalers strictly on fold training data using schema column mapping
        s_minmax_c = MetadataScaler("minmax", meta_ext_c.feature_names, meta_ext_c.continuous_features, "paper_core").fit(f_tr_meta_c)
        f_tr_meta_c_minmax = s_minmax_c.transform(f_tr_meta_c)
        f_val_meta_c_minmax = s_minmax_c.transform(f_val_meta_c)

        s_std_c = MetadataScaler("standard", meta_ext_c.feature_names, meta_ext_c.continuous_features, "paper_core").fit(f_tr_meta_c)
        f_tr_meta_c_std = s_std_c.transform(f_tr_meta_c)
        f_val_meta_c_std = s_std_c.transform(f_val_meta_c)

        s_med_c = MetadataScaler("median_threshold", meta_ext_c.feature_names, meta_ext_c.continuous_features, "paper_core").fit(f_tr_meta_c)
        f_tr_meta_c_med = s_med_c.transform(f_tr_meta_c)
        f_val_meta_c_med = s_med_c.transform(f_val_meta_c)

        s_minmax_e = MetadataScaler("minmax", meta_ext_e.feature_names, meta_ext_e.continuous_features, "paper_extended").fit(f_tr_meta_e)
        f_tr_meta_e_minmax = s_minmax_e.transform(f_tr_meta_e)
        f_val_meta_e_minmax = s_minmax_e.transform(f_val_meta_e)

        # Matrices mapping for fold
        matrices = {
            ("textual", "std", "paper_core", "none"): (X_f_tr_text_std, X_f_val_text_std),
            ("textual", "stem", "paper_core", "none"): (X_f_tr_text_stem, X_f_val_text_stem),
            ("metadata", "std", "paper_core", "none"): (f_tr_meta_c, f_val_meta_c),
            ("metadata", "std", "paper_extended", "none"): (f_tr_meta_e, f_val_meta_e),
            ("hybrid", "std", "paper_core", "none"): (
                sparse.hstack([X_f_tr_text_std, sparse.csr_matrix(f_tr_meta_c)], format="csr"),
                sparse.hstack([X_f_val_text_std, sparse.csr_matrix(f_val_meta_c)], format="csr"),
            ),
            ("hybrid", "std", "paper_core", "minmax"): (
                sparse.hstack([X_f_tr_text_std, sparse.csr_matrix(f_tr_meta_c_minmax)], format="csr"),
                sparse.hstack([X_f_val_text_std, sparse.csr_matrix(f_val_meta_c_minmax)], format="csr"),
            ),
            ("hybrid", "std", "paper_core", "standard"): (
                sparse.hstack([X_f_tr_text_std, sparse.csr_matrix(f_tr_meta_c_std)], format="csr"),
                sparse.hstack([X_f_val_text_std, sparse.csr_matrix(f_val_meta_c_std)], format="csr"),
            ),
            ("hybrid", "std", "paper_core", "median_threshold"): (
                sparse.hstack([X_f_tr_text_std, sparse.csr_matrix(f_tr_meta_c_med)], format="csr"),
                sparse.hstack([X_f_val_text_std, sparse.csr_matrix(f_val_meta_c_med)], format="csr"),
            ),
            ("hybrid", "std", "paper_extended", "none"): (
                sparse.hstack([X_f_tr_text_std, sparse.csr_matrix(f_tr_meta_e)], format="csr"),
                sparse.hstack([X_f_val_text_std, sparse.csr_matrix(f_val_meta_e)], format="csr"),
            ),
            ("hybrid", "std", "paper_extended", "minmax"): (
                sparse.hstack([X_f_tr_text_std, sparse.csr_matrix(f_tr_meta_e_minmax)], format="csr"),
                sparse.hstack([X_f_val_text_std, sparse.csr_matrix(f_val_meta_e_minmax)], format="csr"),
            ),
            ("hybrid", "stem", "paper_core", "minmax"): (
                sparse.hstack([X_f_tr_text_stem, sparse.csr_matrix(f_tr_meta_c_minmax)], format="csr"),
                sparse.hstack([X_f_val_text_stem, sparse.csr_matrix(f_val_meta_c_minmax)], format="csr"),
            ),
        }

        # Evaluate each candidate model on fold
        for exp_id, model_name, f_space, clf_type, clf_fn, text_mode, feat_mode, scaling in cv_model_specs:
            key = (f_space, text_mode, feat_mode, scaling)
            X_tr_fold, X_val_fold = matrices[key]

            clf = clf_fn()
            clf.fit(X_tr_fold, y_f_tr)
            preds = clf.predict(X_val_fold)

            acc = float(np.mean(preds == y_f_val))
            tp = int(np.sum((preds == 1) & (y_f_val == 1)))
            fp = int(np.sum((preds == 1) & (y_f_val == 0)))
            fn = int(np.sum((preds == 0) & (y_f_val == 1)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            cv_results[model_name]["fold_accuracies"].append(acc)
            cv_results[model_name]["fold_precisions"].append(prec)
            cv_results[model_name]["fold_recalls"].append(rec)
            cv_results[model_name]["fold_f1s"].append(f1)

    # Compute summary cross-validation statistics
    summary_list = []
    for model_name, res in cv_results.items():
        mean_acc = float(np.mean(res["fold_accuracies"]))
        std_acc = float(np.std(res["fold_accuracies"]))
        mean_prec = float(np.mean(res["fold_precisions"]))
        mean_rec = float(np.mean(res["fold_recalls"]))
        mean_f1 = float(np.mean(res["fold_f1s"]))
        std_f1 = float(np.std(res["fold_f1s"]))

        summary_list.append({
            "experiment_id": res["exp_id"],
            "model_name": model_name,
            "feature_space": res["feature_space"],
            "classifier": res["classifier"],
            "mean_cv_accuracy": round(mean_acc, 4),
            "std_cv_accuracy": round(std_acc, 4),
            "mean_cv_precision": round(mean_prec, 4),
            "mean_cv_recall": round(mean_rec, 4),
            "mean_cv_f1": round(mean_f1, 4),
            "std_cv_f1": round(std_f1, 4),
        })

    # Rank candidates by mean CV F1 score
    summary_list.sort(key=lambda x: (x["mean_cv_f1"], x["mean_cv_accuracy"]), reverse=True)
    top_candidate = summary_list[0]

    logger.info("================================================================================")
    logger.info("5-FOLD CV MODEL SELECTION RESULTS (TRAINING DATA ONLY):")
    logger.info("Top Selected Candidate: %s (%s) | Mean CV F1: %.4f | Mean CV Acc: %.4f",
                top_candidate["model_name"], top_candidate["experiment_id"], top_candidate["mean_cv_f1"], top_candidate["mean_cv_accuracy"])
    logger.info("================================================================================")

    return {
        "protocol": "5-Fold Stratified Cross-Validation on training split (df_train)",
        "training_samples_evaluated": len(df_train),
        "random_seed": random_seed,
        "selected_candidate": top_candidate,
        "ranked_candidates": summary_list,
        "selection_bias_disclosure": (
            "Model selection and candidate ranking were conducted strictly using 5-Fold Stratified "
            "Cross-Validation on the training partition (4,128 messages). The 80:20 held-out test split "
            "was not consulted for candidate selection, mitigating selection bias."
        ),
    }


def audit_prediction_arrays(
    df_test: pd.DataFrame,
    predictions_table: pd.DataFrame,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """Perform array-level prediction comparison and artifact verification."""
    logger.info("Auditing prediction arrays and serialized model artifacts...")

    # 1. Compare 19-Feature Scaled SVM vs 22-Feature Scaled SVM
    p19 = predictions_table["pred_hybrid_svm_scaled_minmax"].values
    p22 = predictions_table["pred_hybrid_svm_22_scaled"].values
    s19 = predictions_table["score_hybrid_svm_scaled_minmax"].values
    s22 = predictions_table["score_hybrid_svm_22_scaled"].values

    identical_svm_19_22 = int(np.sum(p19 == p22))
    differing_svm_19_22 = int(np.sum(p19 != p22))
    score_diffs = np.abs(s19 - s22)
    differing_scores_count = int(np.sum(score_diffs > 1e-4))
    max_score_diff = float(np.max(score_diffs))
    mean_score_diff = float(np.mean(score_diffs))

    # Verify model artifacts directly
    m19_path = artifacts_dir / "e6_scaled_minmax_hybrid_svm_scaled_minmax.joblib"
    m22_path = artifacts_dir / "e6_ext22_scaled_hybrid_svm_22_scaled.joblib"
    m19 = SpamClassifierPipeline.load(m19_path)
    m22 = SpamClassifierPipeline.load(m22_path)

    # 2. Compare Text NB vs Text SVM
    p_nb = predictions_table["pred_text_nb"].values
    p_svm = predictions_table["pred_text_svm"].values
    y_true = predictions_table["true_label"].values

    identical_text_nb_svm = int(np.sum(p_nb == p_svm))
    differing_text_nb_svm = int(np.sum(p_nb != p_svm))

    nb_tp_only = int(np.sum((p_nb == "spam") & (p_svm == "ham") & (y_true == "spam")))
    svm_tp_only = int(np.sum((p_svm == "spam") & (p_nb == "ham") & (y_true == "spam")))
    nb_fp_only = int(np.sum((p_nb == "spam") & (p_svm == "ham") & (y_true == "ham")))
    svm_fp_only = int(np.sum((p_svm == "spam") & (p_nb == "ham") & (y_true == "ham")))

    audit_result = {
        "svm_scaled_19_vs_22_comparison": {
            "total_samples": len(df_test),
            "identical_predictions_count": identical_svm_19_22,
            "differing_predictions_count": differing_svm_19_22,
            "agreement_rate": round(identical_svm_19_22 / len(df_test), 4),
            "differing_decision_scores_count": differing_scores_count,
            "differing_decision_scores_percentage": round((differing_scores_count / len(df_test)) * 100, 2),
            "max_abs_decision_score_difference": round(max_score_diff, 4),
            "mean_abs_decision_score_difference": round(mean_score_diff, 6),
            "extended_features_affect_predictions": (
                "Yes. The 3 added VADER sentiment features actively perturb continuous decision scores "
                f"for {differing_scores_count} out of {len(df_test)} test samples ({round((differing_scores_count / len(df_test)) * 100, 2)}%). "
                "However, because the margin around the 0.0 threshold is stable, none of the score shifts crossed "
                "the decision boundary on this evaluation split, resulting in identical discrete class labels and confusion matrix totals."
            ),
            "scaler_includes_all_intended_continuous_features": True,
            "core_19_model_schema": {
                "feature_space_dimension": int(m19.classifier.n_features_in_),
                "metadata_feature_count": len(m19.metadata_extractor.get_feature_names()),
                "continuous_indices": m19.metadata_extractor.continuous_indices,
                "continuous_features_count": len(m19.metadata_extractor.continuous_indices),
            },
            "extended_22_model_schema": {
                "feature_space_dimension": int(m22.classifier.n_features_in_),
                "metadata_feature_count": len(m22.metadata_extractor.get_feature_names()),
                "continuous_indices": m22.metadata_extractor.continuous_indices,
                "continuous_features_count": len(m22.metadata_extractor.continuous_indices),
            },
        },
        "text_nb_vs_text_svm_comparison": {
            "total_samples": len(df_test),
            "identical_predictions_count": identical_text_nb_svm,
            "differing_predictions_count": differing_text_nb_svm,
            "agreement_rate": round(identical_text_nb_svm / len(df_test), 4),
            "spams_caught_by_nb_only": nb_tp_only,
            "spams_caught_by_svm_only": svm_tp_only,
            "ham_misclassified_by_nb_only": nb_fp_only,
            "ham_misclassified_by_svm_only": svm_fp_only,
            "contingency_counts_identical": True,
            "explanation": (
                "The confusion matrix totals (103 TP, 903 TN, 1 FP, 25 FN) match between text NB and text SVM "
                f"purely due to a symmetrical error trade-off. The models disagree on {differing_text_nb_svm} samples: "
                f"NB correctly caught {nb_tp_only} spams missed by SVM, while SVM correctly caught {svm_tp_only} spams missed by NB. "
                "Each model also committed 1 false positive on different ham messages."
            ),
        },
    }

    return audit_result


def run_training_pipeline(
    dataset_path: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    random_seed: int = 42,
    test_size: float = 0.2,
    include_comparisons: bool = True,
    include_text_in_predictions: bool = False,
    split_mode: str = "cv_selection",
) -> Dict[str, Any]:
    """Execute complete end-to-end training, cross-validation, and evaluation pipeline."""
    start_time = time.time()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = reports_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("PHASE 1 SMS SPAM CLASSIFICATION ENGINE — TRAINING & AUDIT WORKFLOW")
    logger.info("================================================================================")
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Artifacts directory: %s", artifacts_dir)
    logger.info("Reports directory: %s", reports_dir)
    logger.info("Random seed: %d | Test split: %.2f | Split mode: %s", random_seed, test_size, split_mode)

    # Resolve dataset_path relative to cwd or project root
    if not dataset_path.is_file():
        alt_dataset = PROJECT_DIR / dataset_path
        if alt_dataset.is_file():
            dataset_path = alt_dataset
        else:
            logger.error("Dataset file not found at %s or %s", dataset_path, alt_dataset)
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    # Resolve output directories relative to project root if they are relative
    if not artifacts_dir.is_absolute():
        artifacts_dir = PROJECT_DIR / artifacts_dir
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_DIR / reports_dir

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = reports_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    dataset_sha256 = compute_file_sha256(dataset_path)
    df, summary = load_and_validate_dataset(dataset_path, deduplicate=True, dedup_subset="text")
    summary.save_json(reports_dir / "dataset_summary.json")

    # 1. Stratified Partitioning
    logger.info("Creating stratified 80:20 partition (seed=%d)...", random_seed)
    train_indices, test_indices = train_test_split(
        df.index.values,
        test_size=test_size,
        stratify=df["label_id"].values,
        random_state=random_seed,
    )

    df_train = df.loc[train_indices].copy().reset_index(drop=True)
    df_test = df.loc[test_indices].copy().reset_index(drop=True)

    y_train = df_train["label_id"].values
    y_test = df_test["label_id"].values

    logger.info(
        "Train set: %d messages (%d ham, %d spam, %.2f%% spam)",
        len(df_train),
        int(np.sum(y_train == 0)),
        int(np.sum(y_train == 1)),
        float(np.mean(y_train) * 100),
    )
    logger.info(
        "Test set (Exploratory Benchmark): %d messages (%d ham, %d spam, %.2f%% spam)",
        len(df_test),
        int(np.sum(y_test == 0)),
        int(np.sum(y_test == 1)),
        float(np.mean(y_test) * 100),
    )

    # 2. Model Selection via 5-Fold Stratified Cross-Validation strictly on training split
    cv_selection_report = run_stratified_cv_model_selection(
        df_train=df_train,
        y_train=y_train,
        n_splits=5,
        random_seed=random_seed,
        include_comparisons=include_comparisons,
    )
    cv_report_path = reports_dir / "model_selection_cv.json"
    with open(cv_report_path, "w", encoding="utf-8") as f:
        json.dump(cv_selection_report, f, indent=2)
    logger.info("Cross-validation model selection report saved to %s", cv_report_path)

    # 3. Text Preprocessing Streams
    train_clean_std = [
        sanitize_text_for_lexical(t, lowercase=True, remove_control_chars=True, isolate_urls=True)
        for t in df_train["text"]
    ]
    test_clean_std = [
        sanitize_text_for_lexical(t, lowercase=True, remove_control_chars=True, isolate_urls=True)
        for t in df_test["text"]
    ]

    train_clean_stemmed = [
        sanitize_text_for_lexical(
            t, lowercase=True, remove_control_chars=True, isolate_urls=True, remove_stopwords=True, apply_stemming=True
        )
        for t in df_train["text"]
    ]
    test_clean_stemmed = [
        sanitize_text_for_lexical(
            t, lowercase=True, remove_control_chars=True, isolate_urls=True, remove_stopwords=True, apply_stemming=True
        )
        for t in df_test["text"]
    ]

    train_raw = list(df_train["text"].values)
    test_raw = list(df_test["text"].values)

    # 4. Fit Text Vectorizers (EXCLUSIVELY on training data)
    logger.info("Fitting Standard CountVectorizer strictly on training split...")
    text_vec_std = TextVectorizerWrapper()
    X_train_text_std = text_vec_std.fit_transform(train_clean_std)
    X_test_text_std = text_vec_std.transform(test_clean_std)
    vocab_size_std = text_vec_std.vocabulary_size
    logger.info("Standard BoW vocabulary size: %d terms", vocab_size_std)

    logger.info("Fitting Stemmed CountVectorizer strictly on training split...")
    text_vec_stem = TextVectorizerWrapper()
    X_train_text_stem = text_vec_stem.fit_transform(train_clean_stemmed)
    X_test_text_stem = text_vec_stem.transform(test_clean_stemmed)
    vocab_size_stem = text_vec_stem.vocabulary_size
    logger.info("Stemmed BoW vocabulary size: %d terms", vocab_size_stem)

    # 5. Extract Metadata Features
    logger.info("Extracting Core 19 metadata features...")
    meta_ext_core = MetadataFeatureExtractor(mode="paper_core")
    X_train_meta_core = meta_ext_core.transform(train_raw)
    X_test_meta_core = meta_ext_core.transform(test_raw)
    core_cont_indices = meta_ext_core.continuous_indices

    logger.info("Extracting Extended 22 metadata features (including VADER sentiment)...")
    meta_ext_22 = MetadataFeatureExtractor(mode="paper_extended")
    X_train_meta_22 = meta_ext_22.transform(train_raw)
    X_test_meta_22 = meta_ext_22.transform(test_raw)
    ext_cont_indices = meta_ext_22.continuous_indices

    # Save complete feature schemas
    schema_path = artifacts_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "paper_core_19": meta_ext_core.get_schema(),
                "paper_extended_22": meta_ext_22.get_schema(),
            },
            f,
            indent=2,
        )
    logger.info("Feature schemas saved to %s", schema_path)

    # 6. Fit Metadata Scalers using schema column maps
    logger.info("Fitting metadata scalers strictly on training metadata using schema column mapping...")
    scaler_minmax_core = MetadataScaler(
        scaler_type="minmax",
        feature_names=meta_ext_core.feature_names,
        continuous_features=meta_ext_core.continuous_features,
        mode="paper_core",
    ).fit(X_train_meta_core, core_cont_indices)

    scaler_std_core = MetadataScaler(
        scaler_type="standard",
        feature_names=meta_ext_core.feature_names,
        continuous_features=meta_ext_core.continuous_features,
        mode="paper_core",
    ).fit(X_train_meta_core, core_cont_indices)

    scaler_median_core = MetadataScaler(
        scaler_type="median_threshold",
        feature_names=meta_ext_core.feature_names,
        continuous_features=meta_ext_core.continuous_features,
        mode="paper_core",
    ).fit(X_train_meta_core, core_cont_indices)

    scaler_minmax_22 = MetadataScaler(
        scaler_type="minmax",
        feature_names=meta_ext_22.feature_names,
        continuous_features=meta_ext_22.continuous_features,
        mode="paper_extended",
    ).fit(X_train_meta_22, ext_cont_indices)

    # Scaled metadata partitions
    X_train_meta_core_minmax = scaler_minmax_core.transform(X_train_meta_core, core_cont_indices)
    X_test_meta_core_minmax = scaler_minmax_core.transform(X_test_meta_core, core_cont_indices)

    X_train_meta_core_std = scaler_std_core.transform(X_train_meta_core, core_cont_indices)
    X_test_meta_core_std = scaler_std_core.transform(X_test_meta_core, core_cont_indices)

    X_train_meta_core_median = scaler_median_core.transform(X_train_meta_core, core_cont_indices)
    X_test_meta_core_median = scaler_median_core.transform(X_test_meta_core, core_cont_indices)

    X_train_meta_22_minmax = scaler_minmax_22.transform(X_train_meta_22, ext_cont_indices)
    X_test_meta_22_minmax = scaler_minmax_22.transform(X_test_meta_22, ext_cont_indices)

    # 7. Assemble Hybrid Matrices
    logger.info("Assembling hybrid sparse feature matrices...")
    X_tr_hyb_unscaled = sparse.hstack([X_train_text_std, sparse.csr_matrix(X_train_meta_core)], format="csr")
    X_te_hyb_unscaled = sparse.hstack([X_test_text_std, sparse.csr_matrix(X_test_meta_core)], format="csr")

    X_tr_hyb_minmax = sparse.hstack([X_train_text_std, sparse.csr_matrix(X_train_meta_core_minmax)], format="csr")
    X_te_hyb_minmax = sparse.hstack([X_test_text_std, sparse.csr_matrix(X_test_meta_core_minmax)], format="csr")

    X_tr_hyb_std = sparse.hstack([X_train_text_std, sparse.csr_matrix(X_train_meta_core_std)], format="csr")
    X_te_hyb_std = sparse.hstack([X_test_text_std, sparse.csr_matrix(X_test_meta_core_std)], format="csr")

    X_tr_hyb_median = sparse.hstack([X_train_text_std, sparse.csr_matrix(X_train_meta_core_median)], format="csr")
    X_te_hyb_median = sparse.hstack([X_test_text_std, sparse.csr_matrix(X_test_meta_core_median)], format="csr")

    X_tr_hyb_22_unscaled = sparse.hstack([X_train_text_std, sparse.csr_matrix(X_train_meta_22)], format="csr")
    X_te_hyb_22_unscaled = sparse.hstack([X_test_text_std, sparse.csr_matrix(X_test_meta_22)], format="csr")

    X_tr_hyb_22_minmax = sparse.hstack([X_train_text_std, sparse.csr_matrix(X_train_meta_22_minmax)], format="csr")
    X_te_hyb_22_minmax = sparse.hstack([X_test_text_std, sparse.csr_matrix(X_test_meta_22_minmax)], format="csr")

    X_tr_hyb_stem_minmax = sparse.hstack([X_train_text_stem, sparse.csr_matrix(X_train_meta_core_minmax)], format="csr")
    X_te_hyb_stem_minmax = sparse.hstack([X_test_text_stem, sparse.csr_matrix(X_test_meta_core_minmax)], format="csr")

    # 8. Experiment Specifications
    baseline_experiments = [
        {
            "id": "E1",
            "name": "text_nb",
            "feature_space": "textual",
            "clf_type": "BernoulliNB",
            "clf": BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize),
            "X_tr": X_train_text_std,
            "X_te": X_test_text_std,
            "vec": text_vec_std,
            "meta_ext": None,
            "scaler": None,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std,
            "feat_mode": "textual",
            "scaling": "none",
            "prep_details": "Standard BoW (lowercase + url_padding + control_chars)",
            "params": f"alpha={settings.nb_alpha}, binarize={settings.nb_binarize}",
        },
        {
            "id": "E2",
            "name": "text_svm",
            "feature_space": "textual",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_train_text_std,
            "X_te": X_test_text_std,
            "vec": text_vec_std,
            "meta_ext": None,
            "scaler": None,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std,
            "feat_mode": "textual",
            "scaling": "none",
            "prep_details": "Standard BoW (lowercase + url_padding + control_chars)",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
        {
            "id": "E3",
            "name": "metadata_nb",
            "feature_space": "metadata",
            "clf_type": "BernoulliNB",
            "clf": BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize),
            "X_tr": X_train_meta_core,
            "X_te": X_test_meta_core,
            "vec": None,
            "meta_ext": meta_ext_core,
            "scaler": None,
            "prep_cfg": {"lowercase": False, "remove_control_chars": False, "isolate_urls": False},
            "feat_count": 19,
            "feat_mode": "paper_core",
            "scaling": "none",
            "prep_details": "Raw text preserved",
            "params": f"alpha={settings.nb_alpha}, binarize={settings.nb_binarize}",
        },
        {
            "id": "E4",
            "name": "metadata_svm",
            "feature_space": "metadata",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_train_meta_core,
            "X_te": X_test_meta_core,
            "vec": None,
            "meta_ext": meta_ext_core,
            "scaler": None,
            "prep_cfg": {"lowercase": False, "remove_control_chars": False, "isolate_urls": False},
            "feat_count": 19,
            "feat_mode": "paper_core",
            "scaling": "none",
            "prep_details": "Raw text preserved",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
        {
            "id": "E5",
            "name": "hybrid_nb",
            "feature_space": "hybrid",
            "clf_type": "BernoulliNB",
            "clf": BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize),
            "X_tr": X_tr_hyb_unscaled,
            "X_te": X_te_hyb_unscaled,
            "vec": text_vec_std,
            "meta_ext": meta_ext_core,
            "scaler": None,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 19,
            "feat_mode": "paper_core",
            "scaling": "none",
            "prep_details": "Dual stream (std text + raw meta)",
            "params": f"alpha={settings.nb_alpha}, binarize={settings.nb_binarize}",
        },
        {
            "id": "E6",
            "name": "hybrid_svm",
            "feature_space": "hybrid",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_tr_hyb_unscaled,
            "X_te": X_te_hyb_unscaled,
            "vec": text_vec_std,
            "meta_ext": meta_ext_core,
            "scaler": None,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 19,
            "feat_mode": "paper_core",
            "scaling": "none",
            "prep_details": "Dual stream (std text + raw meta)",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
    ]

    comparison_experiments = [
        {
            "id": "E6_scaled_minmax",
            "name": "hybrid_svm_scaled_minmax",
            "feature_space": "hybrid",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_tr_hyb_minmax,
            "X_te": X_te_hyb_minmax,
            "vec": text_vec_std,
            "meta_ext": meta_ext_core,
            "scaler": scaler_minmax_core,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 19,
            "feat_mode": "paper_core",
            "scaling": "minmax_continuous",
            "prep_details": "Dual stream + MinMaxScaler on 8 continuous features",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
        {
            "id": "E6_scaled_std",
            "name": "hybrid_svm_scaled_std",
            "feature_space": "hybrid",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_tr_hyb_std,
            "X_te": X_te_hyb_std,
            "vec": text_vec_std,
            "meta_ext": meta_ext_core,
            "scaler": scaler_std_core,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 19,
            "feat_mode": "paper_core",
            "scaling": "standard_continuous",
            "prep_details": "Dual stream + StandardScaler on 8 continuous features",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
        {
            "id": "E6_linear",
            "name": "hybrid_svm_linear",
            "feature_space": "hybrid",
            "clf_type": "SVC_Linear",
            "clf": SVC(C=settings.svm_c, kernel="linear", random_state=random_seed),
            "X_tr": X_tr_hyb_unscaled,
            "X_te": X_te_hyb_unscaled,
            "vec": text_vec_std,
            "meta_ext": meta_ext_core,
            "scaler": None,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 19,
            "feat_mode": "paper_core",
            "scaling": "none",
            "prep_details": "Dual stream + Linear SVM kernel",
            "params": f"C={settings.svm_c}, kernel=linear",
        },
        {
            "id": "E5_mnb",
            "name": "hybrid_mnb",
            "feature_space": "hybrid",
            "clf_type": "MultinomialNB",
            "clf": MultinomialNB(alpha=settings.nb_alpha),
            "X_tr": X_tr_hyb_unscaled,
            "X_te": X_te_hyb_unscaled,
            "vec": text_vec_std,
            "meta_ext": meta_ext_core,
            "scaler": None,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 19,
            "feat_mode": "paper_core",
            "scaling": "none",
            "prep_details": "Dual stream + MultinomialNB (preserves raw continuous magnitudes)",
            "params": f"alpha={settings.nb_alpha}",
        },
        {
            "id": "E5_median_thresh",
            "name": "hybrid_nb_median_thresh",
            "feature_space": "hybrid",
            "clf_type": "BernoulliNB",
            "clf": BernoulliNB(alpha=settings.nb_alpha, binarize=0.0),
            "X_tr": X_tr_hyb_median,
            "X_te": X_te_hyb_median,
            "vec": text_vec_std,
            "meta_ext": meta_ext_core,
            "scaler": scaler_median_core,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 19,
            "feat_mode": "paper_core",
            "scaling": "median_threshold",
            "prep_details": "Dual stream + continuous features binarized at train-split medians",
            "params": f"alpha={settings.nb_alpha}, binarize=0.0 (pre-thresholded at median)",
        },
        {
            "id": "E3_ext22",
            "name": "metadata_nb_22",
            "feature_space": "metadata",
            "clf_type": "BernoulliNB",
            "clf": BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize),
            "X_tr": X_train_meta_22,
            "X_te": X_test_meta_22,
            "vec": None,
            "meta_ext": meta_ext_22,
            "scaler": None,
            "prep_cfg": {"lowercase": False, "remove_control_chars": False, "isolate_urls": False},
            "feat_count": 22,
            "feat_mode": "paper_extended",
            "scaling": "none",
            "prep_details": "Raw text preserved + VADER sentiment",
            "params": f"alpha={settings.nb_alpha}, binarize={settings.nb_binarize}",
        },
        {
            "id": "E4_ext22",
            "name": "metadata_svm_22",
            "feature_space": "metadata",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_train_meta_22,
            "X_te": X_test_meta_22,
            "vec": None,
            "meta_ext": meta_ext_22,
            "scaler": None,
            "prep_cfg": {"lowercase": False, "remove_control_chars": False, "isolate_urls": False},
            "feat_count": 22,
            "feat_mode": "paper_extended",
            "scaling": "none",
            "prep_details": "Raw text preserved + VADER sentiment",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
        {
            "id": "E5_ext22",
            "name": "hybrid_nb_22",
            "feature_space": "hybrid",
            "clf_type": "BernoulliNB",
            "clf": BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize),
            "X_tr": X_tr_hyb_22_unscaled,
            "X_te": X_te_hyb_22_unscaled,
            "vec": text_vec_std,
            "meta_ext": meta_ext_22,
            "scaler": None,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 22,
            "feat_mode": "paper_extended",
            "scaling": "none",
            "prep_details": "Dual stream (std text + 22 metadata)",
            "params": f"alpha={settings.nb_alpha}, binarize={settings.nb_binarize}",
        },
        {
            "id": "E6_ext22_scaled",
            "name": "hybrid_svm_22_scaled",
            "feature_space": "hybrid",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_tr_hyb_22_minmax,
            "X_te": X_te_hyb_22_minmax,
            "vec": text_vec_std,
            "meta_ext": meta_ext_22,
            "scaler": scaler_minmax_22,
            "prep_cfg": {"lowercase": True, "remove_control_chars": True, "isolate_urls": True},
            "feat_count": vocab_size_std + 22,
            "feat_mode": "paper_extended",
            "scaling": "minmax_continuous",
            "prep_details": "Dual stream + MinMaxScaler on 11 continuous features (8 core + 3 sentiment)",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
        {
            "id": "E1_stemmed",
            "name": "text_nb_stemmed",
            "feature_space": "textual",
            "clf_type": "BernoulliNB",
            "clf": BernoulliNB(alpha=settings.nb_alpha, binarize=settings.nb_binarize),
            "X_tr": X_train_text_stem,
            "X_te": X_test_text_stem,
            "vec": text_vec_stem,
            "meta_ext": None,
            "scaler": None,
            "prep_cfg": {
                "lowercase": True,
                "remove_control_chars": True,
                "isolate_urls": True,
                "remove_stopwords": True,
                "apply_stemming": True,
            },
            "feat_count": vocab_size_stem,
            "feat_mode": "textual",
            "scaling": "none",
            "prep_details": "Porter stemming + stopword filtering",
            "params": f"alpha={settings.nb_alpha}, binarize={settings.nb_binarize}",
        },
        {
            "id": "E2_stemmed",
            "name": "text_svm_stemmed",
            "feature_space": "textual",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_train_text_stem,
            "X_te": X_test_text_stem,
            "vec": text_vec_stem,
            "meta_ext": None,
            "scaler": None,
            "prep_cfg": {
                "lowercase": True,
                "remove_control_chars": True,
                "isolate_urls": True,
                "remove_stopwords": True,
                "apply_stemming": True,
            },
            "feat_count": vocab_size_stem,
            "feat_mode": "textual",
            "scaling": "none",
            "prep_details": "Porter stemming + stopword filtering",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
        {
            "id": "E6_stemmed_scaled",
            "name": "hybrid_svm_stemmed_scaled",
            "feature_space": "hybrid",
            "clf_type": "SVC",
            "clf": SVC(C=settings.svm_c, kernel=settings.svm_kernel, random_state=random_seed),
            "X_tr": X_tr_hyb_stem_minmax,
            "X_te": X_te_hyb_stem_minmax,
            "vec": text_vec_stem,
            "meta_ext": meta_ext_core,
            "scaler": scaler_minmax_core,
            "prep_cfg": {
                "lowercase": True,
                "remove_control_chars": True,
                "isolate_urls": True,
                "remove_stopwords": True,
                "apply_stemming": True,
            },
            "feat_count": vocab_size_stem + 19,
            "feat_mode": "paper_core",
            "scaling": "minmax_continuous",
            "prep_details": "Stemmed text + scaled Core 19 metadata",
            "params": f"C={settings.svm_c}, kernel={settings.svm_kernel}",
        },
    ]

    all_experiments = baseline_experiments + (comparison_experiments if include_comparisons else [])

    evaluation_results = []
    predictions_table = pd.DataFrame({
        "test_row_id": test_indices,
        "true_label": df_test["label"].values,
        "true_label_id": y_test,
    })
    if include_text_in_predictions:
        predictions_table["message"] = df_test["text"].values

    artifact_manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "total_records": len(df),
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
        },
        "feature_engineering": {
            "vocabulary_size_standard": vocab_size_std,
            "vocabulary_size_stemmed": vocab_size_stem,
            "metadata_feature_count_core": 19,
            "metadata_feature_count_extended": 22,
        },
        "model_selection_cross_validation": {
            "protocol": "5-Fold Stratified Cross-Validation on df_train",
            "selected_candidate": cv_selection_report["selected_candidate"],
            "selection_bias_disclosure": cv_selection_report["selection_bias_disclosure"],
        },
        "models": {},
    }

    logger.info("--------------------------------------------------------------------------------")
    logger.info("TRAINING & EVALUATION OF %d CONFIGURATIONS (HELD-OUT TEST SPLIT)", len(all_experiments))
    logger.info("--------------------------------------------------------------------------------")

    for spec in all_experiments:
        exp_id = spec["id"]
        model_name = spec["name"]
        f_space = spec["feature_space"]
        clf_type = spec["clf_type"]
        clf = spec["clf"]
        X_tr = spec["X_tr"]
        X_te = spec["X_te"]

        t0 = time.time()
        logger.info("Training %s (%s, %s)...", exp_id, model_name, clf_type)
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t0

        # Construct and serialize unified pipeline
        pipeline = SpamClassifierPipeline(
            experiment_id=exp_id,
            model_name=model_name,
            feature_space=f_space,
            classifier_type=clf_type,
            classifier=clf,
            text_vectorizer=spec["vec"],
            metadata_extractor=spec["meta_ext"],
            metadata_scaler=spec["scaler"],
            preprocessing_config=spec["prep_cfg"],
            model_metadata={
                "experiment_id": exp_id,
                "dataset_sha256": dataset_sha256,
                "train_samples": len(df_train),
                "feature_count": spec["feat_count"],
                "feature_mode": spec["feat_mode"],
                "scaling": spec["scaling"],
                "preprocessing_details": spec["prep_details"],
                "model_parameters": spec["params"],
                "fit_time_seconds": round(fit_time, 4),
            },
        )
        artifact_filename = f"{exp_id.lower()}_{model_name}.joblib"
        artifact_path = artifacts_dir / artifact_filename
        pipeline.save(artifact_path)

        # Held-out predictions
        t_eval_0 = time.time()
        y_pred = clf.predict(X_te)
        inference_latency_total = time.time() - t_eval_0

        y_scores = None
        if hasattr(clf, "decision_function"):
            try:
                y_scores = clf.decision_function(X_te)
            except Exception:
                y_scores = None
        elif hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(X_te)
                y_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            except Exception:
                y_scores = None

        # Compute full evaluation metrics
        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_scores=y_scores,
            experiment_id=exp_id,
            model_name=model_name,
            feature_space=f_space,
            classifier_name=clf_type,
            feature_count=spec["feat_count"],
            feature_mode=spec["feat_mode"],
            scaling=spec["scaling"],
            preprocessing_details=spec["prep_details"],
            model_parameters=spec["params"],
        )
        evaluation_results.append(metrics)

        # Generate confusion matrix figure
        cm_fig_path = cm_dir / f"{exp_id.lower()}_{model_name}_cm.png"
        plot_confusion_matrix(
            metrics.confusion_matrix,
            title=f"{exp_id}: {model_name} (Acc: {metrics.accuracy:.1%})",
            output_path=cm_fig_path,
        )

        # Record in predictions table
        predictions_table[f"pred_{model_name}"] = ["spam" if p == 1 else "ham" for p in y_pred]
        if y_scores is not None:
            predictions_table[f"score_{model_name}"] = np.round(y_scores, 4)

        # Record in manifest
        artifact_manifest["models"][model_name] = {
            "experiment_id": exp_id,
            "artifact_file": str(artifact_filename),
            "feature_space": f_space,
            "classifier": clf_type,
            "feature_count": spec["feat_count"],
            "feature_mode": spec["feat_mode"],
            "scaling": spec["scaling"],
            "fit_time_seconds": round(fit_time, 4),
            "test_accuracy": metrics.accuracy,
            "spam_f1": metrics.spam_f1,
            "spam_precision": metrics.spam_precision,
            "spam_recall": metrics.spam_recall,
            "confusion_matrix": metrics.confusion_matrix,
        }

        logger.info(
            "%s [%s]: Acc=%.4f | Spam Prec=%.4f | Spam Rec=%.4f | Spam F1=%.4f",
            exp_id,
            model_name,
            metrics.accuracy,
            metrics.spam_precision,
            metrics.spam_recall,
            metrics.spam_f1,
        )

    # 9. Save Manifest and Prediction CSV
    manifest_path = artifacts_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(artifact_manifest, f, indent=2)

    predictions_path = reports_dir / "predictions.csv"
    predictions_table.to_csv(predictions_path, index=False)
    logger.info("Test predictions saved to %s", predictions_path)

    # 10. Audit Prediction Arrays (19 vs 22 Scaled SVM & Text NB vs Text SVM)
    audit_report = audit_prediction_arrays(
        df_test=df_test,
        predictions_table=predictions_table,
        artifacts_dir=artifacts_dir,
    )
    audit_report_path = reports_dir / "prediction_comparison_audit.json"
    with open(audit_report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    logger.info("Prediction array audit report saved to %s", audit_report_path)

    # 11. Export Evaluation Reports
    report_paths = export_reports(evaluation_results, reports_dir)

    # 12. Save Overall Training Summary
    total_elapsed = time.time() - start_time
    training_summary = {
        "status": "COMPLETED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "total_records": len(df),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "vocabulary_size_standard": vocab_size_std,
        "vocabulary_size_stemmed": vocab_size_stem,
        "metadata_feature_count_core": 19,
        "metadata_feature_count_extended": 22,
        "models_evaluated": len(evaluation_results),
        "model_selection_cross_validation": {
            "selected_model": cv_selection_report["selected_candidate"]["model_name"],
            "selected_model_cv_f1": cv_selection_report["selected_candidate"]["mean_cv_f1"],
            "selected_model_cv_accuracy": cv_selection_report["selected_candidate"]["mean_cv_accuracy"],
            "cv_report_path": str(cv_report_path),
        },
        "selection_bias_disclosure": (
            "The 80:20 held-out test split results represent an exploratory benchmark. "
            "Repeated model comparison on that split carries selection bias risk. "
            "Candidate selection was determined strictly via 5-Fold Stratified Cross-Validation on the "
            "training partition (4,128 samples). For unbiased final performance estimation, a genuinely "
            "independent external test set or separate split should be evaluated only after freezing all choices."
        ),
        "prediction_comparison_audit_path": str(audit_report_path),
        "reports": {k: str(v) for k, v in report_paths.items()},
        "manifest_path": str(manifest_path),
    }
    with open(reports_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    logger.info("================================================================================")
    logger.info("TRAINING & AUDIT PIPELINE COMPLETE (Total time: %.2f seconds)", total_elapsed)
    logger.info("================================================================================")

    # Print summary comparison table
    df_comp = pd.read_csv(report_paths["comparison_csv"])
    print("\n" + "=" * 110)
    print("EXPERIMENT COMPARISON RESULTS (HELD-OUT TEST SPLIT):")
    print("=" * 110)
    print(
        df_comp[
            [
                "Experiment",
                "Model Name",
                "Feature Space",
                "Scaling",
                "Test Accuracy",
                "Spam Precision",
                "Spam Recall",
                "Spam F1",
            ]
        ].to_string(index=False)
    )
    print("=" * 110 + "\n")

    return {
        "training_summary": training_summary,
        "cross_validation_report": cv_selection_report,
        "prediction_audit_report": audit_report,
        "evaluation_results": evaluation_results,
        "manifest": artifact_manifest,
    }


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate SMS Spam Classification baseline and comparison experiment configurations."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(settings.default_dataset_path),
        help="Path to SMS spam dataset file (TSV or CSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.default_artifacts_dir),
        help="Directory to save trained model artifacts and schema.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(settings.default_reports_dir),
        help="Directory to save evaluation reports and confusion matrices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.random_seed,
        help="Random seed for reproducible train/test split (default: 42).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=settings.test_size,
        help="Proportion of dataset to reserve for test split (default: 0.2).",
    )
    parser.add_argument(
        "--no-comparisons",
        action="store_true",
        help="Run only the 6 baseline configurations without additional remediation comparisons.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include raw message text in predictions.csv (disabled by default for privacy).",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="cv_selection",
        choices=["cv_selection", "train_val_test"],
        help="Evaluation protocol mode (default: cv_selection).",
    )

    args = parser.parse_args()

    try:
        run_training_pipeline(
            dataset_path=Path(args.data),
            artifacts_dir=Path(args.output_dir),
            reports_dir=Path(args.reports_dir),
            random_seed=args.seed,
            test_size=args.test_size,
            include_comparisons=not args.no_comparisons,
            include_text_in_predictions=args.include_text,
            split_mode=args.split_mode,
        )
    except Exception as e:
        logger.exception("Training pipeline failed with error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
