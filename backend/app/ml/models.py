"""Model definitions and unified pipeline serialization for all experiment configurations."""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC

from app.ml.features import (
    MetadataFeatureExtractor,
    MetadataScaler,
    TextVectorizerWrapper,
)
from app.ml.preprocessing import preprocess_message

logger = logging.getLogger(__name__)

FeatureSpaceType = Literal["textual", "metadata", "hybrid"]
ClassifierType = Literal["BernoulliNB", "MultinomialNB", "SVC", "SVC_Linear"]


class SpamClassifierPipeline:
    """Unified, end-to-end inference pipeline for SMS spam classification.

    Encapsulates preprocessing, feature extraction/vectorization, scaling, and trained estimator
    to guarantee zero inference-training skew and self-contained artifact serialization.
    """

    def __init__(
        self,
        experiment_id: str,
        model_name: str,
        feature_space: FeatureSpaceType,
        classifier_type: ClassifierType,
        classifier: Any,
        text_vectorizer: Optional[TextVectorizerWrapper] = None,
        metadata_extractor: Optional[MetadataFeatureExtractor] = None,
        metadata_scaler: Optional[MetadataScaler] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.feature_space = feature_space
        self.classifier_type = classifier_type
        self.classifier = classifier
        self.text_vectorizer = text_vectorizer
        self.metadata_extractor = metadata_extractor
        self.metadata_scaler = metadata_scaler
        self.preprocessing_config = preprocessing_config or {
            "lowercase": True,
            "remove_control_chars": True,
            "isolate_urls": True,
            "strip_punctuation": False,
            "remove_stopwords": False,
            "apply_stemming": False,
        }
        self.model_metadata = model_metadata or {}

    def transform_features(self, messages: Sequence[str]) -> Any:
        """Transform raw input messages into feature space expected by the classifier."""
        # 1. Preprocessing into dual stream
        cleaned_texts = []
        raw_texts = []
        p_cfg = self.preprocessing_config or {}
        for msg in messages:
            ds = preprocess_message(
                msg,
                lowercase=p_cfg.get("lowercase", True),
                remove_control_chars=p_cfg.get("remove_control_chars", True),
                isolate_urls=p_cfg.get("isolate_urls", True),
                strip_punctuation=p_cfg.get("strip_punctuation", False),
                remove_stopwords=p_cfg.get("remove_stopwords", False),
                apply_stemming=p_cfg.get("apply_stemming", False),
            )
            raw_texts.append(ds.raw_text)
            cleaned_texts.append(ds.cleaned_text)

        # 2. Extract according to configured feature space
        if self.feature_space == "textual":
            if self.text_vectorizer is None:
                raise ValueError("Text vectorizer is required for textual pipeline.")
            return self.text_vectorizer.transform(cleaned_texts)

        elif self.feature_space == "metadata":
            if self.metadata_extractor is None:
                raise ValueError("Metadata extractor is required for metadata pipeline.")
            meta_dense = self.metadata_extractor.transform(raw_texts)
            if self.metadata_scaler is not None:
                meta_dense = self.metadata_scaler.transform(
                    meta_dense, self.metadata_extractor.continuous_indices
                )
            return meta_dense

        elif self.feature_space == "hybrid":
            if self.text_vectorizer is None or self.metadata_extractor is None:
                raise ValueError("Both vectorizer and metadata extractor are required for hybrid pipeline.")
            text_sparse = self.text_vectorizer.transform(cleaned_texts)
            meta_dense = self.metadata_extractor.transform(raw_texts)
            if self.metadata_scaler is not None:
                meta_dense = self.metadata_scaler.transform(
                    meta_dense, self.metadata_extractor.continuous_indices
                )
            meta_sparse = sparse.csr_matrix(meta_dense)
            return sparse.hstack([text_sparse, meta_sparse], format="csr")

        else:
            raise ValueError(f"Unknown feature space: {self.feature_space}")

    def predict(self, messages: Sequence[str]) -> np.ndarray:
        """Predict class labels (0 for ham, 1 for spam) for a sequence of messages."""
        X = self.transform_features(messages)
        return self.classifier.predict(X)

    def predict_detailed(
        self, messages: Sequence[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Predict classes, decision scores (SVM), and probabilities (NB if supported).

        Returns:
            Tuple of (predictions, decision_scores, probabilities).
        """
        X = self.transform_features(messages)
        preds = self.classifier.predict(X)

        decision_scores: Optional[np.ndarray] = None
        probabilities: Optional[np.ndarray] = None

        if hasattr(self.classifier, "decision_function"):
            try:
                decision_scores = self.classifier.decision_function(X)
            except Exception:
                decision_scores = None

        if hasattr(self.classifier, "predict_proba"):
            try:
                proba = self.classifier.predict_proba(X)
                if proba.shape[1] > 1:
                    probabilities = proba[:, 1]
                else:
                    probabilities = proba[:, 0]
            except Exception:
                probabilities = None

        return preds, decision_scores, probabilities

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize complete pipeline to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Pipeline %s saved to %s", self.model_name, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SpamClassifierPipeline":
        """Deserialize complete pipeline from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at: {path}")
        pipeline = joblib.load(path)
        if not isinstance(pipeline, SpamClassifierPipeline):
            raise TypeError(f"Loaded object is not a SpamClassifierPipeline: {type(pipeline)}")
        return pipeline


def build_experiment_pipeline(
    model_name: str,
    feature_mode: str = "paper_core",
    random_state: int = 42,
    nb_alpha: float = 1.0,
    svm_c: float = 1.0,
    svm_kernel: str = "rbf",
    scaler_type: str = "none",
) -> SpamClassifierPipeline:
    """Factory creating an unfitted experiment pipeline from configuration."""
    model_configs = {
        "text_nb": ("E1", "textual", "BernoulliNB"),
        "text_svm": ("E2", "textual", "SVC"),
        "metadata_nb": ("E3", "metadata", "BernoulliNB"),
        "metadata_svm": ("E4", "metadata", "SVC"),
        "hybrid_nb": ("E5", "hybrid", "BernoulliNB"),
        "hybrid_svm": ("E6", "hybrid", "SVC"),
        "hybrid_svm_scaled_minmax": ("E6_scaled_minmax", "hybrid", "SVC"),
        "hybrid_svm_scaled_std": ("E6_scaled_std", "hybrid", "SVC"),
        "hybrid_svm_linear": ("E6_linear", "hybrid", "SVC_Linear"),
        "hybrid_mnb": ("E5_mnb", "hybrid", "MultinomialNB"),
        "hybrid_nb_median_thresh": ("E5_median_thresh", "hybrid", "BernoulliNB"),
        "hybrid_nb_22": ("E5_ext22", "hybrid", "BernoulliNB"),
        "hybrid_svm_22_scaled": ("E6_ext22_scaled", "hybrid", "SVC"),
    }

    if model_name not in model_configs:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from: {list(model_configs.keys())}"
        )

    exp_id, f_space, clf_type = model_configs[model_name]

    if clf_type == "BernoulliNB":
        clf = BernoulliNB(alpha=nb_alpha, binarize=0.0)
    elif clf_type == "MultinomialNB":
        clf = MultinomialNB(alpha=nb_alpha)
    elif clf_type == "SVC":
        clf = SVC(C=svm_c, kernel=svm_kernel, random_state=random_state)
    elif clf_type == "SVC_Linear":
        clf = SVC(C=svm_c, kernel="linear", random_state=random_state)
    else:
        raise ValueError(f"Unsupported classifier type: {clf_type}")

    text_vec = TextVectorizerWrapper() if f_space in ("textual", "hybrid") else None
    meta_ext = MetadataFeatureExtractor(mode=feature_mode) if f_space in ("metadata", "hybrid") else None
    meta_scaler = (
        MetadataScaler(
            scaler_type=scaler_type,
            feature_names=meta_ext.feature_names,
            continuous_features=meta_ext.continuous_features,
            mode=feature_mode,
        )
        if f_space in ("metadata", "hybrid") and scaler_type != "none" and meta_ext is not None
        else None
    )

    return SpamClassifierPipeline(
        experiment_id=exp_id,
        model_name=model_name,
        feature_space=f_space,
        classifier_type=clf_type,
        classifier=clf,
        text_vectorizer=text_vec,
        metadata_extractor=meta_ext,
        metadata_scaler=meta_scaler,
        model_metadata={
            "created_at": time.time(),
            "random_state": random_state,
            "feature_mode": feature_mode,
            "scaler_type": scaler_type,
            "classifier_params": (
                {"alpha": nb_alpha}
                if clf_type in ("BernoulliNB", "MultinomialNB")
                else {"C": svm_c, "kernel": svm_kernel if clf_type == "SVC" else "linear", "random_state": random_state}
            ),
        },
    )
