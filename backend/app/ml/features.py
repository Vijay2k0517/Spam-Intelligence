"""Feature engineering module for Textual, Non-Textual (Metadata), and Hybrid representations.

Implements:
1. Textual Representation: Scikit-learn CountVectorizer wrapper.
2. Metadata Representation: Handcrafted structural, stylistic, and semantic features (Core 19 and Extended 22).
3. Metadata Scaler: Scaler operating strictly on continuous metadata columns while preserving binary indicators.
4. Hybrid Representation: Memory-efficient sparse horizontal concatenation (hstack) of lexical BoW and metadata.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import string
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

# Precompiled regular expressions for metadata extraction
PHONE_NUMBER_PATTERN = re.compile(
    r"(?:(?:\+?(\d{1,3}))?[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}|\b\d{10,11}\b|\b0\d{10}\b|\b08[0-9]{8,9}\b|\b09[0-9]{8,9}\b"
)

CURRENCY_PATTERN = re.compile(
    r"[$£€¥₹]|(?:\b(?:gbp|usd|eur|inr|cents?|bucks?)\b)",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|org|net|co\.uk|info|biz|tv|cc|ly|me)\b",
    re.IGNORECASE,
)

ALL_CAPS_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")

KEYWORD_PATTERNS = {
    "spam_call": re.compile(r"\bcall\b", re.IGNORECASE),
    "spam_mobil": re.compile(r"\bmobil(?:e)?\b", re.IGNORECASE),
    "spam_free": re.compile(r"\bfree\b", re.IGNORECASE),
    "spam_claim": re.compile(r"\bclaim\b", re.IGNORECASE),
    "spam_text": re.compile(r"\btext\b", re.IGNORECASE),
    "spam_repli": re.compile(r"\brepli(?:ed|es)?\b|\breply\b", re.IGNORECASE),
    "spam_stop": re.compile(r"\bstop\b", re.IGNORECASE),
}

_VADER_ANALYZER = None


def _get_vader_analyzer():
    """Lazy initialization of NLTK VADER sentiment analyzer."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.debug("VADER SentimentIntensityAnalyzer unavailable: %s", e)
            _VADER_ANALYZER = False
    return _VADER_ANALYZER if _VADER_ANALYZER is not False else None


# Ordered feature schema definitions matching Reference Paper Figure 5a and 5b
CORE_CONTINUOUS_FEATURES = [
    "num_capitalized_words",
    "length",
    "unique_words_count",
    "avg_word_length",
    "word_count",
    "exclam_cnt",
    "punct_cnt",
    "quest_cnt",
]

CORE_BINARY_FEATURES = [
    "has_mobile_number",
    "has_money_symbol",
    "spam_call",
    "spam_mobil",
    "spam_free",
    "spam_claim",
    "has_link",
    "spam_text",
    "spam_repli",
    "spam_stop",
    "has_caps_word",
]

EXTENDED_SENTIMENT_FEATURES = [
    "vader_compound",
    "vader_positive",
    "vader_negative",
]


def extract_metadata_dict(
    text: str,
    include_sentiment: bool = False,
) -> Dict[str, float]:
    """Extract named metadata features for a single raw SMS string.

    Guarantees deterministic output and finite float values even on empty,
    punctuation-only, or whitespace-only inputs.

    Args:
        text: Raw SMS message string.
        include_sentiment: Whether to compute VADER sentiment scores.

    Returns:
        Dictionary mapping feature name to float value.
    """
    raw = str(text) if text is not None else ""
    raw_len = len(raw)

    words = raw.split()
    word_count = len(words)
    unique_words_count = len(set(w.lower() for w in words)) if word_count > 0 else 0

    if word_count > 0:
        total_word_chars = sum(len(w) for w in words)
        avg_word_length = total_word_chars / float(word_count)
    else:
        avg_word_length = 0.0

    num_cap_words = sum(1 for w in words if w and (w[0].isupper() or w.isupper()))

    exclam_cnt = float(raw.count("!"))
    quest_cnt = float(raw.count("?"))
    punct_cnt = float(sum(1 for c in raw if c in string.punctuation))

    has_mobile = 1.0 if PHONE_NUMBER_PATTERN.search(raw) else 0.0
    has_money = 1.0 if CURRENCY_PATTERN.search(raw) else 0.0
    has_link = 1.0 if URL_PATTERN.search(raw) else 0.0
    has_caps = 1.0 if ALL_CAPS_WORD_PATTERN.search(raw) else 0.0

    call_flag = 1.0 if KEYWORD_PATTERNS["spam_call"].search(raw) else 0.0
    mobil_flag = 1.0 if KEYWORD_PATTERNS["spam_mobil"].search(raw) else 0.0
    free_flag = 1.0 if KEYWORD_PATTERNS["spam_free"].search(raw) else 0.0
    claim_flag = 1.0 if KEYWORD_PATTERNS["spam_claim"].search(raw) else 0.0
    text_flag = 1.0 if KEYWORD_PATTERNS["spam_text"].search(raw) else 0.0
    repli_flag = 1.0 if KEYWORD_PATTERNS["spam_repli"].search(raw) else 0.0
    stop_flag = 1.0 if KEYWORD_PATTERNS["spam_stop"].search(raw) else 0.0

    feat: Dict[str, float] = {
        # 8 Continuous Features
        "num_capitalized_words": float(num_cap_words),
        "length": float(raw_len),
        "unique_words_count": float(unique_words_count),
        "avg_word_length": float(avg_word_length),
        "word_count": float(word_count),
        "exclam_cnt": exclam_cnt,
        "punct_cnt": punct_cnt,
        "quest_cnt": quest_cnt,
        # 11 Binary Flags
        "has_mobile_number": has_mobile,
        "has_money_symbol": has_money,
        "spam_call": call_flag,
        "spam_mobil": mobil_flag,
        "spam_free": free_flag,
        "spam_claim": claim_flag,
        "has_link": has_link,
        "spam_text": text_flag,
        "spam_repli": repli_flag,
        "spam_stop": stop_flag,
        "has_caps_word": has_caps,
    }

    if include_sentiment:
        analyzer = _get_vader_analyzer()
        if analyzer is not None:
            scores = analyzer.polarity_scores(raw)
            feat["vader_compound"] = float(scores["compound"])
            feat["vader_positive"] = float(scores["pos"])
            feat["vader_negative"] = float(scores["neg"])
        else:
            feat["vader_compound"] = 0.0
            feat["vader_positive"] = 0.0
            feat["vader_negative"] = 0.0

    return feat


class MetadataScaler:
    """Selective metadata scaler operating strictly on continuous columns while preserving binary indicators.

    Supports:
    - 'none': Passthrough
    - 'minmax': Scikit-learn MinMaxScaler scaling continuous features to [0, 1]
    - 'standard': Scikit-learn StandardScaler zero-mean unit-variance
    - 'median_threshold': Informative binarization thresholding continuous features at training split medians

    Uses feature names and schema-specific column mapping to prevent hardcoded index errors
    between core 19-feature and extended 22-feature representations.
    """

    def __init__(
        self,
        scaler_type: str = "none",
        feature_names: Optional[Sequence[str]] = None,
        continuous_features: Optional[Sequence[str]] = None,
        mode: Optional[str] = None,
    ):
        self.scaler_type = scaler_type.lower()
        self.mode = mode
        self.feature_names = list(feature_names) if feature_names else None
        self.continuous_features = list(continuous_features) if continuous_features else None
        self.continuous_indices: Optional[List[int]] = None
        self.schema_column_map: Optional[Dict[str, int]] = None

        if self.feature_names:
            self.schema_column_map = {name: idx for idx, name in enumerate(self.feature_names)}
            if self.continuous_features:
                self.continuous_indices = [
                    self.schema_column_map[name]
                    for name in self.continuous_features
                    if name in self.schema_column_map
                ]

        self.scaler = None
        self.medians: Optional[np.ndarray] = None
        self.is_fitted = False

    def _resolve_indices(
        self,
        X_meta: np.ndarray,
        continuous_indices: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Resolve continuous column indices via explicit argument or schema column mapping."""
        if continuous_indices is not None:
            return list(continuous_indices)
        if self.continuous_indices is not None:
            return list(self.continuous_indices)
        # If schema column map is present, resolve from continuous_features
        if self.schema_column_map and self.continuous_features:
            return [
                self.schema_column_map[name]
                for name in self.continuous_features
                if name in self.schema_column_map
            ]
        # Fallback based on column count if neither is provided
        n_cols = X_meta.shape[1]
        if n_cols == 19:
            # Core schema: first 8 columns are continuous
            return list(range(len(CORE_CONTINUOUS_FEATURES)))
        elif n_cols == 22:
            # Extended schema: first 8 columns + last 3 sentiment columns
            return list(range(len(CORE_CONTINUOUS_FEATURES))) + [19, 20, 21]
        return list(range(n_cols))

    def fit(
        self,
        X_meta: np.ndarray,
        continuous_indices: Optional[Sequence[int]] = None,
    ) -> "MetadataScaler":
        """Fit scaler parameters exclusively on training split continuous columns."""
        resolved_indices = self._resolve_indices(X_meta, continuous_indices)
        self.continuous_indices = resolved_indices

        if self.scaler_type == "none" or len(resolved_indices) == 0:
            self.is_fitted = True
            return self

        X_cont = X_meta[:, resolved_indices]
        if self.scaler_type == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            # Scales continuous values (including potentially negative VADER compound) to [0, 1]
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            self.scaler.fit(X_cont)
        elif self.scaler_type == "standard":
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            self.scaler.fit(X_cont)
        elif self.scaler_type == "median_threshold":
            self.medians = np.median(X_cont, axis=0)
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        self.is_fitted = True
        return self

    def transform(
        self,
        X_meta: np.ndarray,
        continuous_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Transform continuous columns of metadata array, leaving binary flags untouched."""
        resolved_indices = (
            list(continuous_indices)
            if continuous_indices is not None
            else self.continuous_indices
        )
        if resolved_indices is None:
            resolved_indices = self._resolve_indices(X_meta, None)

        if self.scaler_type == "none" or not self.is_fitted or len(resolved_indices) == 0:
            return np.copy(X_meta)

        X_out = np.copy(X_meta)
        X_cont = X_meta[:, resolved_indices]

        if self.scaler_type in ("minmax", "standard") and self.scaler is not None:
            X_out[:, resolved_indices] = self.scaler.transform(X_cont)
        elif self.scaler_type == "median_threshold" and self.medians is not None:
            X_out[:, resolved_indices] = (X_cont > self.medians).astype(np.float64)

        return X_out


class MetadataFeatureExtractor:
    """Extractor for handcrafted metadata features with stable schema and array output."""

    def __init__(self, mode: str = "paper_core"):
        """Initialize metadata extractor.

        Args:
            mode: "paper_core" (19 features from Figure 5) or "paper_extended" (22 features with VADER).
        """
        self.mode = mode
        self.include_sentiment = mode == "paper_extended"
        if self.include_sentiment:
            self.feature_names = (
                CORE_CONTINUOUS_FEATURES + CORE_BINARY_FEATURES + EXTENDED_SENTIMENT_FEATURES
            )
            self._continuous_indices = list(range(len(CORE_CONTINUOUS_FEATURES))) + [
                len(CORE_CONTINUOUS_FEATURES) + len(CORE_BINARY_FEATURES) + i
                for i in range(len(EXTENDED_SENTIMENT_FEATURES))
            ]
            self._binary_indices = list(
                range(len(CORE_CONTINUOUS_FEATURES), len(CORE_CONTINUOUS_FEATURES) + len(CORE_BINARY_FEATURES))
            )
        else:
            self.feature_names = CORE_CONTINUOUS_FEATURES + CORE_BINARY_FEATURES
            self._continuous_indices = list(range(len(CORE_CONTINUOUS_FEATURES)))
            self._binary_indices = list(
                range(len(CORE_CONTINUOUS_FEATURES), len(self.feature_names))
            )

    @property
    def continuous_indices(self) -> List[int]:
        """Indices of continuous numerical columns in feature array."""
        return list(self._continuous_indices)

    @property
    def binary_indices(self) -> List[int]:
        """Indices of binary indicator columns in feature array."""
        return list(self._binary_indices)

    @property
    def continuous_features(self) -> List[str]:
        """Names of continuous numerical features in schema."""
        return [self.feature_names[i] for i in self._continuous_indices]

    @property
    def binary_features(self) -> List[str]:
        """Names of binary indicator features in schema."""
        return [self.feature_names[i] for i in self._binary_indices]

    def transform_single(self, text: str) -> np.ndarray:
        """Transform a single text into a 1D numpy feature vector."""
        d = extract_metadata_dict(text, include_sentiment=self.include_sentiment)
        return np.array([d[name] for name in self.feature_names], dtype=np.float64)

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """Transform an iterable of texts into a 2D numpy array [N, num_features]."""
        rows = [self.transform_single(t) for t in texts]
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        return np.vstack(rows)

    def get_feature_names(self) -> List[str]:
        """Return list of ordered feature names."""
        return list(self.feature_names)

    def get_schema(self) -> Dict[str, Any]:
        """Return schema metadata dictionary for serialization."""
        return {
            "mode": self.mode,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "continuous_features": (
                CORE_CONTINUOUS_FEATURES + EXTENDED_SENTIMENT_FEATURES
                if self.include_sentiment
                else CORE_CONTINUOUS_FEATURES
            ),
            "binary_features": CORE_BINARY_FEATURES,
            "sentiment_features": EXTENDED_SENTIMENT_FEATURES if self.include_sentiment else [],
            "continuous_indices": self.continuous_indices,
            "binary_indices": self.binary_indices,
        }


class TextVectorizerWrapper:
    """Scikit-learn CountVectorizer wrapper for textual representation.

    Ensures the vocabulary is fitted ONLY on training data, returns CSR sparse matrix,
    and handles out-of-vocabulary words deterministically.
    """

    def __init__(
        self,
        lowercase: bool = True,
        token_pattern: str = r"(?u)\b\w+\b",
        min_df: int = 1,
    ):
        self.vectorizer = CountVectorizer(
            lowercase=lowercase,
            token_pattern=token_pattern,
            min_df=min_df,
        )
        self.is_fitted = False

    def fit(self, texts: Iterable[str]) -> "TextVectorizerWrapper":
        """Fit CountVectorizer exclusively on training messages."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info("CountVectorizer fitted: vocabulary size = %d", len(self.vectorizer.vocabulary_))
        return self

    def transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        """Transform texts into sparse CSR matrix using learned vocabulary."""
        if not self.is_fitted:
            raise RuntimeError("TextVectorizerWrapper must be fitted before transform.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        """Fit on texts and return CSR sparse matrix."""
        return self.fit(texts).transform(texts)

    @property
    def vocabulary_size(self) -> int:
        """Return size of learned vocabulary."""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)

    def get_feature_names(self) -> List[str]:
        """Return array of feature/token names."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


class HybridFeaturePipeline:
    """Combines TextVectorizer and MetadataFeatureExtractor with optional scaling via sparse hstack."""

    def __init__(
        self,
        text_vectorizer: Optional[TextVectorizerWrapper] = None,
        metadata_extractor: Optional[MetadataFeatureExtractor] = None,
        metadata_scaler: Optional[MetadataScaler] = None,
    ):
        self.text_vectorizer = text_vectorizer or TextVectorizerWrapper()
        self.metadata_extractor = metadata_extractor or MetadataFeatureExtractor()
        self.metadata_scaler = metadata_scaler or MetadataScaler("none")
        self.is_fitted = False

    def fit(self, raw_texts: Sequence[str], cleaned_texts: Sequence[str]) -> "HybridFeaturePipeline":
        """Fit textual vectorizer and metadata scaler exclusively on training texts."""
        self.text_vectorizer.fit(cleaned_texts)
        meta_dense = self.metadata_extractor.transform(raw_texts)
        self.metadata_scaler.fit(meta_dense, self.metadata_extractor.continuous_indices)
        self.is_fitted = True
        return self

    def transform(
        self,
        raw_texts: Sequence[str],
        cleaned_texts: Sequence[str],
    ) -> sparse.csr_matrix:
        """Transform texts into combined sparse CSR matrix with scaled metadata."""
        if not self.is_fitted:
            raise RuntimeError("HybridFeaturePipeline must be fitted before transform.")

        text_sparse = self.text_vectorizer.transform(cleaned_texts)
        meta_dense = self.metadata_extractor.transform(raw_texts)
        meta_scaled = self.metadata_scaler.transform(meta_dense, self.metadata_extractor.continuous_indices)
        meta_sparse = sparse.csr_matrix(meta_scaled)

        # Horizontal sparse concatenation
        hybrid_sparse = sparse.hstack([text_sparse, meta_sparse], format="csr")
        return hybrid_sparse

    def fit_transform(
        self,
        raw_texts: Sequence[str],
        cleaned_texts: Sequence[str],
    ) -> sparse.csr_matrix:
        """Fit on training texts and return combined CSR matrix."""
        return self.fit(raw_texts, cleaned_texts).transform(raw_texts, cleaned_texts)

    @property
    def feature_counts(self) -> Dict[str, int]:
        """Return breakdown of feature counts."""
        text_dim = self.text_vectorizer.vocabulary_size
        meta_dim = len(self.metadata_extractor.get_feature_names())
        return {
            "textual_features": text_dim,
            "metadata_features": meta_dim,
            "total_hybrid_features": text_dim + meta_dim,
        }
