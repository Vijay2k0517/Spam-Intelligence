"""Unit tests for textual, metadata, and hybrid feature engineering modules."""

import numpy as np
import pytest
from scipy import sparse
from app.ml.features import (
    CORE_BINARY_FEATURES,
    CORE_CONTINUOUS_FEATURES,
    HybridFeaturePipeline,
    MetadataFeatureExtractor,
    TextVectorizerWrapper,
    extract_metadata_dict,
)


def test_text_vectorizer_training_isolation():
    """Verify CountVectorizer fits strictly on training partition and handles OOV words."""
    train_texts = ["free prize call now", "hello friend how are you"]
    test_texts = ["free prize unknown_word_xyz", "completely unseen vocabulary"]

    vec = TextVectorizerWrapper()
    X_train = vec.fit_transform(train_texts)

    assert vec.is_fitted
    assert isinstance(X_train, sparse.csr_matrix)
    assert vec.vocabulary_size > 0

    # Transform test partition with learned vocabulary
    X_test = vec.transform(test_texts)
    assert isinstance(X_test, sparse.csr_matrix)
    assert X_test.shape[1] == vec.vocabulary_size

    # OOV tokens produce 0 counts without error
    # "completely unseen vocabulary" has 0 overlap with train_texts
    assert X_test[1].nnz == 0


def test_metadata_continuous_feature_family():
    """Verify continuous metadata calculations."""
    msg = "Hello World! How are you doing today? Great!!!"
    d = extract_metadata_dict(msg)

    assert d["length"] == len(msg)
    assert d["word_count"] == 8
    assert d["unique_words_count"] == 8
    assert d["num_capitalized_words"] == 4  # Hello, World!, How, Great!!!
    assert d["exclam_cnt"] == 4  # 1 after World, 3 after Great
    assert d["quest_cnt"] == 1
    assert d["punct_cnt"] >= 5


def test_metadata_binary_flags_detection():
    """Verify detection of phone numbers, currency symbols, URLs, all caps, and keywords."""
    spam_sample = (
        "URGENT: Claim your free £500 gift. Call 08452810075 or text REPLI to "
        "stop at http://claim-now.org from your mobil."
    )
    d = extract_metadata_dict(spam_sample)

    assert d["has_mobile_number"] == 1.0
    assert d["has_money_symbol"] == 1.0
    assert d["has_link"] == 1.0
    assert d["has_caps_word"] == 1.0  # URGENT, REPLI
    assert d["spam_call"] == 1.0
    assert d["spam_claim"] == 1.0
    assert d["spam_free"] == 1.0
    assert d["spam_mobil"] == 1.0
    assert d["spam_text"] == 1.0
    assert d["spam_repli"] == 1.0
    assert d["spam_stop"] == 1.0

    # Clean ham message should have flags as 0.0
    ham_sample = "Are we meeting at the library at 3pm?"
    d_ham = extract_metadata_dict(ham_sample)
    assert d_ham["has_mobile_number"] == 0.0
    assert d_ham["has_money_symbol"] == 0.0
    assert d_ham["has_link"] == 0.0
    assert d_ham["spam_call"] == 0.0
    assert d_ham["spam_claim"] == 0.0


def test_metadata_edge_cases_finite_numbers():
    """Verify that edge cases (empty, punctuation-only, digit-only) produce strictly finite values."""
    edge_cases = [
        "",
        "   ",
        "!!!!!!!!!!????????",
        "12345678901",
        "👍🎉🔥 emoji text only",
        "A" * 500,
    ]

    for ec in edge_cases:
        d = extract_metadata_dict(ec, include_sentiment=True)
        for key, val in d.items():
            assert isinstance(val, (int, float)), f"Feature {key} is not numeric"
            assert np.isfinite(val), f"Feature {key} is not finite for input: {ec!r}"


def test_metadata_schema_stability_and_order():
    """Verify schema column names and exact stable ordering."""
    extractor = MetadataFeatureExtractor(mode="paper_core")
    names = extractor.get_feature_names()

    assert len(names) == 19
    assert names[:8] == CORE_CONTINUOUS_FEATURES
    assert names[8:] == CORE_BINARY_FEATURES

    matrix = extractor.transform(["First message", "Second message with 08001234567"])
    assert matrix.shape == (2, 19)
    assert isinstance(matrix, np.ndarray)


def test_hybrid_feature_pipeline():
    """Verify hybrid sparse matrix construction via hstack."""
    train_raw = ["Call 08001234567 now for £1000", "Hello there friend"]
    train_clean = ["call 08001234567 now for £1000", "hello there friend"]

    hybrid_pipe = HybridFeaturePipeline()
    X_hybrid = hybrid_pipe.fit_transform(train_raw, train_clean)

    assert isinstance(X_hybrid, sparse.csr_matrix)
    counts = hybrid_pipe.feature_counts
    assert counts["textual_features"] > 0
    assert counts["metadata_features"] == 19
    assert X_hybrid.shape[1] == counts["total_hybrid_features"]
    assert X_hybrid.shape[0] == 2


def test_metadata_scaler_selective_continuous_scaling():
    """Verify that MetadataScaler transforms continuous features while leaving binary flags intact."""
    from app.ml.features import MetadataScaler, MetadataFeatureExtractor

    extractor = MetadataFeatureExtractor(mode="paper_core")
    texts = [
        "Normal message with some words here.",
        "URGENT: Call 08001234567 to claim £1000 now! Text STOP",
    ]
    meta = extractor.transform(texts)
    cont_idx = extractor.continuous_indices
    bin_idx = extractor.binary_indices

    # 1. MinMaxScaler
    scaler_mm = MetadataScaler("minmax")
    scaler_mm.fit(meta, cont_idx)
    scaled_mm = scaler_mm.transform(meta, cont_idx)

    # Continuous features scaled to [0, 1]
    assert np.all(scaled_mm[:, cont_idx] >= 0.0)
    assert np.all(scaled_mm[:, cont_idx] <= 1.0)
    # Binary indicators remain completely untouched
    np.testing.assert_array_equal(scaled_mm[:, bin_idx], meta[:, bin_idx])

    # 2. Median thresholding
    scaler_med = MetadataScaler("median_threshold")
    scaler_med.fit(meta, cont_idx)
    scaled_med = scaler_med.transform(meta, cont_idx)
    # All features in median thresholding are binary flags in {0.0, 1.0}
    assert set(np.unique(scaled_med[:, cont_idx])).issubset({0.0, 1.0})
    np.testing.assert_array_equal(scaled_med[:, bin_idx], meta[:, bin_idx])


def test_metadata_extended_22_features():
    """Verify that mode='paper_extended' outputs exactly 22 features including VADER sentiment."""
    from app.ml.features import MetadataFeatureExtractor

    ext = MetadataFeatureExtractor(mode="paper_extended")
    assert len(ext.feature_names) == 22
    assert len(ext.continuous_indices) == 11
    assert len(ext.binary_indices) == 11
    assert "vader_compound" in ext.feature_names
    assert "vader_positive" in ext.feature_names
    assert "vader_negative" in ext.feature_names

    sample = ext.transform_single("Great news! Congratulations on your prize! Have a wonderful day.")
    assert sample.shape == (22,)
    assert all(np.isfinite(sample))


def test_metadata_scaler_schema_column_mapping_core_and_extended():
    """Verify that MetadataScaler resolves continuous indices via schema column mapping for both 19 and 22 features."""
    from app.ml.features import (
        CORE_BINARY_FEATURES,
        CORE_CONTINUOUS_FEATURES,
        EXTENDED_SENTIMENT_FEATURES,
        MetadataScaler,
    )

    # 1. Core 19 schema
    core_names = CORE_CONTINUOUS_FEATURES + CORE_BINARY_FEATURES
    core_cont = CORE_CONTINUOUS_FEATURES
    scaler_core = MetadataScaler(
        scaler_type="minmax",
        feature_names=core_names,
        continuous_features=core_cont,
        mode="paper_core",
    )
    assert scaler_core.schema_column_map is not None
    assert scaler_core.schema_column_map["num_capitalized_words"] == 0
    assert scaler_core.schema_column_map["quest_cnt"] == 7
    assert scaler_core.schema_column_map["has_mobile_number"] == 8
    assert scaler_core.continuous_indices == [0, 1, 2, 3, 4, 5, 6, 7]

    # 2. Extended 22 schema
    ext_names = CORE_CONTINUOUS_FEATURES + CORE_BINARY_FEATURES + EXTENDED_SENTIMENT_FEATURES
    ext_cont = CORE_CONTINUOUS_FEATURES + EXTENDED_SENTIMENT_FEATURES
    scaler_ext = MetadataScaler(
        scaler_type="minmax",
        feature_names=ext_names,
        continuous_features=ext_cont,
        mode="paper_extended",
    )
    assert scaler_ext.schema_column_map is not None
    assert scaler_ext.schema_column_map["vader_compound"] == 19
    assert scaler_ext.schema_column_map["vader_positive"] == 20
    assert scaler_ext.schema_column_map["vader_negative"] == 21
    # Ensure continuous indices include 0..7 and 19..21, without touching binary 8..18
    assert scaler_ext.continuous_indices == [0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21]

    # Dynamic fallback check if passed raw array without explicit indices
    dummy_core = np.zeros((3, 19))
    assert scaler_core._resolve_indices(dummy_core) == [0, 1, 2, 3, 4, 5, 6, 7]
    dummy_ext = np.zeros((3, 22))
    assert scaler_ext._resolve_indices(dummy_ext) == [0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21]


def test_metadata_scaler_vader_compound_negative_bounds():
    """Verify that negative VADER compound scores scale properly to [0, 1] while binary indicators are unchanged."""
    from app.ml.features import MetadataFeatureExtractor, MetadataScaler

    extractor = MetadataFeatureExtractor(mode="paper_extended")
    # Synthetic samples designed to produce negative, neutral, and positive compound sentiment
    texts = [
        "HORRIBLE terrible disaster bad sad angry scam!",  # Highly negative compound (< -0.5)
        "Okay sounds normal let us check the schedule.",  # Neutral compound (~ 0.0)
        "URGENT: Call 08001234567 to claim £1000 prize!",  # Spam with binary flags + positive
        "Wonderful love fantastic congratulations great!",  # Highly positive compound (> 0.5)
    ]
    meta_raw = extractor.transform(texts)

    vader_comp_idx = extractor.feature_names.index("vader_compound")
    # Verify that raw feature matrix contains at least one negative compound score
    assert any(meta_raw[:, vader_comp_idx] < 0.0), "Expected negative VADER compound in raw features"

    scaler = MetadataScaler(
        scaler_type="minmax",
        feature_names=extractor.feature_names,
        continuous_features=extractor.continuous_features,
        mode="paper_extended",
    )
    scaler.fit(meta_raw)
    meta_scaled = scaler.transform(meta_raw)

    # 1. All continuous columns (including VADER compound) must be bounded in [0.0, 1.0]
    for c_idx in extractor.continuous_indices:
        assert np.all(meta_scaled[:, c_idx] >= 0.0 - 1e-9)
        assert np.all(meta_scaled[:, c_idx] <= 1.0 + 1e-9)

    # 2. Binary indicators (indices 8..18) must remain strictly in {0.0, 1.0} and match raw values exactly
    bin_indices = extractor.binary_indices
    np.testing.assert_array_equal(meta_scaled[:, bin_indices], meta_raw[:, bin_indices])
    assert set(np.unique(meta_scaled[:, bin_indices])).issubset({0.0, 1.0})


def test_pipeline_training_inference_parity_both_schemas():
    """Verify that training and inference apply identical transformations across both core and extended schemas."""
    from app.ml.models import build_experiment_pipeline

    sample_msgs = [
        "Call 08001234567 now for your free £500 cash gift! Reply STOP",
        "Hey, are we having lunch today at 12?",
        "URGENT warning: account suspended visit http://alert-bank.com",
    ]

    for model_name, f_mode in [
        ("hybrid_svm_scaled_minmax", "paper_core"),
        ("hybrid_svm_22_scaled", "paper_extended"),
    ]:
        pipe = build_experiment_pipeline(
            model_name=model_name,
            feature_mode=f_mode,
            scaler_type="minmax",
        )
        # Fit vectorizer
        pipe.text_vectorizer.fit(sample_msgs)
        # Fit scaler on metadata
        meta_raw = pipe.metadata_extractor.transform(sample_msgs)
        pipe.metadata_scaler.fit(meta_raw)

        # Transform 1 (e.g. at training time)
        X_train_feat = pipe.transform_features(sample_msgs)

        # Transform 2 (e.g. at inference time on same messages)
        X_infer_feat = pipe.transform_features(sample_msgs)

        # Exact parity check
        np.testing.assert_array_equal(
            X_train_feat.toarray() if hasattr(X_train_feat, "toarray") else X_train_feat,
            X_infer_feat.toarray() if hasattr(X_infer_feat, "toarray") else X_infer_feat,
        )

        expected_meta_dim = 19 if f_mode == "paper_core" else 22
        assert X_train_feat.shape[1] == pipe.text_vectorizer.vocabulary_size + expected_meta_dim


