"""Unit tests for inference module, schema validation, latency measurement, and batching."""

from pathlib import Path
import pytest
from app.ml.inference import (
    clear_model_cache,
    get_pipeline,
    predict_batch,
    predict_message,
)
from app.ml.models import build_experiment_pipeline


@pytest.fixture(scope="session")
def mock_trained_artifact(test_fixtures_paths):
    """Create a minimal trained hybrid SVM artifact for inference testing."""
    pipe = build_experiment_pipeline("hybrid_svm")
    texts = [
        "Normal meeting with the engineering team at 2pm.",
        "URGENT: Call 08452810075 now to claim your £1000 prize!",
        "Can you send the slides when ready?",
        "Free cash gift waiting. Reply CLAIM to 88088.",
    ]
    labels = [0, 1, 0, 1]

    pipe.text_vectorizer.fit(texts)
    X = pipe.transform_features(texts)
    pipe.classifier.fit(X, labels)

    artifact_dir = test_fixtures_paths["out_dir"] / "mock_inference"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = artifact_dir / "e6_hybrid_svm.joblib"
    pipe.save(artifact_file)
    return artifact_dir


def test_single_message_inference(mock_trained_artifact):
    """Test single message inference with latency and output validation."""
    clear_model_cache()
    res = predict_message(
        message="Free tickets! Call 08001234567 to claim",
        model_name="hybrid_svm",
        artifacts_dir=mock_trained_artifact,
        include_metadata=True,
    )

    assert res.prediction in ("ham", "spam")
    assert isinstance(res.is_spam, bool)
    assert res.model_name == "hybrid_svm"
    assert res.experiment_id == "E6"
    assert res.inference_latency_ms > 0
    assert res.metadata_features is not None
    assert "has_mobile_number" in res.metadata_features
    assert res.decision_score is not None


def test_batch_message_inference(mock_trained_artifact):
    """Test batch message prediction efficiency."""
    clear_model_cache()
    msgs = [
        "Are you free tomorrow morning?",
        "URGENT! Call 09061743811 to receive £500 immediately.",
        "Thanks for the update.",
    ]
    batch_res = predict_batch(
        messages=msgs,
        model_name="hybrid_svm",
        artifacts_dir=mock_trained_artifact,
    )

    assert batch_res.total_messages == 3
    assert len(batch_res.predictions) == 3
    assert batch_res.total_latency_ms > 0
    assert batch_res.model_name == "hybrid_svm"


def test_invalid_input_validation(mock_trained_artifact):
    """Verify informative errors on empty or non-string inputs."""
    with pytest.raises(ValueError):
        predict_message("", artifacts_dir=mock_trained_artifact)

    with pytest.raises(ValueError):
        predict_message("    ", artifacts_dir=mock_trained_artifact)

    with pytest.raises(ValueError):
        predict_batch([], artifacts_dir=mock_trained_artifact)


def test_unrecognized_model_name(mock_trained_artifact):
    """Verify error on unrecognized model identifier."""
    with pytest.raises(ValueError, match="Unknown model name"):
        predict_message("Hello there", model_name="deep_transformer", artifacts_dir=mock_trained_artifact)


def test_missing_artifact_file(test_fixtures_paths):
    """Verify clear error message when artifact does not exist on disk."""
    clear_model_cache()
    empty_dir = test_fixtures_paths["out_dir"] / "empty_artifacts"
    empty_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Model artifact for 'hybrid_svm' not found"):
        predict_message("Hello there", model_name="hybrid_svm", artifacts_dir=empty_dir)


def test_artifact_caching_behavior(mock_trained_artifact):
    """Verify that models are cached in memory and not reloaded from disk on every call."""
    clear_model_cache()
    pipe1 = get_pipeline("hybrid_svm", artifacts_dir=mock_trained_artifact)
    pipe2 = get_pipeline("hybrid_svm", artifacts_dir=mock_trained_artifact)

    assert pipe1 is pipe2
