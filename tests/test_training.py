"""Unit tests for model pipeline creation, serialization, metrics, and training."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from app.ml.evaluate import compute_metrics
from app.ml.models import build_experiment_pipeline, SpamClassifierPipeline


@pytest.fixture
def synthetic_training_corpus():
    """Small synthetic corpus strictly for testing pipeline training and serialization."""
    texts = [
        "Are you available for lunch today?",
        "Don't forget to submit the report by 5pm.",
        "See you at the conference tomorrow.",
        "Let's grab a coffee at the campus center.",
        "Can you send me the lecture notes?",
        "URGENT! Call 08452810075 to claim your £1000 prize now!",
        "WINNER! Free entry into weekly £500 draw. Text STOP to opt out.",
        "Claim your complimentary mobile upgrade today: http://claim-gift.biz",
        "Congratulations! You won free tickets. Call 08001234567 immediately.",
        "Exclusive cash prize! Reply REPLI to claim free reward now.",
    ]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 5 ham, 5 spam
    return texts, labels


def test_build_all_six_experiment_pipelines():
    """Verify factory builds all six configurations with correct specifications."""
    configs = [
        ("text_nb", "E1", "textual", "BernoulliNB"),
        ("text_svm", "E2", "textual", "SVC"),
        ("metadata_nb", "E3", "metadata", "BernoulliNB"),
        ("metadata_svm", "E4", "metadata", "SVC"),
        ("hybrid_nb", "E5", "hybrid", "BernoulliNB"),
        ("hybrid_svm", "E6", "hybrid", "SVC"),
    ]

    for model_name, exp_id, f_space, clf_type in configs:
        pipe = build_experiment_pipeline(model_name)
        assert pipe.experiment_id == exp_id
        assert pipe.model_name == model_name
        assert pipe.feature_space == f_space
        assert pipe.classifier_type == clf_type


def test_pipeline_fit_predict_save_load(test_fixtures_paths, synthetic_training_corpus):
    """Verify that all 6 pipelines can fit, predict, save, and reload identically."""
    texts, labels = synthetic_training_corpus
    models = ["text_nb", "text_svm", "metadata_nb", "metadata_svm", "hybrid_nb", "hybrid_svm"]

    save_dir = test_fixtures_paths["out_dir"] / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    for m in models:
        pipe = build_experiment_pipeline(m)
        if pipe.feature_space in ("textual", "hybrid"):
            pipe.text_vectorizer.fit(texts)

        X_fit = pipe.transform_features(texts)
        pipe.classifier.fit(X_fit, labels)

        test_msg = ["Win £500 cash now! Call 08001234567", "Hey are we still meeting?"]
        preds_before = pipe.predict(test_msg)

        # Save and reload
        save_file = save_dir / f"{m}.joblib"
        pipe.save(save_file)
        assert save_file.is_file()

        reloaded = SpamClassifierPipeline.load(save_file)
        preds_after = reloaded.predict(test_msg)

        np.testing.assert_array_equal(preds_before, preds_after)


def test_evaluation_metrics_computation():
    """Verify calculation of accuracy, spam precision, recall, F1, and confusion matrix."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 1, 1, 1]  # 3 TN, 1 FP, 1 FN, 3 TP

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        experiment_id="E1",
        model_name="test_model",
    )

    assert metrics.total_samples == 8
    assert metrics.accuracy == 6 / 8
    assert metrics.confusion_matrix["tn"] == 3
    assert metrics.confusion_matrix["fp"] == 1
    assert metrics.confusion_matrix["fn"] == 1
    assert metrics.confusion_matrix["tp"] == 3
    assert metrics.spam_precision == 3 / 4  # TP / (TP + FP)
    assert metrics.spam_recall == 3 / 4  # TP / (TP + FN)


def test_stratified_splitting_proportions():
    """Verify stratified splitting maintains consistent class balance."""
    y = np.array([0] * 80 + [1] * 20)  # 20% spam
    indices = np.arange(100)

    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    assert len(train_idx) == 80
    assert len(test_idx) == 20

    train_spam_rate = np.mean(y[train_idx])
    test_spam_rate = np.mean(y[test_idx])
    assert pytest.approx(train_spam_rate, abs=0.01) == 0.2
    assert pytest.approx(test_spam_rate, abs=0.01) == 0.2
