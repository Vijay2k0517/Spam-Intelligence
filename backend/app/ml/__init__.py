"""Machine Learning pipeline modules for SMS Spam Classification."""

from app.ml.data_loader import load_and_validate_dataset, DatasetSummary
from app.ml.preprocessing import preprocess_message, DualStreamText
from app.ml.features import (
    TextVectorizerWrapper,
    MetadataFeatureExtractor,
    HybridFeaturePipeline,
)
from app.ml.models import SpamClassifierPipeline, build_experiment_pipeline
from app.ml.inference import predict_message, predict_batch

__all__ = [
    "load_and_validate_dataset",
    "DatasetSummary",
    "preprocess_message",
    "DualStreamText",
    "TextVectorizerWrapper",
    "MetadataFeatureExtractor",
    "HybridFeaturePipeline",
    "SpamClassifierPipeline",
    "build_experiment_pipeline",
    "predict_message",
    "predict_batch",
]
