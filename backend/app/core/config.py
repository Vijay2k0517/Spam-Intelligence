"""Configuration settings for the SMS Spam Classification Engine."""

from pathlib import Path
from typing import Dict, List, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized project and API configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API & Service settings
    app_name: str = "SMS Spam Classification Engine API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    default_model: str = "hybrid_mnb"
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    max_message_length: int = 10000
    max_batch_size: int = 100
    save_raw_messages: bool = False
    log_level: str = "INFO"

    # Project directories
    backend_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent
    )

    # File paths
    default_dataset_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent
        / "data"
        / "raw"
        / "SMSSpamCollection"
    )
    default_artifacts_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent
        / "artifacts"
    )
    default_reports_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent
        / "reports"
    )
    database_file: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
        / "data"
        / "app.db"
    )

    @property
    def database_url(self) -> str:
        """Construct normalized SQLite database URL."""
        db_path = self.database_file.resolve().as_posix()
        return f"sqlite:///{db_path}"

    # Splitting and reproducibility
    random_seed: int = 42
    test_size: float = 0.2
    stratify: bool = True

    # Feature representation modes
    # "paper_core" -> 19 verified features from Figure 5 (8 continuous + 11 binary)
    # "paper_extended" -> 22 features (19 core + 3 VADER sentiment scores)
    feature_mode: Literal["paper_core", "paper_extended"] = "paper_core"


    # Core continuous metadata feature names (matches Figure 5a ordering)
    core_continuous_features: List[str] = [
        "num_capitalized_words",
        "length",
        "unique_words_count",
        "avg_word_length",
        "word_count",
        "exclam_cnt",
        "punct_cnt",
        "quest_cnt",
    ]

    # Core binary metadata feature names (matches Figure 5b ordering)
    core_binary_features: List[str] = [
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

    # Extended sentiment features (Section III-C1 & Algorithm 1)
    sentiment_features: List[str] = [
        "vader_compound",
        "vader_positive",
        "vader_negative",
    ]

    # Model specifications for baseline and remediation configurations
    experiment_registry: Dict[str, Dict[str, str]] = {
        # 6 Primary Baseline Configurations (19 Core Features, Unscaled)
        "text_nb": {
            "id": "E1",
            "name": "Textual BernoulliNB",
            "feature_space": "textual",
            "classifier": "BernoulliNB",
            "artifact_file": "e1_text_nb.joblib",
        },
        "text_svm": {
            "id": "E2",
            "name": "Textual SVM (RBF)",
            "feature_space": "textual",
            "classifier": "SVC",
            "artifact_file": "e2_text_svm.joblib",
        },
        "metadata_nb": {
            "id": "E3",
            "name": "Metadata BernoulliNB (Core 19)",
            "feature_space": "metadata",
            "classifier": "BernoulliNB",
            "artifact_file": "e3_metadata_nb.joblib",
        },
        "metadata_svm": {
            "id": "E4",
            "name": "Metadata SVM (RBF, Unscaled)",
            "feature_space": "metadata",
            "classifier": "SVC",
            "artifact_file": "e4_metadata_svm.joblib",
        },
        "hybrid_nb": {
            "id": "E5",
            "name": "Hybrid BernoulliNB (Core 19)",
            "feature_space": "hybrid",
            "classifier": "BernoulliNB",
            "artifact_file": "e5_hybrid_nb.joblib",
        },
        "hybrid_svm": {
            "id": "E6",
            "name": "Hybrid SVM (RBF, Unscaled)",
            "feature_space": "hybrid",
            "classifier": "SVC",
            "artifact_file": "e6_hybrid_svm.joblib",
        },
        # Remediation: Continuous Feature Scaling & SVM Kernel Audits
        "hybrid_svm_scaled_minmax": {
            "id": "E6_scaled_minmax",
            "name": "Hybrid SVM (RBF, MinMaxScaler)",
            "feature_space": "hybrid",
            "classifier": "SVC",
            "artifact_file": "e6_scaled_minmax_hybrid_svm_scaled_minmax.joblib",
        },
        "hybrid_svm_scaled_std": {
            "id": "E6_scaled_std",
            "name": "Hybrid SVM (RBF, StandardScaler)",
            "feature_space": "hybrid",
            "classifier": "SVC",
            "artifact_file": "e6_scaled_std_hybrid_svm_scaled_std.joblib",
        },
        "hybrid_svm_linear": {
            "id": "E6_linear",
            "name": "Hybrid Linear SVM (Unscaled)",
            "feature_space": "hybrid",
            "classifier": "SVC_Linear",
            "artifact_file": "e6_linear_hybrid_svm_linear.joblib",
        },
        # Remediation: BernoulliNB Continuous Feature Handling Audits
        "hybrid_mnb": {
            "id": "E5_mnb",
            "name": "Hybrid MultinomialNB (Magnitude-Preserving)",
            "feature_space": "hybrid",
            "classifier": "MultinomialNB",
            "artifact_file": "e5_mnb_hybrid_mnb.joblib",
        },
        "hybrid_nb_median_thresh": {
            "id": "E5_median_thresh",
            "name": "Hybrid BernoulliNB (Median-Thresholded)",
            "feature_space": "hybrid",
            "classifier": "BernoulliNB",
            "artifact_file": "e5_median_thresh_hybrid_nb_median_thresh.joblib",
        },
        # Remediation: Extended 22-Feature Representations (with VADER Sentiment)
        "metadata_nb_22": {
            "id": "E3_ext22",
            "name": "Metadata BernoulliNB (Extended 22)",
            "feature_space": "metadata",
            "classifier": "BernoulliNB",
            "artifact_file": "e3_ext22_metadata_nb_22.joblib",
        },
        "metadata_svm_22": {
            "id": "E4_ext22",
            "name": "Metadata SVM (Extended 22, Unscaled)",
            "feature_space": "metadata",
            "classifier": "SVC",
            "artifact_file": "e4_ext22_metadata_svm_22.joblib",
        },
        "hybrid_nb_22": {
            "id": "E5_ext22",
            "name": "Hybrid BernoulliNB (Extended 22)",
            "feature_space": "hybrid",
            "classifier": "BernoulliNB",
            "artifact_file": "e5_ext22_hybrid_nb_22.joblib",
        },
        "hybrid_svm_22_scaled": {
            "id": "E6_ext22_scaled",
            "name": "Hybrid SVM (Extended 22, MinMaxScaler)",
            "feature_space": "hybrid",
            "classifier": "SVC",
            "artifact_file": "e6_ext22_scaled_hybrid_svm_22_scaled.joblib",
        },
        # Remediation: Preprocessing Comparison (Stemming & Stopwords Filtering)
        "text_nb_stemmed": {
            "id": "E1_stemmed",
            "name": "Textual BernoulliNB (Stemmed + Stopwords)",
            "feature_space": "textual",
            "classifier": "BernoulliNB",
            "artifact_file": "e1_stemmed_text_nb_stemmed.joblib",
        },
        "text_svm_stemmed": {
            "id": "E2_stemmed",
            "name": "Textual SVM (Stemmed + Stopwords)",
            "feature_space": "textual",
            "classifier": "SVC",
            "artifact_file": "e2_stemmed_text_svm_stemmed.joblib",
        },
        "hybrid_svm_stemmed_scaled": {
            "id": "E6_stemmed_scaled",
            "name": "Hybrid SVM (Stemmed + MinMaxScaler)",
            "feature_space": "hybrid",
            "classifier": "SVC",
            "artifact_file": "e6_stemmed_scaled_hybrid_svm_stemmed_scaled.joblib",
        },
    }

    # Classifier hyperparameters
    nb_alpha: float = 1.0
    nb_binarize: float = 0.0
    svm_c: float = 1.0
    svm_kernel: str = "rbf"


settings = Settings()
