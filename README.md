# Textual, Non-Textual, and Hybrid Feature Engineering for SMS Spam Classification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-63%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible, research-grade machine learning pipeline implementing and evaluating classical probabilistic (**Bernoulli Naïve Bayes**, **Multinomial Naïve Bayes**) and margin-based (**Support Vector Machine**) classifiers across **Textual (Bag-of-Words)**, **Non-Textual (Handcrafted Metadata)**, and **Hybrid (Concatenated)** feature spaces.

This implementation is inspired by the research paper:
> **"Textual, Non-Textual, and Hybrid Feature Engineering for SMS Spam Classification"**  
> *Aditi Ranjit Kumar Verma (Senior Member, IEEE) and Shriya Sadana*  
> Published in IEEE Access, Vol. 13, 2025. Digital Object Identifier: `10.1109/ACCESS.2025.3620751`

---

## 1. Project Overview & Architecture

Contemporary spam detection often deploys compute-intensive Deep Learning and Large Language Models (BERT, LSTMs, Transformers) requiring GPU acceleration and substantial inference latencies. This project benchmarks lightweight, CPU-efficient, interpretable learning models designed for resource-constrained environments (mobile handsets, real-time carrier gateways) operating at sub-millisecond latencies.

### Project Structure

```text
client-project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py         # FastAPI dependency injection (DB sessions, services)
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── router.py           # Combined v1 API router
│   │   │       ├── health.py           # Liveness and readiness endpoints
│   │   │       ├── classify.py         # Single and batch classification endpoints
│   │   │       ├── models.py           # Model catalog and metadata endpoints
│   │   │       ├── experiments.py      # Benchmark comparison reports endpoints
│   │   │       └── history.py          # Paginated prediction history CRUD
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Pydantic v2 BaseSettings configuration
│   │   │   └── logging.py              # Structured, privacy-safe application logging
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # SQLAlchemy 2.x DeclarativeBase
│   │   │   ├── session.py              # Engine, session factory & SQLite PRAGMA init
│   │   │   └── models.py               # PredictionRecord ORM model with indices
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   └── prediction_repository.py# Data access layer for prediction persistence
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── common.py               # Shared API error & pagination schemas
│   │   │   ├── model.py                # Model metadata and summary schemas
│   │   │   ├── experiment.py           # Experiment comparison & metric schemas
│   │   │   ├── history.py              # Prediction history response schemas
│   │   │   └── prediction.py           # Single/Batch request and response schemas
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── classification_service.py # Inference orchestration & latency timing
│   │   │   ├── model_registry_service.py # Manifest validation & artifact readiness
│   │   │   ├── experiment_service.py   # Benchmark metrics & cross-validation reader
│   │   │   └── history_service.py      # History filtering, pagination & privacy masking
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py          # Encoding-safe loader, validator & summary generator
│   │   │   ├── preprocessing.py        # Deterministic dual-stream text cleaner & stemmer
│   │   │   ├── features.py             # Textual BoW, Handcrafted Metadata, Scalers & Hybrid hstack
│   │   │   ├── models.py               # Unified, self-contained pipeline serialization
│   │   │   ├── train.py                # End-to-end reproducible training & evaluation runner
│   │   │   ├── evaluate.py             # Held-out metrics, reports & confusion matrix plotting
│   │   │   └── inference.py            # Cached inference module for FastAPI integration
│   │   └── main.py                     # FastAPI application factory, lifespan & CORS
│   └── data/
│       └── app.db                      # Local SQLite application database
├── data/
│   ├── raw/
│   │   └── SMSSpamCollection           # Authentic UCI SMS Spam benchmark dataset
│   └── processed/
│       └── .gitkeep
├── artifacts/                          # Versioned, standalone model binaries & schemas
│   ├── e1_text_nb.joblib
│   ├── e2_text_svm.joblib
│   ├── e3_metadata_nb.joblib
│   ├── e4_metadata_svm.joblib
│   ├── e5_hybrid_nb.joblib
│   ├── e6_hybrid_svm.joblib
│   ├── e6_scaled_minmax_hybrid_svm_scaled_minmax.joblib
│   ├── e6_scaled_std_hybrid_svm_scaled_std.joblib
│   ├── e6_linear_hybrid_svm_linear.joblib
│   ├── e5_mnb_hybrid_mnb.joblib
│   ├── e5_median_thresh_hybrid_nb_median_thresh.joblib
│   ├── e3_ext22_metadata_nb_22.joblib
│   ├── e4_ext22_metadata_svm_22.joblib
│   ├── e5_ext22_hybrid_nb_22.joblib
│   ├── e6_ext22_scaled_hybrid_svm_22_scaled.joblib
│   ├── e1_stemmed_text_nb_stemmed.joblib
│   ├── e2_stemmed_text_svm_stemmed.joblib
│   ├── e6_stemmed_scaled_hybrid_svm_stemmed_scaled.joblib
│   ├── feature_schema.json             # Core 19 and Extended 22 schemas
│   └── manifest.json                   # Cryptographic hashes, split indices & parameters
├── reports/                            # Machine-readable audit reports & figures
│   ├── dataset_summary.json
│   ├── experiment_metrics.json
│   ├── experiment_comparison.csv       # Tabular comparison for all 18 configurations
│   ├── model_selection_cv.json         # 5-Fold Stratified Cross-Validation report
│   ├── prediction_comparison_audit.json# Prediction array & decision score comparisons
│   ├── classification_reports.json
│   ├── predictions.csv                 # Test predictions and scores for all models
│   ├── training_summary.json
│   └── confusion_matrices/             # Static Matplotlib heatmaps for each model
├── tests/                              # Pytest automated test suite (63 automated tests)
│   ├── conftest.py                     # Test fixtures, client & temporary DB session
│   ├── fixtures/
│   ├── test_health.py                  # API health and readiness tests
│   ├── test_classification_api.py      # Single classification API & validation tests
│   ├── test_batch_classification.py    # Batch classification API & limit tests
│   ├── test_models_api.py              # Model catalog & metadata tests
│   ├── test_experiments_api.py         # Experiment benchmark report tests
│   ├── test_history_api.py             # Prediction history CRUD & pagination tests
│   ├── test_database.py                # SQLite repository & privacy tests
│   ├── test_data_loader.py             # Phase 1 data loader tests
│   ├── test_preprocessing.py           # Phase 1 text cleaning tests
│   ├── test_features.py                # Phase 1 feature extraction tests
│   ├── test_training.py                # Phase 1 model training tests
│   └── test_inference.py               # Phase 1 inference tests
├── scripts/
│   └── train_models.py                 # Convenience CLI runner
├── .env.example                        # Documented environment configuration template
├── requirements.txt                    # Project dependencies (FastAPI, SQLAlchemy, scikit-learn)
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 2. Environment Setup & Dependency Installation

This pipeline requires **Python 3.11** or **Python 3.12**.

### Installation Steps

1. Navigate to the project repository:
   ```powershell
   cd client-project
   ```

2. Create and activate a Python virtual environment:
   ```powershell
   # Windows (PowerShell):
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Download the optional NLTK VADER lexicon and stopwords:
   ```powershell
   py -3.12 -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
   ```

---

## 3. Dataset Requirements & Provenance

The benchmark dataset is the canonical **UCI SMS Spam Collection** corpus:
- **Source**: [UCI Machine Learning Repository: SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Direct Archive URL**: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- **Expected Location**: `data/raw/SMSSpamCollection`

### Actual Measured Dataset Statistics
| Attribute | Actual Measured Value | Reference Paper Stated Value | Notes / Discrepancy Analysis |
|---|---|---|---|
| Total Raw Messages | **5,574** | 5,574 | Exact match with original corpus. |
| Raw Class Distribution | 4,827 ham / 747 spam | 4,827 ham / 747 spam | Spam prevalence: 13.40%. |
| Exact Duplicate Removal | **414 records** | 415 records | Deduplication by exact message text. |
| Retained Unique Records | **5,160 records** | 5,159 records | 1 record difference due to verbatim string parsing. |
| Retained Class Balance | **4,518 ham / 642 spam** | 4,518 ham / 641 spam | Exact ham count matches paper; spam count is 642 vs 641. |
| Retained Spam Prevalence | **12.44%** | 12.40% | Close match. |
| Mean Char Length (Ham) | **70.86 chars** | ~71 chars | Verified. |
| Mean Char Length (Spam) | **137.41 chars** | ~139 chars | Verified. |
| Mean Token Count (Ham) | **14.23 tokens** | ~14 tokens | Verified. |
| Mean Token Count (Spam) | **23.70 tokens** | ~24 tokens | Verified. |

---

## 4. Text Preprocessing & Feature Engineering

### Dual-Stream Architecture
Raw SMS messages undergo dual-stream separation:
- **Stream 1 (Raw Text)**: Retains raw casing, punctuation, hyperlinks, currency symbols, and phone numbers untouched for handcrafted stylistic and structural feature extraction.
- **Stream 2 (Cleaned Text)**: Trimmed whitespace, excised ASCII control characters (`[\x00-\x1f\x7f-\x9f]`), isolated embedded URLs (`https?://\S+|www\.\S+`), lowercased, and normalized spaces for lexical Bag-of-Words modeling.

### A. Textual Representation (Bag-of-Words)
- Scikit-learn `CountVectorizer` fitted **strictly on the 80% training partition** (4,128 messages).
- Standard BoW vocabulary size: **7,712 terms**.
- Stemmed BoW vocabulary size (Porter stemming + stopword removal): **7,279 terms**.
- Stored as `scipy.sparse.csr_matrix` (zero-copy transform, memory-efficient).

### B. Non-Textual Handcrafted Metadata Representation
- **Core 19 Features (`paper_core`)**:
  - **8 Continuous Features**: `num_capitalized_words`, `length`, `unique_words_count`, `avg_word_length`, `word_count`, `exclam_cnt`, `punct_cnt`, `quest_cnt`.
  - **11 Binary Indicators**: `has_mobile_number`, `has_money_symbol`, `spam_call`, `spam_mobil`, `spam_free`, `spam_claim`, `has_link`, `spam_text`, `spam_repli`, `spam_stop`, `has_caps_word`.
- **Extended 22 Features (`paper_extended`)**:
  - Adds 3 VADER sentiment scores: `vader_compound`, `vader_positive`, `vader_negative`.

---

## 5. Final Phase 1 Verification & Technical Audit Findings

### Audit 1: Experimental Evaluation Protocol & Model Selection Bias Disclosure
- **Selection Bias Disclosure**: In exploratory machine learning, repeatedly evaluating and selecting candidate model configurations on a single fixed test split (such as the standard 80:20 split of 1,032 samples) introduces **selection bias**. The test split inadvertently serves as an implicit validation set, causing optimistic performance estimates. Therefore, all metrics evaluated on the 80:20 test split in this repository are explicitly designated as an **exploratory benchmark**, and we do not claim this test split is untouched.
- **Reproducible Model Selection Protocol**: To establish a rigorous, unbiased model-selection workflow, candidate evaluation was implemented strictly using **5-Fold Stratified Cross-Validation on the training partition (`df_train`, 4,128 messages)**. The test set was not consulted during model ranking or candidate selection.
- **Cross-Validation Model Selection Results**:
  - Top Candidate: **`E5_mnb` (`hybrid_mnb`, MultinomialNB)**: Mean CV Accuracy = **0.9881 $\pm$ 0.0045**, Mean CV F1 = **0.9501 $\pm$ 0.0193**, Mean CV Recall = **0.9125**.
  - Second Candidate: **`E6_linear` (`hybrid_svm_linear`)**: Mean CV Accuracy = **0.9862 $\pm$ 0.0063**, Mean CV F1 = **0.9424 $\pm$ 0.0273**.
  - Third Candidate: **`E6_scaled_minmax` (`hybrid_svm_scaled_minmax`)**: Mean CV Accuracy = **0.9859 $\pm$ 0.0044**, Mean CV F1 = **0.9402 $\pm$ 0.0195**.
  - Fourth Candidate: **`E6_ext22_scaled` (`hybrid_svm_22_scaled`)**: Mean CV Accuracy = **0.9859 $\pm$ 0.0044**, Mean CV F1 = **0.9402 $\pm$ 0.0195**.
  - Lowest Ranked: Unscaled Hybrid SVM `E6` (CV Accuracy = **0.9339 $\pm$ 0.0066**, CV F1 = **0.6947**) and unscaled metadata models (`E4`, `E4_ext22` with CV Acc $\approx$ 0.926).
- **Protocol for Unbiased Final Performance Estimation**: For a genuinely unbiased final evaluation, the frozen top candidate must be evaluated on an independent, previously unconsulted external corpus (e.g., Enron-Spam, Ling-Spam, or an untouched SMS collection split).

### Audit 2: Verification of Extended Metadata Scaler & VADER Handling
- **Implementation Inspection**: Inspected `MetadataScaler`, `SpamClassifierPipeline`, and `feature_schema.json`.
- **Name-Based Schema Column Mapping**: `MetadataScaler` utilizes explicit schema column mapping (`{name: idx}`) derived from `feature_names` and `continuous_features`. This prevents hardcoded index slices (`[:8]`) from corrupting extended representations.
  - **Core 19 Schema**: Continuous indices are strictly `[0, 1, 2, 3, 4, 5, 6, 7]`; binary indicators are `[8..18]`.
  - **Extended 22 Schema**: Continuous indices are strictly `[0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21]` (8 core continuous + 3 sentiment); binary indicators remain unperturbed at `[8..18]`.
- **Handling of Potentially Negative `vader_compound`**: `vader_compound` scores span $[-1.0, 1.0]$. `MinMaxScaler(feature_range=(0.0, 1.0))` maps this continuous interval into $[0.0, 1.0]$ based strictly on training-set extrema, while binary indicators remain strictly $\{0.0, 1.0\}$.
- **Training and Inference Parity**: Verified via automated unit tests that `SpamClassifierPipeline.transform_features` applies the identical transformation pipeline at inference time as was learned during training.

### Audit 3: Empirical Array-Level Comparison of Scaled SVMs (Core 19 vs Extended 22)
The aggregate confusion matrix counts for `E6_scaled_minmax` (Core 19) and `E6_ext22_scaled` (Extended 22) are identical ($\text{TP}=114, \text{TN}=903, \text{FP}=1, \text{FN}=14$). A complete record-by-record comparison of their predictions and continuous decision scores was executed on all 1,032 test messages (`reports/prediction_comparison_audit.json`):
- **Identical Predictions**: **1,032 / 1,032 (100.0%)**.
- **Differing Predictions**: **0 / 1,032 (0.0%)**.
- **Differing Continuous Decision Scores**: **1,017 / 1,032 (98.55%)**!
- **Maximum Absolute Score Difference**: **0.0927**; **Mean Absolute Difference**: **0.009486**.
- **Do Extended Features Affect Predictions?**: **Yes**. The 3 added VADER sentiment features actively perturb the SVM hyperplane and continuous decision margins for 98.55% of samples. However, because the decision boundary threshold is at $0.0$ and the separating margin for clear ham/spam messages is sufficiently wide, none of the continuous perturbations cross $0.0$ on this split.
- **Feature Space Dimensions**: Verified in saved model artifacts that Core 19 uses dimension **7,731** (7,712 text + 19 metadata) with 8 continuous features, whereas Extended 22 uses dimension **7,734** (7,712 text + 22 metadata) with 11 continuous features.

### Audit 4: Scientific Research Interpretation Corrections
- **Distinction Between Baseline Reproduction and Remediations**:
  - The paper-style RBF-SVM baseline (`E6`) without feature scaling achieves **94.19% accuracy** on the held-out test split (and 93.39% in 5-fold CV). This is the honest, unscaled reproduction measurement.
  - The higher-performing models (`E6_scaled_minmax` at 98.55%, `E6_linear` at 98.55%, `E5_mnb` at 98.84%, and `E6_stemmed_scaled` at 98.74%) represent post-hoc **engineering remediations** to solve the RBF kernel distance distortion.
  - **No False Reproduction Claim**: We do *not* claim exact reproduction of the paper's claimed 98.3% unscaled SVM simply because our scaled and linear models reach $\ge 98.55\%$. The paper may have utilized undocumented feature normalization, linear formulations, or different optimization parameters.
- **Bernoulli Naïve Bayes Magnitude Breakdown**:
  - `BernoulliNB(binarize=0.0)` applies indicator mapping $x \mapsto \mathbb{I}[x > 0.0]$.
  - **Features Collapsing to Constant 1s**: `length`, `word_count`, `unique_words_count`, and `avg_word_length`. Because all SMS messages contain $>0$ characters and words, these features have zero variance and provide zero mutual information to the Bernoulli model.
  - **Features Retaining Binary Presence Information**: `exclam_cnt`, `quest_cnt`, `punct_cnt`, and `num_capitalized_words`. Messages with zero punctuation/caps receive $0.0$, while messages with $\ge 1$ receive $1.0$. Frequency magnitude is lost, but presence/absence signal is preserved.
  - This explains why `E5_mnb` (MultinomialNB), which naturally models raw continuous frequencies, outperforms BernoulliNB by $+0.78\%$ accuracy (98.84% vs 98.06%).

### Audit 5: Symmetrical Error Exchange Between Text NB and Text SVM
- Both `E1` (Text NB) and `E2` (Text SVM) reported $\text{TP}=103, \text{TN}=903, \text{FP}=1, \text{FN}=25$.
- Record-by-record comparison revealed **14 prediction disagreements** (1,018 identical, 14 differing):
  - Naïve Bayes correctly detected 6 spam messages that SVM missed.
  - SVM correctly detected 6 spam messages that Naïve Bayes missed.
  - Each model committed 1 false positive on completely different ham messages (rows 3517 vs 4410).
- This confirms the identical aggregate metrics are due to symmetrical error exchange between generative and discriminative hypothesis classes, rather than redundant predictions.

---

## 6. Comprehensive Benchmark & Cross-Validation Results

### A. Model Selection: 5-Fold Stratified Cross-Validation on Training Set (`df_train`, 4,128 samples)
*Evaluated strictly on training data without consulting the test set:*

| Rank | Experiment ID | Model Name | Feature Space | Classifier | Mean CV Accuracy | Std CV Accuracy | Mean CV Precision | Mean CV Recall | Mean CV F1 | Std CV F1 |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **E5_mnb** | hybrid_mnb | hybrid | MultinomialNB | **0.9881** | 0.0045 | 0.9916 | **0.9125** | **0.9501** | 0.0193 |
| 2 | **E6_linear** | hybrid_svm_linear | hybrid | SVC_Linear | **0.9862** | 0.0063 | 0.9729 | 0.9144 | **0.9424** | 0.0273 |
| 3 | **E6_scaled_minmax** | hybrid_svm_scaled_minmax | hybrid | SVC (RBF) | **0.9859** | 0.0044 | 0.9937 | 0.8931 | **0.9402** | 0.0195 |
| 4 | **E6_ext22_scaled** | hybrid_svm_22_scaled | hybrid | SVC (RBF) | **0.9859** | 0.0044 | 0.9937 | 0.8931 | **0.9402** | 0.0195 |
| 5 | **E6_scaled_std** | hybrid_svm_scaled_std | hybrid | SVC (RBF) | **0.9845** | 0.0045 | 0.9852 | 0.8891 | **0.9342** | 0.0197 |
| 6 | **E6_stemmed_scaled** | hybrid_svm_stemmed_scaled | hybrid | SVC (RBF) | **0.9830** | 0.0060 | 0.9936 | 0.8697 | **0.9266** | 0.0284 |
| 7 | **E5_median_thresh** | hybrid_nb_median_thresh | hybrid | BernoulliNB | **0.9821** | 0.0049 | 0.9977 | 0.8581 | **0.9222** | 0.0225 |
| 8 | **E2** | text_svm | textual | SVC (RBF) | **0.9811** | 0.0059 | 0.9910 | 0.8561 | **0.9180** | 0.0267 |
| 9 | **E5** | hybrid_nb | hybrid | BernoulliNB | **0.9799** | 0.0066 | 0.9976 | 0.8406 | **0.9116** | 0.0308 |
| 10 | **E5_ext22** | hybrid_nb_22 | hybrid | BernoulliNB | **0.9799** | 0.0063 | 0.9977 | 0.8405 | **0.9116** | 0.0291 |
| 11 | **E3** | metadata_nb | metadata | BernoulliNB | **0.9753** | 0.0049 | 0.9330 | 0.8639 | **0.8966** | 0.0221 |
| 12 | **E2_stemmed** | text_svm_stemmed | textual | SVC (RBF) | **0.9748** | 0.0058 | 0.9975 | 0.7997 | **0.8871** | 0.0278 |
| 13 | **E3_ext22** | metadata_nb_22 | metadata | BernoulliNB | **0.9721** | 0.0050 | 0.9119 | 0.8600 | **0.8847** | 0.0221 |
| 14 | **E1** | text_nb | textual | BernoulliNB | **0.9729** | 0.0061 | 0.9973 | 0.7842 | **0.8774** | 0.0300 |
| 15 | **E1_stemmed** | text_nb_stemmed | textual | BernoulliNB | **0.9680** | 0.0060 | 1.0000 | 0.7433 | **0.8519** | 0.0317 |
| 16 | **E6** | hybrid_svm | hybrid | SVC (RBF) | **0.9339** | 0.0066 | 0.8174 | 0.6052 | **0.6947** | 0.0322 |
| 17 | **E4** | metadata_svm | metadata | SVC (RBF) | **0.9264** | 0.0065 | 0.7910 | 0.5565 | **0.6529** | 0.0300 |
| 18 | **E4_ext22** | metadata_svm_22 | metadata | SVC (RBF) | **0.9261** | 0.0064 | 0.7922 | 0.5527 | **0.6504** | 0.0310 |

### B. Exploratory Benchmark: Held-Out Test Split (1,032 samples: 904 ham, 128 spam)
*Reported as exploratory evaluation baseline; not claimed as untouched:*

| Experiment | Model Name | Feature Space | Feature Count | Scaling | Classifier | Test Accuracy | Spam Precision | Spam Recall | Spam F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E1** *(Baseline)* | text_nb | textual | 7,712 | none | BernoulliNB | **97.48%** | 0.9904 | 0.8047 | 0.8879 | 103 | 903 | 1 | 25 |
| **E2** *(Baseline)* | text_svm | textual | 7,712 | none | SVC (RBF) | **97.48%** | 0.9904 | 0.8047 | 0.8879 | 103 | 903 | 1 | 25 |
| **E3** *(Baseline)* | metadata_nb | metadata | 19 | none | BernoulliNB | **97.97%** | 0.9573 | 0.8750 | 0.9143 | 112 | 899 | 5 | 16 |
| **E4** *(Baseline)* | metadata_svm | metadata | 19 | none | SVC (RBF) | **93.70%** | 0.8621 | 0.5859 | 0.6977 | 75 | 892 | 12 | 53 |
| **E5** *(Baseline)* | hybrid_nb | hybrid | 7,731 | none | BernoulliNB | **98.06%** | 0.9821 | 0.8594 | 0.9167 | 110 | 902 | 2 | 18 |
| **E6** *(Baseline)* | hybrid_svm | hybrid | 7,731 | none | SVC (RBF) | **94.19%** | 0.8778 | 0.6172 | 0.7248 | 79 | 893 | 11 | 49 |
| **E6_scaled_minmax** | hybrid_svm_scaled_minmax | hybrid | 7,731 | minmax | SVC (RBF) | **98.55%** | **0.9913** | **0.8906** | **0.9383** | 114 | 903 | 1 | 14 |
| **E6_scaled_std** | hybrid_svm_scaled_std | hybrid | 7,731 | standard | SVC (RBF) | **97.87%** | 0.9649 | 0.8594 | 0.9091 | 110 | 900 | 4 | 18 |
| **E6_linear** | hybrid_svm_linear | hybrid | 7,731 | none | SVC_Linear | **98.55%** | 0.9748 | 0.9062 | 0.9393 | 116 | 901 | 3 | 12 |
| **E5_mnb** | hybrid_mnb | hybrid | 7,731 | none | MultinomialNB | **98.84%** | **0.9915** | **0.9141** | **0.9512** | 117 | 903 | 1 | 11 |
| **E5_median_thresh** | hybrid_nb_median_thresh | hybrid | 7,731 | median | BernoulliNB | **98.26%** | 0.9825 | 0.8750 | 0.9256 | 112 | 902 | 2 | 16 |
| **E3_ext22** | metadata_nb_22 | metadata | 22 | none | BernoulliNB | **97.58%** | 0.9256 | 0.8750 | 0.8996 | 112 | 895 | 9 | 16 |
| **E4_ext22** | metadata_svm_22 | metadata | 22 | none | SVC (RBF) | **93.60%** | 0.8605 | 0.5781 | 0.6916 | 74 | 892 | 12 | 54 |
| **E5_ext22** | hybrid_nb_22 | hybrid | 7,734 | none | BernoulliNB | **98.16%** | 0.9823 | 0.8672 | 0.9212 | 111 | 902 | 2 | 17 |
| **E6_ext22_scaled** | hybrid_svm_22_scaled | hybrid | 7,734 | minmax | SVC (RBF) | **98.55%** | 0.9913 | 0.8906 | 0.9383 | 114 | 903 | 1 | 14 |
| **E1_stemmed** | text_nb_stemmed | textual | 7,279 | none | BernoulliNB | **97.09%** | 0.9900 | 0.7734 | 0.8684 | 99 | 903 | 1 | 29 |
| **E2_stemmed** | text_svm_stemmed | textual | 7,279 | none | SVC (RBF) | **97.67%** | 0.9906 | 0.8203 | 0.8974 | 105 | 903 | 1 | 23 |
| **E6_stemmed_scaled** | hybrid_svm_stemmed_scaled | hybrid | 7,298 | minmax | SVC (RBF) | **98.74%** | **1.0000** | **0.8984** | **0.9465** | 115 | 904 | 0 | 13 |

---

## 7. Production-Grade FastAPI Backend & SQLite Persistence (Phase 2)

Phase 2 wraps the verified Phase 1 machine-learning engine into an asynchronous, production-ready **FastAPI** web service backed by a persistent **SQLite** database using **SQLAlchemy 2.x**.

### A. Architectural Principles & Separation of Concerns

The backend enforces strict modularity across four isolated layers:
1. **API Router Layer (`app/api/v1/`)**: Fast, lightweight HTTP request handlers validating input and invoking business services. Never interacts with raw disk artifacts or database cursors directly.
2. **Service Layer (`app/services/`)**: Encapsulates orchestration logic:
   - `ClassificationService`: High-resolution latency profiling (`time.perf_counter()`), model selection, score integrity mapping, and conditional history persistence.
   - `ModelRegistryService`: Validates model identifiers against `artifacts/manifest.json`, reports artifact file readiness, and manages in-memory pipeline caching.
   - `ExperimentService`: Reads and parses machine-readable reports (`reports/experiment_metrics.json`, `reports/model_selection_cv.json`, `reports/experiment_comparison.csv`).
   - `HistoryService`: Enforces pagination limits, multi-field filtering, and privacy masking.
3. **Repository Layer (`app/repositories/`)**: Encapsulates all SQLAlchemy ORM database queries, transactions, pagination, and record deletion.
4. **Machine Learning Engine (`app/ml/`)**: Completely preserved Phase 1 source of truth for text cleaning, metadata feature extraction, vectorization, scaling, and estimator inference.

### B. SQLite Storage & Schema Design

- **Database Engine**: Local SQLite file (`backend/data/app.db`), configured via `SQLAlchemy 2.x`.
- **Concurrency & Thread Safety**: Initialized with `check_same_thread=False` and connection pool settings suited for ASGI request loops.
- **Foreign Key Enforcement**: Explicitly enforces `PRAGMA foreign_keys=ON` on every SQLite connection checkout.
- **Automatic Initialization**: Database directory and tables are created automatically during FastAPI application startup lifespan.

#### `PredictionRecord` ORM Table Schema (`app/db/models.py`)

| Column Name | SQL Type | Nullable | Description & Constraints |
|---|---|---|---|
| `id` | `INTEGER` | No | Auto-incrementing primary key. |
| `created_at` | `DATETIME` | No | UTC timestamp of the prediction event (`index=True`). |
| `model_id` | `VARCHAR(64)` | No | Validated model identifier from the registry (`index=True`). |
| `prediction` | `VARCHAR(16)` | No | `"spam"` or `"ham"` (`index=True`). |
| `is_spam` | `BOOLEAN` | No | Boolean spam flag. |
| `message_length` | `INTEGER` | No | Character length of the input message. |
| `inference_ms` | `FLOAT` | No | Measured inference latency in milliseconds (feature transform + estimator). |
| `score_type` | `VARCHAR(32)` | Yes | Score interpretation: `"probability"` or `"decision_score"`. |
| `spam_score` | `FLOAT` | Yes | Numerical output value (probability or hyperplane distance). |
| `message_text` | `TEXT` | Yes | Raw message text. **Only stored when `save_raw_messages=True`**. |

### C. Privacy & Data Governance

- **Default Privacy Mode**: `save_raw_messages=False` by default in `.env`.
- When disabled, `message_text` is stored as `None` in the database, even if `save_history=true` was specified by the client. Only non-sensitive audit metadata (`created_at`, `model_id`, `prediction`, `message_length`, `inference_ms`) is retained.
- **Log Masking**: Application logs (`app/core/logging.py`) record client IP, method, route, latency, and status code, but **never log raw SMS message text**.

### D. Model Score Integrity

- **Probabilistic Estimators (`score_type: "probability"`)**: Multinomial Naïve Bayes (`hybrid_mnb`) and Bernoulli Naïve Bayes models expose calibrated class probabilities $P(\text{spam} \mid x) \in [0.0, 1.0]$ via `predict_proba`.
- **Margin-Based Estimators (`score_type: "decision_score"`)**: Support Vector Machines (RBF and Linear SVC) expose continuous signed hyperplane geometric distances $(-\infty, +\infty)$ via `decision_function`.
- **Scientific Integrity Guarantee**: SVM decision scores are **never** normalized to $[0.0, 1.0]$ or misrepresented as probabilities. Clients receive explicit `score_type` annotations to prevent misleading interpretations.

### E. API Endpoints & Request/Response Contracts

All API endpoints are versioned under `/api/v1`:

#### 1. System Health & Readiness
- `GET /api/v1/health` — Service liveness, application version, and UTC timestamp.
- `GET /api/v1/health/ready` — Readiness check verifying SQLite connectivity, table schemas, and default model artifact (`hybrid_mnb`) availability.

#### 2. Message Classification
- `POST /api/v1/classify` — Classifies a single SMS message.
  ```json
  // Request:
  {
    "message": "URGENT! You have won a 1 week FREE holiday to Spain! Call 09061701461 to claim.",
    "model_id": "hybrid_mnb",
    "save_history": true
  }
  
  // Response (HTTP 200):
  {
    "prediction": "spam",
    "is_spam": true,
    "model_id": "hybrid_mnb",
    "inference_ms": 1.45,
    "score": 0.9998,
    "score_type": "probability",
    "history_id": 42
  }
  ```

- `POST /api/v1/classify/batch` — Classifies a batch of 1 to 100 messages with input order preservation.
  ```json
  // Request:
  {
    "messages": [
      "Are we having lunch at the cafeteria today?",
      "Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005."
    ],
    "model_id": "hybrid_mnb",
    "save_history": false
  }
  
  // Response (HTTP 200):
  {
    "model_id": "hybrid_mnb",
    "batch_size": 2,
    "total_batch_ms": 3.12,
    "results": [
      {
        "index": 0,
        "prediction": "ham",
        "is_spam": false,
        "inference_ms": 1.22,
        "score": 0.0001,
        "score_type": "probability",
        "history_id": null
      },
      {
        "index": 1,
        "prediction": "spam",
        "is_spam": true,
        "inference_ms": 1.15,
        "score": 0.9999,
        "score_type": "probability",
        "history_id": null
      }
    ]
  }
  ```

#### 3. Model Catalog & Metadata
- `GET /api/v1/models` — Lists all 18 registered models, including classifier type, feature representation, feature counts, readiness status, default flag, and score type.
- `GET /api/v1/models/{model_id}` — Detailed hyperparameters, preprocessing flags, and evaluation basis for a specific model identifier.

#### 4. Experiment Benchmark Reports
- `GET /api/v1/experiments` — Returns stored evaluation metrics across all 18 baseline and engineering comparison configurations.
- `GET /api/v1/experiments/{experiment_id}` — Returns detailed breakdown, confusion matrix, and 5-fold cross-validation metrics for a specific experiment ID.

#### 5. Prediction History CRUD
- `GET /api/v1/history?page=1&page_size=20&model_id=hybrid_mnb&prediction=spam` — Paginated history with multi-field database filtering.
- `GET /api/v1/history/{record_id}` — Retrieves one specific prediction record.
- `DELETE /api/v1/history/{record_id}` — Deletes a specific prediction record from SQLite.

---

## 8. Local Setup, Execution & Testing Commands

### A. Environment Configuration
Copy the template configuration file and review defaults:
```powershell
cp .env.example .env
```
Default `.env` settings:
```ini
APP_NAME="SMS Spam Classification Engine API"
APP_VERSION="1.0.0"
API_V1_PREFIX="/api/v1"
DATABASE_FILE="backend/data/app.db"
DEFAULT_MODEL="hybrid_mnb"
CORS_ORIGINS=["http://localhost:3000"]
MAX_MESSAGE_LENGTH=10000
MAX_BATCH_SIZE=100
SAVE_RAW_MESSAGES=false
LOG_LEVEL="INFO"
```

### B. Run FastAPI Development Server
Start Uvicorn with ASGI reload:
```powershell
# Set PYTHONPATH to include backend:
$env:PYTHONPATH="backend"
py -3.12 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema**: `http://127.0.0.1:8000/openapi.json`

### C. Run Full Automated Test Suite
Execute the complete test suite covering Phase 1 ML unit tests, FastAPI integration tests, SQLite repository operations, and validation checks:
```powershell
$env:PYTHONPATH="backend"
py -3.12 -m pytest -v tests
```
Result: **63 passed, 0 failed in 5.26s** (100% pass rate).

### D. Reproduce Model Training Pipeline (Optional)
To retrain and regenerate all 18 model artifacts and cross-validation reports:
```powershell
py -3.12 scripts/train_models.py
```

---

## 9. Phase 3: SPAM INTELLIGENCE Web Application

A cybersecurity-grade, responsive Next.js frontend connecting directly to the FastAPI service.

### A. Technology Stack
- **Framework**: Next.js 16.4 (App Router, Turbopack)
- **Language**: TypeScript 5.9 (Strict type safety, zero `any`)
- **Styling**: Tailwind CSS v4, custom cybersecurity and research design tokens
- **Data Visualization**: Recharts (Responsive bar charts, accessible tooltips)
- **Icons**: Lucide React
- **Testing**: Vitest, React Testing Library, jsdom (19 passed unit & component tests)

### B. Frontend Setup & Run Commands

```powershell
# 1. Navigate to frontend directory
cd frontend

# 2. Configure environment
cp .env.local.example .env.local
# Set: NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1

# 3. Install dependencies
npm install

# 4. Start Next.js development server
npm run dev

# 5. Run tests & linters
npm run test     # Vitest (19 passed tests)
npm run lint     # ESLint 9 (0 errors, 0 warnings)
npm run build    # Production build (8 static routes prerendered)
```

### C. Available Application Routes

1. **Overview (`/`)**: Operational health summary, 4-card metric grid, comparative benchmark chart (Recharts), and recent prediction audit log.
2. **Analyze Message (`/analyze`)**: Single-message classifier with character counter, harmless test prompts, model picker, privacy recording toggle, and score interpretation. Includes **Batch Analysis Mode** for up to 100 messages with ordered results and CSV export.
3. **Experiment Lab (`/experiments`)**: Deep research bench comparing all 18 evaluated models. Explicitly discloses **5-Fold Stratified Cross-Validation** (model selection) vs **80:20 Exploratory Holdout**. Includes interactive contingency confusion matrix inspector.
4. **Model Catalog (`/models`)**: Searchable directory of all 18 registered scikit-learn pipeline artifacts. Filterable by feature space (Textual, Metadata, Hybrid) and classifier family. Modal inspector displays exact vectorizer parameters, scalers, and readiness.
5. **Prediction History (`/history`)**: Paginated SQLite-backed audit log of classification events. Filters by prediction label and model. Respects user privacy: displays "Message text not stored" mask. Modal deletion confirmation connected to `DELETE /api/v1/history/{id}`.

---

## 10. Research Context & Methodological Disclosures

1. **Model Selection Protocol**: The default production model `hybrid_mnb` was selected through **5-fold Stratified Cross-Validation on `df_train`** (Accuracy = $0.9881 \pm 0.0045$, F1 = $0.9501 \pm 0.0193$), avoiding test-set selection bias.
2. **Exploratory Test Benchmark**: The 80:20 held-out test split results reported in `reports/experiment_metrics.json` are explicitly marked as **exploratory benchmarks** because the split was consulted during earlier exploratory modeling.
3. **Score Semantics**: Naïve Bayes models return estimated posterior probabilities ($P(\text{spam}|x)$). SVM models return continuous signed distance to the separating hyperplane (`decision_score`). SVM margins are never represented as probabilities or percentages.
4. **Database Concurrency**: The SQLite backend is optimized for single-host deployments and local application usage. SQLite automatically manages database write locks with transaction isolation.

---

## 11. License

This project is licensed under the MIT License.

