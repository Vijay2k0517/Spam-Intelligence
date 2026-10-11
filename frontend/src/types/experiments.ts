/**
 * Types for Phase 1 experiment benchmark and CV reports.
 */

export interface ConfusionMatrixData {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface ExperimentSummary {
  experiment_id: string;
  model_name: string;
  feature_space: 'textual' | 'metadata' | 'hybrid' | string;
  classifier: string;
  scaling: string;
  evaluation_protocol: '80:20 Exploratory Benchmark' | '5-Fold Stratified Cross-Validation' | string;
  test_accuracy?: number | null;
  spam_precision?: number | null;
  spam_recall?: number | null;
  spam_f1?: number | null;
}

export interface CrossValidationMetrics {
  mean_accuracy?: number;
  std_accuracy?: number;
  mean_precision?: number;
  std_precision?: number;
  mean_recall?: number;
  std_recall?: number;
  mean_f1?: number;
  std_f1?: number;
  folds?: number;
  samples?: number;
  [key: string]: unknown;
}

export interface ExperimentDetail extends ExperimentSummary {
  feature_count: number;
  feature_mode: string;
  preprocessing?: string | null;
  model_parameters?: string | null;
  macro_f1?: number | null;
  weighted_f1?: number | null;
  roc_auc?: number | null;
  pr_auc?: number | null;
  confusion_matrix?: ConfusionMatrixData | null;
  cross_validation_metrics?: CrossValidationMetrics | null;
  selection_bias_disclosure: string;
}

export interface ExperimentListResponse {
  experiments: ExperimentSummary[];
  total_experiments: number;
  selection_bias_disclosure: string;
}
