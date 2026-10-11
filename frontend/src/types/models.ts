/**
 * Model registry metadata types matching FastAPI /api/v1/models endpoints.
 */

export interface ModelSummary {
  model_id: string;
  display_name: string;
  classifier_type: string;
  feature_space: 'textual' | 'metadata' | 'hybrid' | string;
  feature_count: number;
  is_default: boolean;
  is_available: boolean;
  supports_probability: boolean;
  supports_decision_score: boolean;
  score_type?: 'probability' | 'decision_score' | string | null;
  selection_basis: string;
}

export interface ModelDetail extends ModelSummary {
  experiment_id: string;
  scaling: string;
  feature_mode: 'paper_core' | 'paper_extended' | string;
  preprocessing_details?: string | null;
  model_parameters?: string | null;
  artifact_file?: string | null;
  test_metrics?: Record<string, unknown> | null;
}

export interface ModelListResponse {
  models: ModelSummary[];
  total_models: number;
  default_model: string;
}
