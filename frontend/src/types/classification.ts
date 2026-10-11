/**
 * Types for single and batch message classification matching FastAPI contracts.
 */

export interface ClassifyRequest {
  message: string;
  model_id?: string | null;
  save_history?: boolean;
  include_metadata?: boolean;
}

export interface ClassifyResponse {
  prediction: 'spam' | 'ham';
  is_spam: boolean;
  model_id: string;
  inference_ms: number;
  score?: number | null;
  score_type?: 'probability' | 'decision_score' | string | null;
  probability?: number | null;
  history_id?: number | null;
  metadata_features?: Record<string, number> | null;
}

export interface BatchClassifyRequest {
  messages: string[];
  model_id?: string | null;
  save_history?: boolean;
  include_metadata?: boolean;
}

export interface BatchItemResult {
  index: number;
  prediction: 'spam' | 'ham';
  is_spam: boolean;
  inference_ms: number;
  score?: number | null;
  score_type?: 'probability' | 'decision_score' | string | null;
  probability?: number | null;
  history_id?: number | null;
  metadata_features?: Record<string, number> | null;
}

export interface BatchClassifyResponse {
  predictions: BatchItemResult[];
  total_messages: number;
  total_latency_ms: number;
  model_id: string;
}
