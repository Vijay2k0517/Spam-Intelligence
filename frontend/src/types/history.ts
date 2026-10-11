/**
 * Types for SQLite prediction history records and query filters.
 */

import { PageMeta } from './api';

export interface PredictionHistoryItem {
  id: number;
  created_at: string;
  model_id: string;
  prediction: 'spam' | 'ham';
  is_spam: boolean;
  message_length: number;
  inference_ms: number;
  score_type?: 'probability' | 'decision_score' | string | null;
  spam_score?: number | null;
  message_text?: string | null;
}

export interface PaginatedHistoryResponse {
  items: PredictionHistoryItem[];
  pagination: PageMeta;
}

export interface HistoryFilters {
  page?: number;
  page_size?: number;
  model_id?: string;
  prediction?: 'spam' | 'ham';
  start_date?: string;
  end_date?: string;
}
