/**
 * Standard API error and pagination types matching FastAPI backend contracts.
 */

export interface ErrorDetail {
  code: string;
  message: string;
  field?: string | null;
}

export interface ErrorResponse {
  error: string;
  message: string;
  details?: ErrorDetail[] | null;
  timestamp: string;
}

export interface PageMeta {
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  app_version: string;
  api_version: string;
  timestamp: string;
}

export interface ReadinessResponse {
  status: 'ready' | 'not_ready';
  timestamp: string;
  checks: {
    database?: string;
    default_model?: string;
    model_status?: string;
    [key: string]: string | undefined;
  };
}
