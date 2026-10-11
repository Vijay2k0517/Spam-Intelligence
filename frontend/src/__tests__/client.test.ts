import { describe, it, expect, vi, beforeEach } from 'vitest';
import { apiClient, ApiError, getApiBaseUrl } from '../lib/api/client';

describe('API Client Unit Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('resolves default API base URL correctly', () => {
    const url = getApiBaseUrl();
    expect(url).toBe('http://127.0.0.1:8000/api/v1');
  });

  it('performs successful GET request and parses JSON response', async () => {
    const mockData = { status: 'healthy', app_name: 'SMS Spam Engine' };
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => mockData,
    });

    const result = await apiClient<{ status: string }>('/health');
    expect(result).toEqual(mockData);
    expect(global.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/v1/health',
      expect.objectContaining({
        headers: expect.objectContaining({
          Accept: 'application/json',
        }),
      })
    );
  });

  it('parses structured ErrorResponse on HTTP failure', async () => {
    const mockErrorResponse = {
      error: 'VALIDATION_ERROR',
      message: 'Input message exceeds maximum length.',
      details: [{ code: 'max_length', message: 'Too long', field: 'message' }],
      timestamp: '2026-10-11T12:00:00Z',
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Content',
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => mockErrorResponse,
    });

    await expect(apiClient('/classify')).rejects.toThrow('Input message exceeds maximum length.');
  });

  it('throws ApiError with statusCode on 404', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({ detail: 'Model unknown_model was not found.' }),
    });

    try {
      await apiClient('/models/unknown_model');
      expect.fail('Should have thrown');
    } catch (err: unknown) {
      expect(err).toBeInstanceOf(ApiError);
      if (err instanceof ApiError) {
        expect(err.statusCode).toBe(404);
        expect(err.message).toContain('unknown_model');
      }
    }
  });

  it('handles network failure with informative connection instructions', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

    await expect(apiClient('/health')).rejects.toThrow('Ensure the FastAPI server is running on port 8000');
  });
});
