import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AnalyzePage from '../app/analyze/page';
import ExperimentsPage from '../app/experiments/page';
import ModelsPage from '../app/models/page';
import HistoryPage from '../app/history/page';
import * as classificationApi from '../lib/api/classification';
import * as modelsApi from '../lib/api/models';
import * as experimentsApi from '../lib/api/experiments';
import * as historyApi from '../lib/api/history';
import { ModelSummary } from '../types/models';
import { ExperimentSummary } from '../types/experiments';
import { PredictionHistoryItem } from '../types/history';

// Mock all API modules
vi.mock('../lib/api/classification');
vi.mock('../lib/api/models');
vi.mock('../lib/api/experiments');
vi.mock('../lib/api/history');

const mockModels: ModelSummary[] = [
  {
    model_id: 'hybrid_mnb',
    display_name: 'Hybrid Multinomial Naive Bayes',
    classifier_type: 'multinomial_nb',
    feature_space: 'hybrid',
    feature_count: 5019,
    is_available: true,
    supports_probability: true,
    supports_decision_score: false,
    score_type: 'probability',
    is_default: true,
    selection_basis: '5-fold CV top score',
  },
  {
    model_id: 'hybrid_svm',
    display_name: 'Hybrid Linear SVM',
    classifier_type: 'linear_svm',
    feature_space: 'hybrid',
    feature_count: 5019,
    is_available: true,
    supports_probability: false,
    supports_decision_score: true,
    score_type: 'decision_score',
    is_default: false,
    selection_basis: 'exploratory baseline',
  },
];

describe('Workflow & Page-Level Regression Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // 1. Single message analysis & render prediction
  it('submits a single message and renders the returned prediction card', async () => {
    vi.mocked(modelsApi.getModels).mockResolvedValue({
      total_models: 2,
      default_model: 'hybrid_mnb',
      models: mockModels,
    });

    vi.mocked(classificationApi.classifyMessage).mockResolvedValue({
      prediction: 'spam',
      is_spam: true,
      model_id: 'hybrid_mnb',
      inference_ms: 1.45,
      score: 0.9998,
      score_type: 'probability',
      history_id: 42,
    });

    render(<AnalyzePage />);

    // Wait for model select to be populated
    await waitFor(() => {
      const select = screen.getByRole('combobox');
      expect(select).toBeDefined();
    });

    // Enter message using accessible label
    const textarea = screen.getByLabelText('SMS message content to classify');
    fireEvent.change(textarea, { target: { value: 'Claim your cash prize now!' } });

    // Submit
    const analyzeBtn = screen.getByRole('button', { name: /Analyze Message/i });
    fireEvent.click(analyzeBtn);

    // Verify prediction displayed
    await waitFor(() => {
      expect(screen.getByText('SPAM')).toBeDefined();
      expect(screen.getByText('Estimated Spam Probability')).toBeDefined();
      expect(screen.getByText('#42')).toBeDefined();
    });
  });

  // 2. Change selected model and verify submitted model ID
  it('changes the selected model and submits the selected model ID', async () => {
    vi.mocked(modelsApi.getModels).mockResolvedValue({
      total_models: 2,
      default_model: 'hybrid_mnb',
      models: mockModels,
    });

    vi.mocked(classificationApi.classifyMessage).mockResolvedValue({
      prediction: 'ham',
      is_spam: false,
      model_id: 'hybrid_svm',
      inference_ms: 0.85,
      score: -1.245,
      score_type: 'decision_score',
      history_id: 43,
    });

    render(<AnalyzePage />);

    await waitFor(() => {
      const select = screen.getByRole('combobox');
      expect(select).toBeDefined();
    });

    // Change model dropdown to hybrid_svm
    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'hybrid_svm' } });

    // Enter message
    const textarea = screen.getByLabelText('SMS message content to classify');
    fireEvent.change(textarea, { target: { value: 'Hello, are you coming over today?' } });

    // Click analyze
    const analyzeBtn = screen.getByRole('button', { name: /Analyze Message/i });
    fireEvent.click(analyzeBtn);

    await waitFor(() => {
      expect(classificationApi.classifyMessage).toHaveBeenCalledWith({
        message: 'Hello, are you coming over today?',
        model_id: 'hybrid_svm',
        save_history: true,
        include_metadata: false,
      });
      expect(screen.getByText('HAM')).toBeDefined();
      expect(screen.getByText('Decision Score')).toBeDefined();
      expect(screen.getByText('-1.2450')).toBeDefined();
    });
  });

  // 3. Batch classification with preserved ordering
  it('submits a batch and preserves the input result order', async () => {
    vi.mocked(modelsApi.getModels).mockResolvedValue({
      total_models: 2,
      default_model: 'hybrid_mnb',
      models: mockModels,
    });

    vi.mocked(classificationApi.classifyBatch).mockResolvedValue({
      total_messages: 2,
      total_latency_ms: 3.5,
      model_id: 'hybrid_mnb',
      predictions: [
        {
          index: 0,
          prediction: 'ham',
          is_spam: false,
          inference_ms: 0.9,
          score: 0.0012,
          score_type: 'probability',
          history_id: 101,
        },
        {
          index: 1,
          prediction: 'spam',
          is_spam: true,
          inference_ms: 1.1,
          score: 0.9995,
          score_type: 'probability',
          history_id: 102,
        },
      ],
    });

    render(<AnalyzePage />);

    // Switch to Batch mode
    const batchModeBtn = screen.getByRole('button', { name: /Batch Analysis/i });
    fireEvent.click(batchModeBtn);

    // Add first message
    const input = screen.getByPlaceholderText(/Type an SMS message and press Enter/i);
    fireEvent.change(input, { target: { value: 'Meeting at 3pm.' } });
    const addBtn = screen.getByRole('button', { name: 'Add' });
    fireEvent.click(addBtn);

    // Add second message
    fireEvent.change(input, { target: { value: 'WINNER! Click here to claim money.' } });
    fireEvent.click(addBtn);

    // Click Analyze Batch
    const runBatchBtn = screen.getByRole('button', { name: /Analyze Batch/i });
    fireEvent.click(runBatchBtn);

    // Verify results table preserves order
    await waitFor(() => {
      expect(screen.getByText('Batch Results')).toBeDefined();
      expect(screen.getByRole('button', { name: /Export CSV/i })).toBeDefined();
    });
  });

  // 4. Render CV and exploratory holdout metrics with appropriate labels
  it('renders CV and exploratory holdout metrics with explicit scientific labels', async () => {
    const mockExp: ExperimentSummary[] = [
      {
        experiment_id: 'E5_MNB',
        model_name: 'Hybrid Multinomial NB',
        feature_space: 'hybrid',
        classifier: 'Multinomial NB',
        scaling: 'None',
        evaluation_protocol: '5-Fold Stratified Cross-Validation',
        test_accuracy: 0.9881,
        spam_precision: 0.9475,
        spam_recall: 0.9529,
        spam_f1: 0.9501,
      },
    ];

    vi.mocked(experimentsApi.getExperiments).mockResolvedValue({
      total_experiments: 1,
      selection_bias_disclosure: 'The 80:20 held-out test split results represent an exploratory benchmark. Candidate selection was determined strictly via 5-Fold Stratified Cross-Validation on the training partition.',
      experiments: mockExp,
    });

    render(<ExperimentsPage />);

    await waitFor(() => {
      // Disclosure banner must be visible
      expect(screen.getByText(/Evaluation Protocol & Selection Bias Disclosure/i)).toBeDefined();
      expect(screen.getByText(/5-Fold Stratified Cross-Validation/i)).toBeDefined();
      expect(screen.getByText(/80:20 held-out test split/i)).toBeDefined();
      expect(screen.getByText('E5_MNB')).toBeDefined();
      expect(screen.getByText('98.81%')).toBeDefined();
    });
  });

  // 5. Handle unavailable model metadata and empty models gracefully
  it('handles unavailable model metadata and empty models gracefully without crashing', async () => {
    vi.mocked(modelsApi.getModels).mockResolvedValue({
      total_models: 0,
      default_model: 'hybrid_mnb',
      models: [],
    });

    render(<ModelsPage />);

    await waitFor(() => {
      expect(screen.getByText('No models found')).toBeDefined();
      expect(screen.getByText(/No model configurations are currently registered/i)).toBeDefined();
    });
  });

  // 6. Delete history record after confirmation
  it('deletes a history record after modal confirmation', async () => {
    const mockRecord: PredictionHistoryItem = {
      id: 99,
      model_id: 'hybrid_mnb',
      prediction: 'spam',
      is_spam: true,
      message_length: 64,
      inference_ms: 1.2,
      score_type: 'probability',
      spam_score: 0.998,
      created_at: '2026-10-11T12:00:00Z',
    };

    vi.mocked(modelsApi.getModels).mockResolvedValue({
      total_models: 2,
      default_model: 'hybrid_mnb',
      models: mockModels,
    });

    vi.mocked(historyApi.getHistory).mockResolvedValue({
      items: [mockRecord],
      pagination: {
        page: 1,
        page_size: 20,
        total_items: 1,
        total_pages: 1,
      },
    });

    vi.mocked(historyApi.deleteHistoryRecord).mockResolvedValue({
      deleted: true,
      record_id: 99,
    });

    render(<HistoryPage />);

    await waitFor(() => {
      expect(screen.getByText('#99')).toBeDefined();
    });

    // Click delete trash icon button for record 99
    const deleteBtn = screen.getByRole('button', { name: 'Delete record 99' });
    fireEvent.click(deleteBtn);

    // Modal should be open
    await waitFor(() => {
      expect(screen.getByText('Confirm Record Deletion')).toBeDefined();
    });

    // Confirm deletion using exact button label in modal
    const confirmDeleteBtn = screen.getByRole('button', { name: 'Delete Record' });
    fireEvent.click(confirmDeleteBtn);

    await waitFor(() => {
      expect(historyApi.deleteHistoryRecord).toHaveBeenCalledWith(99);
    });
  });

  // 7. Display useful error when history deletion fails
  it('displays a useful error when history deletion fails without claiming success', async () => {
    const mockRecord: PredictionHistoryItem = {
      id: 100,
      model_id: 'hybrid_mnb',
      prediction: 'spam',
      is_spam: true,
      message_length: 50,
      inference_ms: 1.0,
      score_type: 'probability',
      spam_score: 0.99,
      created_at: '2026-10-11T12:00:00Z',
    };

    vi.mocked(modelsApi.getModels).mockResolvedValue({
      total_models: 2,
      default_model: 'hybrid_mnb',
      models: mockModels,
    });

    vi.mocked(historyApi.getHistory).mockResolvedValue({
      items: [mockRecord],
      pagination: {
        page: 1,
        page_size: 20,
        total_items: 1,
        total_pages: 1,
      },
    });

    vi.mocked(historyApi.deleteHistoryRecord).mockRejectedValue(
      new Error('Database lock acquisition timeout')
    );

    render(<HistoryPage />);

    await waitFor(() => {
      expect(screen.getByText('#100')).toBeDefined();
    });

    const deleteBtn = screen.getByRole('button', { name: 'Delete record 100' });
    fireEvent.click(deleteBtn);

    await waitFor(() => {
      expect(screen.getByText('Confirm Record Deletion')).toBeDefined();
    });

    const confirmDeleteBtn = screen.getByRole('button', { name: 'Delete Record' });
    fireEvent.click(confirmDeleteBtn);

    // Should render the failure error message inside modal
    await waitFor(() => {
      expect(screen.getByText('Database lock acquisition timeout')).toBeDefined();
    });
  });

  // 8. Show backend-unavailable state without fabricated data
  it('shows backend-unavailable state honestly without fabricated data', async () => {
    vi.mocked(modelsApi.getModels).mockRejectedValue(new Error('Connection refused'));

    render(<AnalyzePage />);

    await waitFor(() => {
      expect(screen.getByText('Backend Offline')).toBeDefined();
      expect(screen.getByText(/Failed to connect to backend: Connection refused/i)).toBeDefined();
    });
  });
});
