'use client';

import React, { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { LoadingState } from '@/components/ui/LoadingState';
import { ErrorState } from '@/components/ui/ErrorState';
import { classifyMessage, classifyBatch } from '@/lib/api/classification';
import { getModels } from '@/lib/api/models';
import {
  MAX_MESSAGE_LENGTH,
  MAX_BATCH_SIZE,
  SAMPLE_MESSAGES,
} from '@/lib/constants';
import { formatLatency, formatScore } from '@/lib/formatters';
import {
  ChevronDown,
  ChevronUp,
  Download,
  Play,
  Plus,
  ShieldAlert,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import { ClassifyResponse, BatchClassifyResponse } from '@/types/classification';
import { ModelSummary } from '@/types/models';

export default function AnalyzePage() {
  const [mode, setMode] = useState<'single' | 'batch'>('single');

  // Models catalog
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('hybrid_mnb');
  const [loadingModels, setLoadingModels] = useState<boolean>(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Single mode state
  const [message, setMessage] = useState<string>('');
  const [saveHistory, setSaveHistory] = useState<boolean>(true);
  const [includeMetadata, setIncludeMetadata] = useState<boolean>(false);
  const [analyzingSingle, setAnalyzingSingle] = useState<boolean>(false);
  const [singleResult, setSingleResult] = useState<ClassifyResponse | null>(null);
  const [singleError, setSingleError] = useState<string | null>(null);
  const [showMetadataFeatures, setShowMetadataFeatures] = useState<boolean>(false);

  // Batch mode state
  const [batchMessages, setBatchMessages] = useState<string[]>([
    'URGENT! You have won a 1 week FREE holiday to Spain or £1000 cash! Call 09061701461 to claim.',
    'Are we still having lunch at the cafeteria today at 12:30?',
  ]);
  const [newBatchItem, setNewBatchItem] = useState<string>('');
  const [analyzingBatch, setAnalyzingBatch] = useState<boolean>(false);
  const [batchResult, setBatchResult] = useState<BatchClassifyResponse | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchFilter, setBatchFilter] = useState<'all' | 'spam' | 'ham'>('all');

  // Load models on mount
  useEffect(() => {
    async function fetchModelsCatalog() {
      try {
        setModelsError(null);
        const res = await getModels();
        setModels(res.models);
        if (res.default_model) {
          setSelectedModel(res.default_model);
        }
      } catch (err: unknown) {
        setModelsError(err instanceof Error ? err.message : 'Connection refused');
      } finally {
        setLoadingModels(false);
      }
    }
    fetchModelsCatalog();

    // Check URL parameters for direct workflow testing
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      if (params.get('mode') === 'batch') {
        setMode('batch');
        if (params.get('auto') === 'true') {
          setTimeout(() => {
            classifyBatch({
              messages: [
                'URGENT! You have won a 1 week FREE holiday to Spain or £1000 cash! Call 09061701461 to claim.',
                'Are we still having lunch at the cafeteria today at 12:30?',
              ],
              model_id: 'hybrid_mnb',
              save_history: true,
            }).then((res) => setBatchResult(res)).catch(() => {});
          }, 400);
        }
      } else if (params.get('sample') === 'spam') {
        const sampleText = SAMPLE_MESSAGES[0].text;
        setMessage(sampleText);
        if (params.get('auto') === 'true') {
          setTimeout(() => {
            classifyMessage({
              message: sampleText,
              model_id: 'hybrid_mnb',
              save_history: true,
              include_metadata: false,
            }).then((res) => setSingleResult(res)).catch(() => {});
          }, 400);
        }
      }
    }
  }, []);

  // Handle single classification
  const handleSingleClassify = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) {
      setSingleError('Please enter a message to analyze.');
      return;
    }
    if (message.length > MAX_MESSAGE_LENGTH) {
      setSingleError(`Message exceeds maximum limit of ${MAX_MESSAGE_LENGTH.toLocaleString()} characters.`);
      return;
    }

    setAnalyzingSingle(true);
    setSingleError(null);
    setSingleResult(null);

    try {
      const res = await classifyMessage({
        message: message.trim(),
        model_id: selectedModel,
        save_history: saveHistory,
        include_metadata: includeMetadata,
      });
      setSingleResult(res);
    } catch (err: unknown) {
      setSingleError(err instanceof Error ? err.message : 'Message classification failed.');
    } finally {
      setAnalyzingSingle(false);
    }
  };

  // Handle batch classification
  const handleBatchClassify = async () => {
    const validMessages = batchMessages.filter((m) => m.trim().length > 0);
    if (validMessages.length === 0) {
      setBatchError('Add at least one message to the batch.');
      return;
    }
    if (validMessages.length > MAX_BATCH_SIZE) {
      setBatchError(`Batch size cannot exceed ${MAX_BATCH_SIZE} messages.`);
      return;
    }

    setAnalyzingBatch(true);
    setBatchError(null);
    setBatchResult(null);

    try {
      const res = await classifyBatch({
        messages: validMessages,
        model_id: selectedModel,
        save_history: saveHistory,
      });
      setBatchResult(res);
    } catch (err: unknown) {
      setBatchError(err instanceof Error ? err.message : 'Batch classification failed.');
    } finally {
      setAnalyzingBatch(false);
    }
  };

  const handleAddBatchItem = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newBatchItem.trim()) return;
    if (batchMessages.length >= MAX_BATCH_SIZE) {
      setBatchError(`Maximum limit of ${MAX_BATCH_SIZE} batch items reached.`);
      return;
    }
    setBatchMessages([...batchMessages, newBatchItem.trim()]);
    setNewBatchItem('');
    setBatchError(null);
  };

  const handleRemoveBatchItem = (index: number) => {
    setBatchMessages(batchMessages.filter((_, i) => i !== index));
  };

  const handleExportBatchCSV = () => {
    if (!batchResult || !batchResult.predictions) return;
    const headers = ['index', 'prediction', 'is_spam', 'score', 'score_type', 'latency_ms', 'history_id'];
    const rows = batchResult.predictions.map((p) => [
      p.index,
      p.prediction,
      p.is_spam ? 'true' : 'false',
      p.score ?? '',
      p.score_type ?? '',
      p.inference_ms ?? '',
      p.history_id ?? '',
    ]);
    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `spam_batch_results_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const selectedModelDetail = models.find((m) => m.model_id === selectedModel);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
            Analyze message
          </h1>
          <p className="text-[15px] text-slate-500 mt-1">
            Check a message for potential spam using a lightweight classification model.
          </p>
        </div>

        {/* Secondary Mode Switcher */}
        <div className="inline-flex p-1 bg-slate-100 rounded-lg shrink-0 self-start sm:self-auto border border-slate-200/60">
          <button
            type="button"
            onClick={() => setMode('single')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              mode === 'single'
                ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Single message
          </button>
          <button
            type="button"
            onClick={() => setMode('batch')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              mode === 'batch'
                ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Batch analysis {batchMessages.length > 0 && `(${batchMessages.length})`}
          </button>
        </div>
      </div>

      {modelsError && (
        <ErrorState
          title="Backend Offline"
          message={`Failed to connect to backend: ${modelsError}. Showing offline state without fabricated data.`}
        />
      )}

      {/* SINGLE MESSAGE WORKFLOW (PRIMARY EXPERIENCE) */}
      {mode === 'single' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs p-5 sm:p-6 space-y-5">
            {/* Top Control Bar: Compact Model Selector & Quick Sample Buttons */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
              <div className="flex flex-wrap items-center gap-2.5">
                <label htmlFor="model-select" className="text-xs font-semibold text-slate-700">
                  Model:
                </label>
                <select
                  id="model-select"
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    if (singleResult) setSingleResult(null);
                  }}
                  disabled={loadingModels || models.length === 0}
                  className="px-3 py-1.5 text-xs bg-white border border-slate-300 rounded-md shadow-2xs focus:outline-none focus:ring-2 focus:ring-[#1E3A8A] font-medium text-slate-800"
                >
                  {models.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.display_name} {m.is_default ? '★ Default' : ''}
                    </option>
                  ))}
                </select>

                {selectedModelDetail && (
                  <span className="text-[11px] text-slate-500 font-mono hidden sm:inline">
                    · {selectedModelDetail.feature_space} ({selectedModelDetail.feature_count.toLocaleString()} feats)
                  </span>
                )}
              </div>

              {/* 3 Optional Sample Message Buttons */}
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-slate-400 mr-1 hidden lg:inline">Try sample:</span>
                {SAMPLE_MESSAGES.slice(0, 3).map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => {
                      setMessage(sample.text);
                      setSingleError(null);
                      if (singleResult) setSingleResult(null);
                    }}
                    className="px-2.5 py-1 text-xs rounded bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors"
                  >
                    {sample.title.split(' (')[0]}
                  </button>
                ))}
              </div>
            </div>

            {/* Core Message Editor */}
            <form onSubmit={handleSingleClassify} className="space-y-4">
              <div>
                <textarea
                  id="message-text"
                  rows={5}
                  value={message}
                  onChange={(e) => {
                    setMessage(e.target.value);
                    if (singleError) setSingleError(null);
                  }}
                  placeholder="Paste or type an SMS message to evaluate (e.g., URGENT! You have won £1000 cash...)"
                  className="w-full p-4 text-[15px] text-slate-900 border border-slate-300 rounded-lg shadow-2xs focus:outline-none focus:ring-2 focus:ring-[#1E3A8A] focus:border-[#1E3A8A] font-sans resize-y leading-relaxed"
                  maxLength={MAX_MESSAGE_LENGTH}
                  aria-label="SMS message content to classify"
                />

                {/* Editor Bottom Info & Actions */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2 text-xs text-slate-500">
                  <div className="flex items-center gap-4">
                    <span className="tabular-nums font-mono text-[11px] text-slate-400">
                      {message.length.toLocaleString()} / {MAX_MESSAGE_LENGTH.toLocaleString()} chars
                    </span>
                    <label className="flex items-center gap-1.5 cursor-pointer select-none text-[12px] text-slate-600">
                      <input
                        type="checkbox"
                        checked={saveHistory}
                        onChange={(e) => setSaveHistory(e.target.checked)}
                        className="rounded border-slate-300 text-[#1E3A8A] focus:ring-[#1E3A8A]"
                      />
                      <span>Save record to history</span>
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer select-none text-[12px] text-slate-600 hidden md:flex">
                      <input
                        type="checkbox"
                        checked={includeMetadata}
                        onChange={(e) => setIncludeMetadata(e.target.checked)}
                        className="rounded border-slate-300 text-[#1E3A8A] focus:ring-[#1E3A8A]"
                      />
                      <span>Extract 19 non-textual features</span>
                    </label>
                  </div>

                  <div className="flex items-center gap-2 self-end sm:self-auto">
                    {message && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setMessage('');
                          setSingleResult(null);
                          setSingleError(null);
                        }}
                        className="text-slate-500 hover:text-slate-800"
                      >
                        Clear
                      </Button>
                    )}

                    <Button
                      type="submit"
                      isLoading={analyzingSingle}
                      leftIcon={<Play className="w-3.5 h-3.5 fill-current" />}
                      disabled={!message.trim()}
                      className="bg-[#1E3A8A] hover:bg-[#172554] text-white px-5 py-2 text-sm shadow-xs"
                    >
                      Analyze Message
                    </Button>
                  </div>
                </div>
              </div>
            </form>

            {singleError && <ErrorState message={singleError} />}
          </div>

          {/* RESULTS REGION */}
          {analyzingSingle ? (
            <div className="bg-white rounded-xl border border-slate-200/80 p-8 text-center shadow-2xs">
              <LoadingState label="Extracting features & evaluating model pipeline..." />
            </div>
          ) : singleResult ? (
            <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs p-6 space-y-5 animate-in fade-in duration-200">
              {/* Prominent semantic result banner */}
              <div
                className={`p-4 rounded-lg border flex items-start justify-between gap-4 ${
                  singleResult.is_spam
                    ? 'bg-[#FEF2F2] border-[#FECACA] text-[#991B1B]'
                    : 'bg-[#ECFDF5] border-[#A7F3D0] text-[#065F46]'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className="p-1.5 rounded-full bg-white shadow-2xs shrink-0 mt-0.5">
                    {singleResult.is_spam ? (
                      <ShieldAlert className="w-5 h-5 text-red-600" />
                    ) : (
                      <ShieldCheck className="w-5 h-5 text-emerald-600" />
                    )}
                  </div>
                  <div>
                    <div className="flex items-center gap-2.5">
                      <span className="text-lg font-bold tracking-tight">
                        {singleResult.is_spam ? 'Likely Spam' : 'Likely Legitimate'}
                      </span>
                      <Badge variant={singleResult.is_spam ? 'spam' : 'ham'} size="sm">
                        {singleResult.prediction.toUpperCase()}
                      </Badge>
                    </div>
                    <p className="text-xs mt-1 text-slate-600 leading-relaxed max-w-xl">
                      {singleResult.is_spam
                        ? 'The model detected elevated urgency and promotional patterns typical of SMS spam.'
                        : 'The model detected standard interpersonal communication characteristics.'}
                    </p>
                  </div>
                </div>

                {singleResult.history_id && (
                  <span className="text-xs font-mono font-medium text-slate-500 shrink-0 tabular-nums">
                    #{singleResult.history_id}
                  </span>
                )}
              </div>

              {/* Exact Metrics Strip */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                {/* Score Semantics Card */}
                {(() => {
                  const scoreDisplay = formatScore(
                    singleResult.score,
                    singleResult.score_type,
                    singleResult.probability
                  );
                  if (!scoreDisplay) return null;
                  return (
                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/80 space-y-1">
                      <span className="text-[11px] text-slate-500 uppercase tracking-wider block font-semibold">
                        {scoreDisplay.label}
                      </span>
                      <span className="font-mono text-sm font-bold text-slate-900 block tabular-nums">
                        {scoreDisplay.formattedValue}
                      </span>
                      <p className="text-[11px] text-slate-500 leading-tight">
                        {scoreDisplay.interpretation}
                      </p>
                    </div>
                  );
                })()}

                {/* Active Model */}
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/80 space-y-1">
                  <span className="text-[11px] text-slate-500 uppercase tracking-wider block font-semibold">
                    Evaluating Model
                  </span>
                  <span className="font-mono text-sm font-semibold text-slate-900 block truncate">
                    {singleResult.model_id}
                  </span>
                  <p className="text-[11px] text-slate-500">
                    {selectedModelDetail?.feature_space || 'hybrid'} representation
                  </p>
                </div>

                {/* Latency */}
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/80 space-y-1">
                  <span className="text-[11px] text-slate-500 uppercase tracking-wider block font-semibold">
                    Inference Latency
                  </span>
                  <span className="font-mono text-sm font-bold text-slate-900 block tabular-nums">
                    {formatLatency(singleResult.inference_ms)}
                  </span>
                  <p className="text-[11px] text-slate-500">
                    Feature extraction & classification time
                  </p>
                </div>
              </div>

              {/* Optional Handcrafted Metadata Vector */}
              {singleResult.metadata_features && Object.keys(singleResult.metadata_features).length > 0 && (
                <div className="pt-2 border-t border-slate-100">
                  <button
                    type="button"
                    onClick={() => setShowMetadataFeatures(!showMetadataFeatures)}
                    className="flex items-center justify-between w-full text-xs font-semibold text-slate-700 hover:text-slate-900 py-1"
                  >
                    <span>Extracted Metadata Features ({Object.keys(singleResult.metadata_features).length} dimensions)</span>
                    {showMetadataFeatures ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>

                  {showMetadataFeatures && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
                      {Object.entries(singleResult.metadata_features).map(([key, val]) => (
                        <div key={key} className="flex justify-between items-center py-0.5 border-b border-slate-100">
                          <span className="text-slate-500 truncate mr-2" title={key}>{key}:</span>
                          <span className="font-mono font-medium text-slate-900 tabular-nums">
                            {typeof val === 'number' ? val.toFixed(2) : String(val)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            /* Restrained, Compact Empty State */
            <div className="py-8 px-4 rounded-xl border border-dashed border-slate-200 bg-white/60 text-center">
              <p className="text-xs text-slate-500">
                Enter or paste a message above and click <span className="font-semibold text-slate-700">Analyze Message</span> to view classification results.
              </p>
            </div>
          )}
        </div>
      )}

      {/* BATCH ANALYSIS WORKFLOW (SECONDARY MODE) */}
      {mode === 'batch' && (
        <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs p-5 sm:p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
            <div>
              <h2 className="text-base font-semibold text-slate-900">Batch SMS Analysis</h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Classify multiple SMS messages in a single atomic request (up to {MAX_BATCH_SIZE} items).
              </p>
            </div>

            <div className="flex items-center gap-2">
              <label htmlFor="batch-model-select" className="text-xs font-semibold text-slate-700">
                Model:
              </label>
              <select
                id="batch-model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="px-3 py-1.5 text-xs bg-white border border-slate-300 rounded-md shadow-2xs text-slate-800"
              >
                {models.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.display_name} {m.is_default ? '★ Default' : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Add Message Form */}
          <form onSubmit={handleAddBatchItem} className="flex gap-2">
            <input
              type="text"
              value={newBatchItem}
              onChange={(e) => setNewBatchItem(e.target.value)}
              placeholder="Type an SMS message and press Enter or click Add..."
              className="flex-1 px-3.5 py-2 text-xs border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1E3A8A]"
            />
            <Button type="submit" size="sm" variant="secondary" leftIcon={<Plus className="w-3.5 h-3.5" />}>
              Add
            </Button>
          </form>

          {/* Message List */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
              <span>Queued Messages ({batchMessages.length})</span>
              {batchMessages.length > 0 && (
                <button
                  type="button"
                  onClick={() => setBatchMessages([])}
                  className="text-red-600 hover:text-red-700 hover:underline"
                >
                  Clear all
                </button>
              )}
            </div>

            <div className="max-h-56 overflow-y-auto border border-slate-200 rounded-lg divide-y divide-slate-100 bg-slate-50/50">
              {batchMessages.map((msg, idx) => (
                <div key={idx} className="p-2.5 flex items-start justify-between gap-3 text-xs hover:bg-white transition-colors">
                  <span className="font-mono text-slate-400 w-5 shrink-0 tabular-nums">#{idx + 1}</span>
                  <span className="flex-1 text-slate-800 font-sans truncate">{msg}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveBatchItem(idx)}
                    className="text-slate-400 hover:text-red-600 p-0.5"
                    aria-label={`Remove message ${idx + 1}`}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Batch Submit Bar */}
          <div className="flex items-center justify-between pt-2">
            <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
              <input
                type="checkbox"
                checked={saveHistory}
                onChange={(e) => setSaveHistory(e.target.checked)}
                className="rounded border-slate-300 text-[#1E3A8A] focus:ring-[#1E3A8A]"
              />
              <span>Save batch records in SQLite history</span>
            </label>

            <Button
              type="button"
              onClick={handleBatchClassify}
              isLoading={analyzingBatch}
              disabled={batchMessages.length === 0}
              className="bg-[#1E3A8A] hover:bg-[#172554] text-white px-5 py-2 text-sm shadow-xs"
            >
              Analyze Batch
            </Button>
          </div>

          {batchError && <ErrorState message={batchError} />}

          {/* Batch Results Table */}
          {batchResult && (
            <div className="pt-4 border-t border-slate-100 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">Batch Results</h3>
                  <p className="text-xs text-slate-500 tabular-nums">
                    Processed {batchResult.total_messages} messages in {formatLatency(batchResult.total_latency_ms)}
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <div className="flex items-center bg-slate-100 rounded p-0.5 text-xs">
                    {(['all', 'spam', 'ham'] as const).map((f) => (
                      <button
                        key={f}
                        type="button"
                        onClick={() => setBatchFilter(f)}
                        className={`px-2.5 py-1 rounded capitalize ${
                          batchFilter === f ? 'bg-white font-semibold text-slate-900 shadow-2xs' : 'text-slate-600'
                        }`}
                      >
                        {f}
                      </button>
                    ))}
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleExportBatchCSV}
                    leftIcon={<Download className="w-3.5 h-3.5" />}
                  >
                    Export CSV
                  </Button>
                </div>
              </div>

              <div className="border border-slate-200 rounded-lg overflow-hidden">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
                    <tr>
                      <th className="p-3 w-12">#</th>
                      <th className="p-3">Prediction</th>
                      <th className="p-3">Score / Margin</th>
                      <th className="p-3">Latency</th>
                      <th className="p-3">Audit ID</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {batchResult.predictions
                      .filter((p) => {
                        if (batchFilter === 'spam') return p.is_spam;
                        if (batchFilter === 'ham') return !p.is_spam;
                        return true;
                      })
                      .map((p) => {
                        const scoreDisplay = formatScore(p.score, p.score_type);
                        return (
                          <tr key={p.index} className="hover:bg-slate-50/50">
                            <td className="p-3 font-mono text-slate-400 tabular-nums">#{p.index + 1}</td>
                            <td className="p-3">
                              <Badge variant={p.is_spam ? 'spam' : 'ham'} size="sm">
                                {p.prediction.toUpperCase()}
                              </Badge>
                            </td>
                            <td className="p-3 font-mono tabular-nums">
                              {scoreDisplay ? scoreDisplay.formattedValue : '—'}
                            </td>
                            <td className="p-3 font-mono tabular-nums text-slate-600">
                              {formatLatency(p.inference_ms)}
                            </td>
                            <td className="p-3 font-mono text-slate-400 tabular-nums">
                              {p.history_id ? `#${p.history_id}` : '—'}
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
