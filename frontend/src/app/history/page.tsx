'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { LoadingState } from '@/components/ui/LoadingState';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import { Modal } from '@/components/ui/Modal';
import { getHistory, deleteHistoryRecord } from '@/lib/api/history';
import { getModels } from '@/lib/api/models';
import { formatDateTime, formatLatency, formatScore } from '@/lib/formatters';
import {
  ChevronLeft,
  ChevronRight,
  History,
  Lock,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import { PageMeta } from '@/types/api';
import { PredictionHistoryItem } from '@/types/history';
import { ModelSummary } from '@/types/models';

export default function HistoryPage() {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Table items & pagination
  const [records, setRecords] = useState<PredictionHistoryItem[]>([]);
  const [pagination, setPagination] = useState<PageMeta>({
    page: 1,
    page_size: 20,
    total_items: 0,
    total_pages: 1,
  });

  // Filter states
  const [predictionFilter, setPredictionFilter] = useState<'all' | 'spam' | 'ham'>('all');
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [models, setModels] = useState<ModelSummary[]>([]);

  // Record Inspection Modal
  const [inspectRecord, setInspectRecord] = useState<PredictionHistoryItem | null>(null);

  // Delete Confirmation Modal
  const [recordToDelete, setRecordToDelete] = useState<PredictionHistoryItem | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Load models for filter dropdown
  useEffect(() => {
    async function loadModelsList() {
      try {
        const res = await getModels();
        setModels(res.models);
      } catch {
        // non-fatal
      }
    }
    loadModelsList();
  }, []);

  // Fetch history with active filters
  const fetchHistoryRecords = useCallback(
    async (pageIndex: number, pageSize: number = 20) => {
      try {
        setLoading(true);
        setError(null);
        const res = await getHistory({
          page: pageIndex,
          page_size: pageSize,
          prediction: predictionFilter === 'all' ? undefined : predictionFilter,
          model_id: modelFilter === 'all' ? undefined : modelFilter,
        });
        setRecords(res.items);
        setPagination(res.pagination);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to query prediction history.');
      } finally {
        setLoading(false);
      }
    },
    [predictionFilter, modelFilter]
  );

  useEffect(() => {
    fetchHistoryRecords(1, 20);
  }, [fetchHistoryRecords]);

  // Handle record deletion
  const handleDeleteConfirm = async () => {
    if (!recordToDelete) return;
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await deleteHistoryRecord(recordToDelete.id);
      setRecordToDelete(null);
      await fetchHistoryRecords(pagination.page, pagination.page_size);
    } catch (err: unknown) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete record.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="pb-4 border-b border-slate-200">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
          Prediction History
        </h1>
        <p className="text-[15px] text-slate-500 mt-1">
          Audit log and latency records of SMS classifications persisted in SQLite.
        </p>
      </div>

      {/* Compact Filters Bar */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {/* Result Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-medium">Result:</span>
            {(['all', 'spam', 'ham'] as const).map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setPredictionFilter(r)}
                className={`px-2.5 py-1 rounded-md text-xs transition-colors capitalize ${
                  predictionFilter === r
                    ? 'bg-[#1E3A8A] text-white font-medium'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {r}
              </button>
            ))}
          </div>

          {/* Model Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-medium">Model:</span>
            <select
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              className="px-2.5 py-1 text-xs bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:ring-1 focus:ring-[#1E3A8A] font-medium text-slate-800"
            >
              <option value="all">All Models</option>
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.display_name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Refresh Action */}
        <div className="flex items-center gap-2 self-end sm:self-auto">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => fetchHistoryRecords(pagination.page, pagination.page_size)}
            isLoading={loading}
            leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
            className="text-xs text-slate-600 hover:text-slate-900"
          >
            Refresh
          </Button>
        </div>
      </div>

      {error && <ErrorState message={error} />}

      {/* Main History Table */}
      {loading && records.length === 0 ? (
        <LoadingState label="Retrieving prediction audit records..." className="py-20" />
      ) : records.length === 0 ? (
        <EmptyState
          title="No history records found"
          description="Classification records saved during single or batch analysis will appear here."
          icon={<History className="w-10 h-10 text-slate-300" />}
        />
      ) : (
        <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs overflow-hidden">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
              <tr>
                <th className="p-3 w-16">ID</th>
                <th className="p-3">Prediction</th>
                <th className="p-3">Model</th>
                <th className="p-3">Length</th>
                <th className="p-3">Inference Latency</th>
                <th className="p-3">Timestamp</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {records.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => setInspectRecord(r)}
                  className="hover:bg-slate-50/70 cursor-pointer transition-colors"
                >
                  <td className="p-3 font-mono text-slate-500 tabular-nums">#{r.id}</td>
                  <td className="p-3">
                    <Badge variant={r.is_spam ? 'spam' : 'ham'} size="sm">
                      {r.prediction.toUpperCase()}
                    </Badge>
                  </td>
                  <td className="p-3 font-mono font-medium text-slate-900">
                    {r.model_id}
                  </td>
                  <td className="p-3 font-mono text-slate-600 tabular-nums">
                    {r.message_length ? `${r.message_length} chars` : '—'}
                  </td>
                  <td className="p-3 font-mono text-slate-600 tabular-nums">
                    {formatLatency(r.inference_ms)}
                  </td>
                  <td className="p-3 text-slate-500 tabular-nums">
                    {formatDateTime(r.created_at)}
                  </td>
                  <td className="p-3 text-right">
                    <div className="flex items-center justify-end gap-1.5" onClick={(e) => e.stopPropagation()}>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setInspectRecord(r)}
                        className="text-xs text-slate-600 hover:text-slate-900"
                      >
                        Inspect
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setRecordToDelete(r);
                          setDeleteError(null);
                        }}
                        aria-label={`Delete record ${r.id}`}
                        className="text-xs text-slate-400 hover:text-red-600 p-1.5"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Server-Side Pagination Bar */}
          <div className="p-3.5 border-t border-slate-100 bg-slate-50/60 flex items-center justify-between text-xs text-slate-600">
            <span className="tabular-nums">
              Page <strong className="text-slate-900 font-semibold">{pagination.page}</strong> of{' '}
              <strong className="text-slate-900 font-semibold">{pagination.total_pages}</strong> ({pagination.total_items} total)
            </span>

            <div className="flex items-center gap-1.5">
              <Button
                variant="outline"
                size="sm"
                disabled={pagination.page <= 1}
                onClick={() => fetchHistoryRecords(pagination.page - 1, pagination.page_size)}
                leftIcon={<ChevronLeft className="w-3.5 h-3.5" />}
                className="text-xs h-7 px-2.5"
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={pagination.page >= pagination.total_pages}
                onClick={() => fetchHistoryRecords(pagination.page + 1, pagination.page_size)}
                rightIcon={<ChevronRight className="w-3.5 h-3.5" />}
                className="text-xs h-7 px-2.5"
              >
                Next
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Record Inspection Modal */}
      <Modal
        isOpen={inspectRecord !== null}
        onClose={() => setInspectRecord(null)}
        title={inspectRecord ? `Prediction Audit Record #${inspectRecord.id}` : 'Record Details'}
        maxWidth="md"
      >
        {inspectRecord && (
          <div className="space-y-4 text-xs text-slate-700">
            <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-200">
              <div>
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Prediction Result</span>
                <div className="flex items-center gap-2 mt-0.5">
                  <Badge variant={inspectRecord.is_spam ? 'spam' : 'ham'} size="md">
                    {inspectRecord.prediction.toUpperCase()}
                  </Badge>
                  <span className="text-xs font-semibold text-slate-800">
                    {inspectRecord.is_spam ? 'Classified as Spam' : 'Classified as Legitimate'}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Timestamp</span>
                <span className="font-mono text-slate-600 tabular-nums">
                  {formatDateTime(inspectRecord.created_at)}
                </span>
              </div>
            </div>

            {/* Score interpretation */}
            {(() => {
              const scoreDisplay = formatScore(inspectRecord.spam_score, inspectRecord.score_type);
              if (!scoreDisplay) return null;
              return (
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="font-semibold text-slate-700">{scoreDisplay.label}:</span>
                    <span className="font-mono font-bold text-slate-900 tabular-nums">{scoreDisplay.formattedValue}</span>
                  </div>
                  <p className="text-[11px] text-slate-500">{scoreDisplay.interpretation}</p>
                </div>
              );
            })()}

            {/* Operational details */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
                <span className="text-[10px] text-slate-400 block font-semibold">Model ID</span>
                <span className="font-mono text-slate-800 font-semibold truncate block mt-0.5">{inspectRecord.model_id}</span>
              </div>
              <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
                <span className="text-[10px] text-slate-400 block font-semibold">Latency</span>
                <span className="font-mono text-slate-800 font-semibold block mt-0.5 tabular-nums">{formatLatency(inspectRecord.inference_ms)}</span>
              </div>
            </div>

            {/* Message Text or Privacy Preservation Notice */}
            <div className="space-y-1.5 pt-2 border-t border-slate-100">
              <span className="text-[11px] font-semibold text-slate-700 block">Message Content:</span>
              {inspectRecord.message_text ? (
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-slate-800 leading-relaxed font-sans">
                  {inspectRecord.message_text}
                </div>
              ) : (
                <div className="p-3 bg-slate-50 rounded-lg border border-dashed border-slate-300 text-slate-500 flex items-center gap-2">
                  <Lock className="w-4 h-4 text-slate-400 shrink-0" />
                  <span>Message text not stored (privacy preservation mode enabled).</span>
                </div>
              )}
            </div>
          </div>
        )}
      </Modal>

      {/* Restrained Delete Confirmation Modal */}
      <Modal
        isOpen={recordToDelete !== null}
        onClose={() => setRecordToDelete(null)}
        title="Confirm Record Deletion"
        maxWidth="sm"
      >
        {recordToDelete && (
          <div className="space-y-4 text-xs text-slate-600">
            <p className="leading-relaxed">
              Are you sure you want to permanently delete audit record <strong className="text-slate-900 font-semibold font-mono">#{recordToDelete.id}</strong>? This action will remove the entry from SQLite.
            </p>

            {deleteError && (
              <div className="p-2.5 bg-red-50 border border-red-200 rounded text-red-700 font-medium">
                {deleteError}
              </div>
            )}

            <div className="flex items-center justify-end gap-2 pt-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setRecordToDelete(null)}
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <Button
                variant="danger"
                size="sm"
                onClick={handleDeleteConfirm}
                isLoading={isDeleting}
              >
                Delete Record
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
