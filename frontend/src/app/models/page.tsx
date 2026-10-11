'use client';

import React, { useEffect, useState, useMemo } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/Button';
import { LoadingState } from '@/components/ui/LoadingState';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import { Modal } from '@/components/ui/Modal';
import { getModels, getModelDetail } from '@/lib/api/models';
import { getFeatureSpaceLabel } from '@/lib/formatters';
import {
  ArrowRight,
  Boxes,
  CheckCircle,
  Search,
  XCircle,
} from 'lucide-react';
import { ModelDetail, ModelSummary } from '@/types/models';

export default function ModelsPage() {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [defaultModel, setDefaultModel] = useState<string>('hybrid_mnb');

  // Search & Filter
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [spaceFilter, setSpaceFilter] = useState<string>('all');
  const [classifierFilter, setClassifierFilter] = useState<string>('all');

  // Modal
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [modelDetail, setModelDetail] = useState<ModelDetail | null>(null);

  useEffect(() => {
    async function loadModelsCatalog() {
      try {
        setLoading(true);
        setError(null);
        const res = await getModels();
        setModels(res.models);
        setDefaultModel(res.default_model);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to retrieve models catalog.');
      } finally {
        setLoading(false);
      }
    }
    loadModelsCatalog();
  }, []);

  const handleOpenDetail = async (modelId: string) => {
    setSelectedModelId(modelId);
    setDetailLoading(true);
    try {
      const res = await getModelDetail(modelId);
      setModelDetail(res);
    } catch (err) {
      console.error('Failed to load model details:', err);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleCloseDetail = () => {
    setSelectedModelId(null);
    setModelDetail(null);
  };

  const filteredModels = useMemo(() => {
    return models.filter((m) => {
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesName = m.display_name.toLowerCase().includes(q);
        const matchesId = m.model_id.toLowerCase().includes(q);
        const matchesClassifier = m.classifier_type.toLowerCase().includes(q);
        if (!matchesName && !matchesId && !matchesClassifier) return false;
      }
      if (spaceFilter !== 'all' && m.feature_space !== spaceFilter) return false;
      if (classifierFilter !== 'all') {
        if (!m.classifier_type.toLowerCase().includes(classifierFilter.toLowerCase())) return false;
      }
      return true;
    });
  }, [models, searchQuery, spaceFilter, classifierFilter]);

  if (loading) {
    return <LoadingState label="Reading model registry and verifying artifact readiness..." className="py-24" />;
  }

  if (error) {
    return <ErrorState message={error} />;
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="pb-4 border-b border-slate-200">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
          Model Catalog
        </h1>
        <p className="text-[15px] text-slate-500 mt-1">
          Registered estimators, feature spaces, and artifact readiness available for real-time inference.
        </p>
      </div>

      {/* Search and Filters Bar */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        {/* Search Field */}
        <div className="relative flex-1 max-w-sm">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search by model name or algorithm..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1E3A8A] focus:bg-white transition-all"
          />
        </div>

        {/* Filter Tabs */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-medium">Space:</span>
            {(['all', 'textual', 'metadata', 'hybrid'] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSpaceFilter(s)}
                className={`px-2.5 py-1 rounded-md text-xs transition-colors ${
                  spaceFilter === s
                    ? 'bg-[#1E3A8A] text-white font-medium'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {s === 'all' ? 'All' : s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-medium">Classifier:</span>
            {[
              { id: 'all', label: 'All' },
              { id: 'nb', label: 'Naive Bayes' },
              { id: 'svm', label: 'SVM' },
            ].map((c) => (
              <button
                key={c.id}
                type="button"
                onClick={() => setClassifierFilter(c.id)}
                className={`px-2.5 py-1 rounded-md text-xs transition-colors ${
                  classifierFilter === c.id
                    ? 'bg-[#1E3A8A] text-white font-medium'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Model Table */}
      {filteredModels.length === 0 ? (
        <EmptyState
          title="No models found"
          description={
            searchQuery || spaceFilter !== 'all' || classifierFilter !== 'all'
              ? 'No registered models match the active search filters.'
              : 'No model configurations are currently registered.'
          }
          icon={<Boxes className="w-10 h-10 text-slate-300" />}
        />
      ) : (
        <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs overflow-hidden">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
              <tr>
                <th className="p-3">Model Name</th>
                <th className="p-3">Classifier</th>
                <th className="p-3">Feature Representation</th>
                <th className="p-3">Readiness</th>
                <th className="p-3">Designation</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredModels.map((m) => {
                const isDefault = m.model_id === defaultModel || m.is_default;
                return (
                  <tr
                    key={m.model_id}
                    onClick={() => handleOpenDetail(m.model_id)}
                    className="hover:bg-slate-50/70 cursor-pointer transition-colors"
                  >
                    <td className="p-3">
                      <div className="font-semibold text-slate-900">{m.display_name}</div>
                      <span className="font-mono text-[11px] text-slate-400">{m.model_id}</span>
                    </td>
                    <td className="p-3 font-medium text-slate-700">
                      {m.classifier_type}
                    </td>
                    <td className="p-3 text-slate-600">
                      <span className="capitalize">{getFeatureSpaceLabel(m.feature_space)}</span>
                      <span className="text-[11px] text-slate-400 font-mono ml-1.5 tabular-nums">
                        ({m.feature_count.toLocaleString()} feats)
                      </span>
                    </td>
                    <td className="p-3">
                      {m.is_available ? (
                        <span className="inline-flex items-center gap-1.5 text-emerald-700 font-medium">
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                          Ready
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 text-amber-700 font-medium">
                          <XCircle className="w-3.5 h-3.5 text-amber-500" />
                          Not Loaded
                        </span>
                      )}
                    </td>
                    <td className="p-3">
                      {isDefault ? (
                        <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-blue-50 text-[#1E3A8A] border border-blue-200">
                          ★ Default Model
                        </span>
                      ) : m.model_id.includes('mnb') ? (
                        <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200">
                          Top CV Score
                        </span>
                      ) : (
                        <span className="text-slate-400 text-[11px]">—</span>
                      )}
                    </td>
                    <td className="p-3 text-right">
                      <div className="flex items-center justify-end gap-2" onClick={(e) => e.stopPropagation()}>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleOpenDetail(m.model_id)}
                          className="text-xs text-slate-600 hover:text-slate-900"
                        >
                          Inspect
                        </Button>
                        <Link href={`/analyze`}>
                          <Button
                            variant="secondary"
                            size="sm"
                            className="text-xs text-[#1E3A8A]"
                          >
                            Analyze
                          </Button>
                        </Link>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Model Metadata Details Modal */}
      <Modal
        isOpen={selectedModelId !== null}
        onClose={handleCloseDetail}
        title={modelDetail ? modelDetail.display_name : 'Model Inspection'}
        maxWidth="md"
      >
        {detailLoading ? (
          <LoadingState label="Loading serialized model pipeline metadata..." className="py-8" />
        ) : modelDetail ? (
          <div className="space-y-4 text-xs text-slate-700">
            <div>
              <span className="font-mono text-[11px] text-slate-400 block">{modelDetail.model_id}</span>
              <p className="text-slate-600 mt-1 leading-relaxed">
                {modelDetail.preprocessing_details || 'Lightweight scikit-learn classifier pipeline.'}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2.5 p-3 bg-slate-50 rounded-lg border border-slate-200">
              <div>
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Classifier</span>
                <span className="font-semibold text-slate-900">{modelDetail.classifier_type}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Feature Space</span>
                <span className="font-semibold text-slate-900 capitalize">{modelDetail.feature_space}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Feature Count</span>
                <span className="font-mono font-semibold text-slate-900 tabular-nums">
                  {modelDetail.feature_count.toLocaleString()} dimensions
                </span>
              </div>
              <div>
                <span className="text-[10px] text-slate-400 uppercase font-semibold block">Score Metric Type</span>
                <span className="font-mono font-semibold text-slate-900">
                  {modelDetail.score_type === 'probability' ? 'Posterior Probability [0, 1]' : 'Signed Hyperplane Distance'}
                </span>
              </div>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1 font-mono text-[11px] text-slate-600">
              <div><span className="text-slate-400">Scaling:</span> {modelDetail.scaling || 'Unscaled'}</div>
              <div><span className="text-slate-400">Artifact:</span> {modelDetail.artifact_file || 'Serialized joblib artifact'}</div>
              <div><span className="text-slate-400">Status:</span> {modelDetail.is_available ? 'Ready for inference' : 'Artifact missing'}</div>
            </div>

            <div className="pt-2 flex justify-end">
              <Link href="/analyze">
                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleCloseDetail}
                  rightIcon={<ArrowRight className="w-3.5 h-3.5" />}
                  className="bg-[#1E3A8A] hover:bg-[#172554] text-white"
                >
                  Analyze message with this model
                </Button>
              </Link>
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
