'use client';

import React, { useEffect, useState, useMemo } from 'react';
import { Button } from '@/components/ui/Button';
import { LoadingState } from '@/components/ui/LoadingState';
import { ErrorState } from '@/components/ui/ErrorState';
import { Modal } from '@/components/ui/Modal';
import { getExperiments, getExperimentDetail } from '@/lib/api/experiments';
import { formatPercent, getFeatureSpaceLabel } from '@/lib/formatters';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Info } from 'lucide-react';
import { ExperimentDetail, ExperimentSummary } from '@/types/experiments';

type MetricType = 'accuracy' | 'precision' | 'recall' | 'f1';
type ProtocolType = 'cv' | 'holdout';

export default function ExperimentsPage() {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [disclosure, setDisclosure] = useState<string>('');

  // Primary controls
  const [protocol, setProtocol] = useState<ProtocolType>('cv');
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('accuracy');

  // Filters
  const [featureSpaceFilter, setFeatureSpaceFilter] = useState<string>('all');
  const [classifierFilter, setClassifierFilter] = useState<string>('all');

  // Detail Modal
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);

  useEffect(() => {
    async function loadExperimentsList() {
      try {
        setLoading(true);
        setError(null);
        const res = await getExperiments();
        setExperiments(res.experiments);
        setDisclosure(res.selection_bias_disclosure);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to retrieve experiment reports.');
      } finally {
        setLoading(false);
      }
    }
    loadExperimentsList();
  }, []);

  const handleOpenDetail = async (expId: string) => {
    setSelectedExpId(expId);
    setDetailLoading(true);
    try {
      const res = await getExperimentDetail(expId);
      setDetail(res);
    } catch (err) {
      console.error('Failed to load experiment detail:', err);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleCloseDetail = () => {
    setSelectedExpId(null);
    setDetail(null);
  };

  // Filtered experiments
  const filteredExperiments = useMemo(() => {
    return experiments
      .filter((exp) => {
        if (featureSpaceFilter !== 'all' && exp.feature_space !== featureSpaceFilter) return false;
        if (classifierFilter !== 'all') {
          if (classifierFilter === 'SVC' && !exp.classifier.includes('SVC')) return false;
          if (classifierFilter === 'MultinomialNB' && !exp.classifier.includes('MultinomialNB')) return false;
          if (classifierFilter === 'BernoulliNB' && !exp.classifier.includes('BernoulliNB')) return false;
        }
        return true;
      })
      .sort((a, b) => {
        const getVal = (item: ExperimentSummary) => {
          if (selectedMetric === 'accuracy') return item.test_accuracy || 0;
          if (selectedMetric === 'precision') return item.spam_precision || 0;
          if (selectedMetric === 'recall') return item.spam_recall || 0;
          return item.spam_f1 || 0;
        };
        return getVal(b) - getVal(a);
      });
  }, [experiments, featureSpaceFilter, classifierFilter, selectedMetric]);

  // Chart data formatted for horizontal bar chart
  const chartData = useMemo(() => {
    return filteredExperiments.slice(0, 10).map((exp) => {
      let metricVal = exp.test_accuracy || 0;
      if (selectedMetric === 'precision') metricVal = exp.spam_precision || 0;
      if (selectedMetric === 'recall') metricVal = exp.spam_recall || 0;
      if (selectedMetric === 'f1') metricVal = exp.spam_f1 || 0;

      return {
        id: exp.experiment_id,
        name: exp.model_name.replace('Hybrid ', 'Hyb. ').replace('Textual ', 'Txt. ').replace('Metadata ', 'Meta. '),
        fullName: exp.model_name,
        value: Number((metricVal * 100).toFixed(2)),
        rawValue: metricVal,
        space: exp.feature_space,
      };
    });
  }, [filteredExperiments, selectedMetric]);

  const metricLabelMap: Record<MetricType, string> = {
    accuracy: 'Accuracy',
    precision: 'Spam Precision',
    recall: 'Spam Recall',
    f1: 'Spam F1-Score',
  };

  if (loading) {
    return <LoadingState label="Loading research benchmark metrics..." className="py-24" />;
  }

  if (error) {
    return <ErrorState message={error} />;
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="pb-4 border-b border-slate-200">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
          Experiment Lab
        </h1>
        <p className="text-[15px] text-slate-500 mt-1">
          Systematic comparison of 18 model architectures across textual, non-textual, and hybrid representations.
        </p>
      </div>

      {/* Protocol Disclosure Banner */}
      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1.5">
        <div className="flex items-center gap-2 font-semibold text-slate-800">
          <Info className="w-4 h-4 text-[#1E3A8A]" />
          <span>Evaluation Protocol & Selection Bias Disclosure</span>
        </div>
        <p className="leading-relaxed text-slate-600">
          {disclosure ||
            'Candidate selection was determined strictly via 5-Fold Stratified Cross-Validation on the training partition to prevent data leakage. The 80:20 held-out test split results represent an exploratory benchmark.'}
        </p>
      </div>

      {/* Protocol & Metric Selector Row */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs p-5 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Protocol Selector */}
          <div className="space-y-1">
            <span className="text-xs font-semibold text-slate-700 block">Evaluation Protocol</span>
            <div className="inline-flex p-1 bg-slate-100 rounded-lg border border-slate-200/60">
              <button
                type="button"
                onClick={() => setProtocol('cv')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  protocol === 'cv'
                    ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                5-Fold Stratified CV (Model Selection)
              </button>
              <button
                type="button"
                onClick={() => setProtocol('holdout')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  protocol === 'holdout'
                    ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                80:20 Exploratory Holdout
              </button>
            </div>
          </div>

          {/* Metric Selector */}
          <div className="space-y-1">
            <span className="text-xs font-semibold text-slate-700 block">Comparison Metric</span>
            <div className="inline-flex p-1 bg-slate-100 rounded-lg border border-slate-200/60">
              {(['accuracy', 'precision', 'recall', 'f1'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setSelectedMetric(m)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    selectedMetric === m
                      ? 'bg-white text-[#1E3A8A] font-semibold shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {metricLabelMap[m]}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Filter Pills */}
        <div className="flex flex-wrap items-center gap-4 pt-3 border-t border-slate-100 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-medium">Feature Space:</span>
            {(['all', 'textual', 'metadata', 'hybrid'] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setFeatureSpaceFilter(s)}
                className={`px-2.5 py-1 rounded-md text-xs transition-colors ${
                  featureSpaceFilter === s
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
              { id: 'MultinomialNB', label: 'MultinomialNB' },
              { id: 'BernoulliNB', label: 'BernoulliNB' },
              { id: 'SVC', label: 'Linear SVM' },
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

      {/* ONE Clear Model-Comparison Chart (Horizontal Bars) */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              {metricLabelMap[selectedMetric]} Benchmark
            </h2>
            <p className="text-xs text-slate-500">
              Comparing top configurations for {metricLabelMap[selectedMetric]} under {protocol === 'cv' ? '5-Fold CV' : 'Exploratory 80:20 Split'}.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Showing top {chartData.length} models
          </span>
        </div>

        <div className="h-[340px] w-full pt-2">
          <ResponsiveContainer width="100%" height="100%" minWidth={300} minHeight={340}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#F1F5F9" />
              <XAxis
                type="number"
                domain={[85, 100]}
                unit="%"
                tick={{ fontSize: 11, fill: '#64748B' }}
                axisLine={{ stroke: '#E2E8F0' }}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={150}
                tick={{ fontSize: 11, fill: '#334155' }}
                axisLine={{ stroke: '#E2E8F0' }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-slate-900 text-white p-2.5 rounded-lg text-xs shadow-lg space-y-1">
                        <p className="font-semibold">{data.fullName}</p>
                        <p className="text-slate-300">Space: {data.space}</p>
                        <p className="text-emerald-400 font-mono font-bold">
                          {metricLabelMap[selectedMetric]}: {data.value}%
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar
                dataKey="value"
                fill="#1E3A8A"
                radius={[0, 4, 4, 0]}
                barSize={14}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Compact Detailed Experiment Table */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-2xs overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <h2 className="text-base font-semibold text-slate-900">
            Experiment Matrix ({filteredExperiments.length} configurations)
          </h2>
          <span className="text-xs text-slate-400">Click row or Details to inspect confusion matrix</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
              <tr>
                <th className="p-3">ID</th>
                <th className="p-3">Model Architecture</th>
                <th className="p-3">Feature Space</th>
                <th className="p-3 text-right">Accuracy</th>
                <th className="p-3 text-right">Spam Precision</th>
                <th className="p-3 text-right">Spam Recall</th>
                <th className="p-3 text-right">Spam F1</th>
                <th className="p-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredExperiments.map((exp) => (
                <tr
                  key={exp.experiment_id}
                  onClick={() => handleOpenDetail(exp.experiment_id)}
                  className="hover:bg-slate-50/70 cursor-pointer transition-colors"
                >
                  <td className="p-3 font-mono font-semibold text-slate-900">{exp.experiment_id}</td>
                  <td className="p-3 font-medium text-slate-800">
                    {exp.model_name}
                    {exp.experiment_id === 'E5_MNB' && (
                      <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-blue-50 text-blue-800 font-semibold border border-blue-200">
                        Selected Candidate
                      </span>
                    )}
                  </td>
                  <td className="p-3 text-slate-600 capitalize">
                    {getFeatureSpaceLabel(exp.feature_space)}
                  </td>
                  <td className="p-3 text-right font-mono font-medium text-slate-900 tabular-nums">
                    {formatPercent(exp.test_accuracy)}
                  </td>
                  <td className="p-3 text-right font-mono text-slate-600 tabular-nums">
                    {formatPercent(exp.spam_precision)}
                  </td>
                  <td className="p-3 text-right font-mono text-slate-600 tabular-nums">
                    {formatPercent(exp.spam_recall)}
                  </td>
                  <td className="p-3 text-right font-mono font-medium text-slate-900 tabular-nums">
                    {formatPercent(exp.spam_f1)}
                  </td>
                  <td className="p-3 text-center">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOpenDetail(exp.experiment_id);
                      }}
                      className="text-xs text-[#1E3A8A] hover:bg-slate-100"
                    >
                      Details
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail Modal: Progressive Disclosure for Confusion Matrix & Folds */}
      <Modal
        isOpen={selectedExpId !== null}
        onClose={handleCloseDetail}
        title={`Experiment Report — ${selectedExpId || ''}`}
        maxWidth="lg"
      >
        {detailLoading ? (
          <LoadingState label="Loading detailed confusion matrix and parameters..." className="py-12" />
        ) : detail ? (
          <div className="space-y-5 text-xs text-slate-700">
            <div>
              <h3 className="font-semibold text-sm text-slate-900">{detail.model_name}</h3>
              <p className="text-slate-500 mt-0.5">{detail.model_parameters || 'SMS classification pipeline experiment.'}</p>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-semibold">Test Accuracy</span>
                <span className="font-mono text-base font-bold text-slate-900 tabular-nums">{formatPercent(detail.test_accuracy)}</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-semibold">Spam Precision</span>
                <span className="font-mono text-base font-bold text-slate-900 tabular-nums">{formatPercent(detail.spam_precision)}</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-semibold">Spam Recall</span>
                <span className="font-mono text-base font-bold text-slate-900 tabular-nums">{formatPercent(detail.spam_recall)}</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-semibold">Spam F1-Score</span>
                <span className="font-mono text-base font-bold text-slate-900 tabular-nums">{formatPercent(detail.spam_f1)}</span>
              </div>
            </div>

            {/* Confusion Matrix */}
            {detail.confusion_matrix && (
              <div className="space-y-2">
                <span className="font-semibold text-slate-900 block">Confusion Matrix</span>
                <div className="grid grid-cols-2 gap-2 text-center p-3 bg-slate-50 rounded-lg border border-slate-200 font-mono text-xs">
                  <div className="p-2.5 bg-white rounded border border-slate-100">
                    <span className="text-[10px] text-slate-400 block">True Legitimate (TN)</span>
                    <span className="text-sm font-bold text-emerald-700 tabular-nums">{detail.confusion_matrix.tn.toLocaleString()}</span>
                  </div>
                  <div className="p-2.5 bg-white rounded border border-slate-100">
                    <span className="text-[10px] text-slate-400 block">False Spam (FP)</span>
                    <span className="text-sm font-bold text-amber-700 tabular-nums">{detail.confusion_matrix.fp.toLocaleString()}</span>
                  </div>
                  <div className="p-2.5 bg-white rounded border border-slate-100">
                    <span className="text-[10px] text-slate-400 block">False Legitimate (FN)</span>
                    <span className="text-sm font-bold text-red-700 tabular-nums">{detail.confusion_matrix.fn.toLocaleString()}</span>
                  </div>
                  <div className="p-2.5 bg-white rounded border border-slate-100">
                    <span className="text-[10px] text-slate-400 block">True Spam (TP)</span>
                    <span className="text-sm font-bold text-emerald-700 tabular-nums">{detail.confusion_matrix.tp.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Pipeline Configuration Details */}
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1 text-slate-600 font-mono text-[11px]">
              <div><span className="text-slate-400">Classifier:</span> {detail.classifier}</div>
              <div><span className="text-slate-400">Feature Space:</span> {detail.feature_space}</div>
              <div><span className="text-slate-400">Feature Scaling:</span> {detail.scaling || 'None'}</div>
              <div><span className="text-slate-400">Protocol:</span> {detail.evaluation_protocol || 'Cross-Validation'}</div>
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
