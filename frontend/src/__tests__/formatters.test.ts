import { describe, it, expect } from 'vitest';
import {
  formatLatency,
  formatPercent,
  formatMetricNumber,
  formatScore,
  getFeatureSpaceLabel,
} from '../lib/formatters';

describe('Formatters & Score Integrity Unit Tests', () => {
  it('formats latency properly', () => {
    expect(formatLatency(1.452)).toBe('1.45 ms');
    expect(formatLatency(0.85)).toBe('0.85 ms');
    expect(formatLatency(null)).toBe('—');
    expect(formatLatency(undefined)).toBe('—');
  });

  it('formats percentages properly', () => {
    expect(formatPercent(0.9881)).toBe('98.81%');
    expect(formatPercent(0.9419, 1)).toBe('94.2%');
    expect(formatPercent(null)).toBe('—');
  });

  it('formats metrics with specified decimals', () => {
    expect(formatMetricNumber(0.123456, 4)).toBe('0.1235');
    expect(formatMetricNumber(null)).toBe('—');
  });

  describe('formatScore integrity guarantee', () => {
    it('formats Naive Bayes spam probability as percentage and decimal with P(spam) interpretation', () => {
      const probScore = formatScore(0.9998, 'probability');
      expect(probScore).not.toBeNull();
      expect(probScore?.label).toBe('Estimated Spam Probability');
      expect(probScore?.formattedValue).toBe('99.98% (0.9998)');
      expect(probScore?.isProbability).toBe(true);
      expect(probScore?.interpretation).toContain('high likelihood of spam');
      expect(probScore?.interpretation).toContain('P(spam) ≥ 0.50');
      expect(probScore?.interpretation).not.toContain('Calibrated');
    });

    it('formats Ham outcome probability specifically as spam probability with P(spam) < 0.50', () => {
      const probScore = formatScore(0.0002, 'probability');
      expect(probScore).not.toBeNull();
      expect(probScore?.label).toBe('Estimated Spam Probability');
      expect(probScore?.formattedValue).toBe('0.02% (0.0002)');
      expect(probScore?.isProbability).toBe(true);
      expect(probScore?.interpretation).toContain('low likelihood of spam');
      expect(probScore?.interpretation).toContain('P(spam) < 0.50');
      expect(probScore?.interpretation).not.toContain('Calibrated');
    });

    it('never labels or formats SVM decision scores as percentages', () => {
      const svmScore = formatScore(1.2845, 'decision_score');
      expect(svmScore).not.toBeNull();
      expect(svmScore?.label).toBe('Decision Score');
      expect(svmScore?.formattedValue).toBe('+1.2845');
      // Must NOT contain '%' symbol
      expect(svmScore?.formattedValue).not.toContain('%');
      expect(svmScore?.isProbability).toBe(false);
      expect(svmScore?.interpretation).toContain('separating hyperplane');
      expect(svmScore?.interpretation).toContain('decision threshold: 0.0');
    });

    it('formats negative SVM decision margin properly for Ham outcome', () => {
      const svmNegative = formatScore(-0.9521, 'decision_score');
      expect(svmNegative).not.toBeNull();
      expect(svmNegative?.label).toBe('Decision Score');
      expect(svmNegative?.formattedValue).toBe('-0.9521');
      expect(svmNegative?.formattedValue).not.toContain('%');
      expect(svmNegative?.isProbability).toBe(false);
      expect(svmNegative?.interpretation).toContain('decision threshold: 0.0');
    });

    it('formats zero boundary SVM decision score properly', () => {
      const svmZero = formatScore(0.0, 'decision_score');
      expect(svmZero).not.toBeNull();
      expect(svmZero?.formattedValue).toBe('0.0000');
      expect(svmZero?.isProbability).toBe(false);
    });

    it('handles fallback raw score without declared scoreType', () => {
      const fallbackScore = formatScore(0.85);
      expect(fallbackScore).not.toBeNull();
      expect(fallbackScore?.label).toBe('Model Output Score');
      expect(fallbackScore?.formattedValue).toBe('0.8500');
      expect(fallbackScore?.isProbability).toBe(false);
    });

    it('returns null when score is missing or invalid', () => {
      expect(formatScore(null, null)).toBeNull();
      expect(formatScore(undefined, undefined)).toBeNull();
      expect(formatScore(NaN, 'probability')).toBeNull();
      expect(formatScore(NaN, 'decision_score')).toBeNull();
    });
  });

  it('returns clean feature space labels', () => {
    expect(getFeatureSpaceLabel('textual')).toBe('Textual (BoW)');
    expect(getFeatureSpaceLabel('metadata')).toBe('Non-Textual (Metadata)');
    expect(getFeatureSpaceLabel('hybrid')).toBe('Hybrid (Concatenated)');
  });
});
