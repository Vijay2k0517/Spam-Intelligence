import React from 'react';
import { cn } from '@/lib/utils';

export type BackendStatus = 'connected' | 'checking' | 'offline' | 'not_ready';

interface StatusPillProps {
  status: BackendStatus;
  label?: string;
  className?: string;
}

export const StatusPill: React.FC<StatusPillProps> = ({ status, label, className }) => {
  const configs: Record<
    BackendStatus,
    { dot: string; text: string; bg: string; border: string; defaultLabel: string }
  > = {
    connected: {
      dot: 'bg-[#18794E]',
      text: 'text-[#18794E]',
      bg: 'bg-[#F0FDF4]',
      border: 'border-[#BBF7D0]',
      defaultLabel: 'Live API',
    },
    checking: {
      dot: 'bg-amber-500 animate-pulse',
      text: 'text-amber-700',
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      defaultLabel: 'Checking...',
    },
    offline: {
      dot: 'bg-[#C83D4D]',
      text: 'text-[#C83D4D]',
      bg: 'bg-[#FEF2F2]',
      border: 'border-[#FECACA]',
      defaultLabel: 'Offline',
    },
    not_ready: {
      dot: 'bg-amber-500',
      text: 'text-amber-800',
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      defaultLabel: 'Not Ready',
    },
  };

  const config = configs[status] || configs.checking;

  return (
    <div
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border shadow-2xs transition-colors',
        config.bg,
        config.border,
        config.text,
        className
      )}
      role="status"
      aria-label={`API Status: ${label || config.defaultLabel}`}
    >
      <span className={cn('w-2 h-2 rounded-full shrink-0', config.dot)} />
      <span className="tabular-nums">{label || config.defaultLabel}</span>
    </div>
  );
};
