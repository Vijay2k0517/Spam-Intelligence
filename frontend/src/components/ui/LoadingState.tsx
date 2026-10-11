import React from 'react';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  label?: string;
  className?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  label = 'Loading data...',
  className,
}) => {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center p-8 text-center text-slate-500 gap-3 min-h-[160px]',
        className
      )}
      role="status"
    >
      <Loader2 className="w-6 h-6 animate-spin text-[#234A84]" />
      <span className="text-xs font-medium text-slate-600 tracking-wide">{label}</span>
    </div>
  );
};
