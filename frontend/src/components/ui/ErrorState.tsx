import React from 'react';
import { cn } from '@/lib/utils';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from './Button';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Service Error',
  message,
  onRetry,
  className,
}) => {
  return (
    <div
      className={cn(
        'p-5 rounded-lg border border-[#FECACA] bg-[#FEF2F2] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4',
        className
      )}
      role="alert"
    >
      <div className="flex items-start gap-3">
        <div className="p-1 rounded-full bg-[#FEE2E2] text-[#C83D4D] shrink-0 mt-0.5">
          <AlertCircle className="w-5 h-5" />
        </div>
        <div>
          <h4 className="text-sm font-semibold text-[#991B1B]">{title}</h4>
          <p className="text-xs text-[#B91C1C] mt-0.5 max-w-xl leading-relaxed">{message}</p>
        </div>
      </div>
      {onRetry && (
        <Button
          variant="secondary"
          size="sm"
          onClick={onRetry}
          leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
          className="shrink-0 border-[#FECACA] bg-white text-[#991B1B] hover:bg-[#FEE2E2]"
        >
          Retry Request
        </Button>
      )}
    </div>
  );
};
