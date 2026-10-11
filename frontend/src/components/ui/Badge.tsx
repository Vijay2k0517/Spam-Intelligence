import React from 'react';
import { cn } from '@/lib/utils';

export type BadgeVariant =
  | 'default'
  | 'primary'
  | 'spam'
  | 'ham'
  | 'warning'
  | 'success'
  | 'outline'
  | 'neutral';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: 'sm' | 'md';
  children: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  size = 'md',
  className,
  children,
  ...props
}) => {
  const baseClasses =
    'inline-flex items-center font-medium rounded-md whitespace-nowrap transition-colors';

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs',
  };

  const variantClasses: Record<BadgeVariant, string> = {
    default: 'bg-slate-100 text-slate-800 border border-slate-200',
    primary: 'bg-[#F0F5FA] text-[#234A84] border border-[#D9E4F0]',
    spam: 'bg-[#FEF2F2] text-[#C83D4D] border border-[#FECACA]',
    ham: 'bg-[#F0FDF4] text-[#18794E] border border-[#BBF7D0]',
    warning: 'bg-[#FEFCE8] text-[#A16207] border border-[#FEF08A]',
    success: 'bg-[#F0FDF4] text-[#18794E] border border-[#BBF7D0]',
    outline: 'bg-transparent text-slate-700 border border-slate-300',
    neutral: 'bg-slate-50 text-slate-600 border border-slate-200',
  };

  return (
    <span
      className={cn(baseClasses, sizeClasses[size], variantClasses[variant], className)}
      {...props}
    >
      {children}
    </span>
  );
};
