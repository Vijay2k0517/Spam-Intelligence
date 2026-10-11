'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  NAVIGATION_ITEMS,
  PRODUCT_NAME,
} from '@/lib/constants';
import {
  Boxes,
  FlaskConical,
  History,
  Shield,
  ShieldAlert,
} from 'lucide-react';

interface SidebarProps {
  onNavigate?: () => void;
  className?: string;
}

const iconMap: Record<string, React.ElementType> = {
  ShieldAlert,
  FlaskConical,
  Boxes,
  History,
};

export const Sidebar: React.FC<SidebarProps> = ({
  onNavigate,
  className,
}) => {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        'w-[220px] bg-white border-r border-[#E2E8F0] flex flex-col h-full select-none shrink-0',
        className
      )}
    >
      {/* Brand Wordmark */}
      <div className="h-14 px-5 border-b border-[#EDF2F7] flex items-center">
        <Link
          href="/analyze"
          onClick={onNavigate}
          className="flex items-center gap-2.5 group transition-opacity"
        >
          <div className="w-7 h-7 rounded-md bg-[#1E3A8A] flex items-center justify-center text-white shadow-2xs group-hover:bg-[#172554] transition-colors">
            <Shield className="w-4 h-4" />
          </div>
          <div>
            <span className="font-semibold text-xs tracking-tight text-slate-900 block leading-tight">
              {PRODUCT_NAME}
            </span>
            <span className="text-[10px] text-slate-400 font-medium tracking-normal block">
              Spam Research Lab
            </span>
          </div>
        </Link>
      </div>

      {/* Navigation Links (4 Destinations Only) */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
          Workspace
        </div>
        {NAVIGATION_ITEMS.map((item) => {
          const Icon = iconMap[item.icon] || Shield;
          const isActive =
            pathname === item.href || (item.href !== '/analyze' && pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onNavigate}
              className={cn(
                'flex items-center gap-2.5 px-3 py-2 rounded-md text-[13px] font-medium transition-all group relative',
                isActive
                  ? 'bg-slate-100 text-slate-900 font-semibold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              )}
            >
              {/* Subtle active state bar */}
              {isActive && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-1 bg-[#1E3A8A] rounded-r-full" />
              )}
              <Icon
                className={cn(
                  'w-4 h-4 shrink-0 transition-colors',
                  isActive ? 'text-[#1E3A8A]' : 'text-slate-400 group-hover:text-slate-600'
                )}
              />
              <span className="truncate">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Subtle Sidebar Footer */}
      <div className="p-4 border-t border-[#EDF2F7] bg-[#F8F9FA] text-[11px] text-slate-400 flex items-center justify-between tabular-nums">
        <span>Engine v1.0.0</span>
        <span className="font-mono text-[10px]">FastAPI · SQLite</span>
      </div>
    </aside>
  );
};
