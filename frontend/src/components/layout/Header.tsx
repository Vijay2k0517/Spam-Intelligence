'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import { NAVIGATION_ITEMS } from '@/lib/constants';
import { Menu, RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';
import { StatusPill, BackendStatus } from '../ui/StatusPill';

interface HeaderProps {
  backendStatus: BackendStatus;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  onOpenMobileNav: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  backendStatus,
  onRefresh,
  isRefreshing,
  onOpenMobileNav,
}) => {
  const pathname = usePathname();

  const currentNav =
    NAVIGATION_ITEMS.find((item) =>
      item.href === '/analyze' ? pathname === '/' || pathname.startsWith('/analyze') : pathname.startsWith(item.href)
    ) || NAVIGATION_ITEMS[0];

  return (
    <header className="h-14 bg-white border-b border-[#E2E8F0] flex items-center justify-between px-4 sm:px-6 sticky top-0 z-30 select-none">
      {/* Left: Mobile Toggle & Page Context */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobileNav}
          className="lg:hidden p-1.5 rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-900">
            {currentNav.label}
          </span>
          <span className="text-xs text-slate-400 font-normal hidden sm:inline">
            · {currentNav.description}
          </span>
        </div>
      </div>

      {/* Right: Exactly ONE compact backend status indicator & optional refresh */}
      <div className="flex items-center gap-2.5">
        <StatusPill status={backendStatus} />

        {onRefresh && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRefresh}
            isLoading={isRefreshing}
            aria-label="Refresh page data"
            className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 h-8 w-8"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </Button>
        )}
      </div>
    </header>
  );
};
