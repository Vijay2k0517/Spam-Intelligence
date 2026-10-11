'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { BackendStatus } from '../ui/StatusPill';
import { getReadiness } from '@/lib/api/health';
import { X } from 'lucide-react';

interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const checkConnectivity = useCallback(async () => {
    try {
      const res = await getReadiness();
      if (res.status === 'ready') {
        setBackendStatus('connected');
      } else {
        setBackendStatus('not_ready');
      }
    } catch {
      setBackendStatus('offline');
    }
  }, []);

  useEffect(() => {
    checkConnectivity();
    const interval = setInterval(checkConnectivity, 30000);
    return () => clearInterval(interval);
  }, [checkConnectivity]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await checkConnectivity();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  return (
    <div className="flex h-screen bg-[#F8F9FA] overflow-hidden">
      {/* Desktop Sidebar (Compact 220px) */}
      <div className="hidden lg:flex lg:shrink-0 h-full">
        <Sidebar />
      </div>

      {/* Mobile Drawer Backdrop */}
      {mobileNavOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/30 backdrop-blur-2xs lg:hidden"
          onClick={() => setMobileNavOpen(false)}
        />
      )}

      {/* Mobile Drawer */}
      <div
        className={`fixed inset-y-0 left-0 z-50 w-[230px] bg-white shadow-xl transition-transform duration-200 ease-in-out lg:hidden flex flex-col ${
          mobileNavOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between p-3.5 border-b border-[#EDF2F7]">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Navigation</span>
          <button
            onClick={() => setMobileNavOpen(false)}
            className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100"
            aria-label="Close menu"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          <Sidebar
            onNavigate={() => setMobileNavOpen(false)}
            className="w-full border-r-0 h-full"
          />
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        <Header
          backendStatus={backendStatus}
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
          onOpenMobileNav={() => setMobileNavOpen(true)}
        />

        <main className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
};
