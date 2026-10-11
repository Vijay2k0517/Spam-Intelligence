import type { Metadata } from 'next';
import './globals.css';
import { AppShell } from '@/components/layout/AppShell';
import { PRODUCT_NAME, PRODUCT_DESCRIPTOR } from '@/lib/constants';

export const metadata: Metadata = {
  title: `${PRODUCT_NAME} — ${PRODUCT_DESCRIPTOR}`,
  description:
    'Research-grade SMS spam intelligence and lightweight classification platform comparing textual, non-textual, and hybrid feature representations.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="h-full text-slate-900 bg-[#F7F8FA]">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
