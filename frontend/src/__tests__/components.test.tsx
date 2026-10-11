import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { StatusPill } from '../components/ui/StatusPill';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { EmptyState } from '../components/ui/EmptyState';
import { ErrorState } from '../components/ui/ErrorState';

describe('UI Primitives Unit Tests', () => {
  it('renders Badge with proper variants', () => {
    const { rerender } = render(<Badge variant="spam">SPAM</Badge>);
    expect(screen.getByText('SPAM')).toBeDefined();

    rerender(<Badge variant="ham">HAM</Badge>);
    expect(screen.getByText('HAM')).toBeDefined();
  });

  it('renders Button and handles click events and loading state', () => {
    const handleClick = vi.fn();
    const { rerender } = render(
      <Button onClick={handleClick}>Analyze</Button>
    );
    const btn = screen.getByRole('button', { name: /analyze/i });
    fireEvent.click(btn);
    expect(handleClick).toHaveBeenCalledTimes(1);

    // When loading, button should be disabled
    rerender(<Button isLoading onClick={handleClick}>Analyze</Button>);
    expect((screen.getByRole('button') as HTMLButtonElement).disabled).toBe(true);
  });

  it('renders StatusPill with live and offline states', () => {
    const { rerender } = render(<StatusPill status="connected" />);
    expect(screen.getByText('Live API')).toBeDefined();

    rerender(<StatusPill status="offline" />);
    expect(screen.getByText('Offline')).toBeDefined();
  });

  it('renders Card hierarchy correctly', () => {
    render(
      <Card>
        <CardHeader>
          <CardTitle>Test Title</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Card body content</p>
        </CardContent>
      </Card>
    );
    expect(screen.getByText('Test Title')).toBeDefined();
    expect(screen.getByText('Card body content')).toBeDefined();
  });

  it('renders EmptyState and ErrorState with action triggers', () => {
    const handleRetry = vi.fn();
    render(
      <ErrorState message="Connection rejected" onRetry={handleRetry} />
    );
    expect(screen.getByText('Connection rejected')).toBeDefined();
    const retryBtn = screen.getByRole('button', { name: /retry/i });
    fireEvent.click(retryBtn);
    expect(handleRetry).toHaveBeenCalledTimes(1);

    render(
      <EmptyState title="No items" description="Nothing found in database" />
    );
    expect(screen.getByText('No items')).toBeDefined();
    expect(screen.getByText('Nothing found in database')).toBeDefined();
  });
});
