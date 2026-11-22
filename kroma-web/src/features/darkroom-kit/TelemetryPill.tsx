import type { ReactNode } from 'react';

export type TelemetryTone = 'neutral' | 'emerald' | 'amber' | 'tangerine';

interface TelemetryPillProps {
  readonly tone?: TelemetryTone;
  readonly pulseDot?: boolean;
  readonly dot?: boolean;
  readonly className?: string;
  readonly children: ReactNode;
}

const TONE_STYLES: Record<TelemetryTone, { wrapper: string; dot: string }> = {
  neutral: {
    wrapper: 'bg-canvas text-ink-secondary border-hairline',
    dot: 'bg-ink-telemetry',
  },
  emerald: {
    wrapper: 'bg-emerald-focus/12 text-emerald-bright border-emerald-focus/35',
    dot: 'bg-emerald-focus',
  },
  amber: {
    wrapper: 'bg-amber-core/12 text-amber-soft border-amber-core/35',
    dot: 'bg-amber-core',
  },
  tangerine: {
    wrapper: 'bg-tangerine-alert/15 text-tangerine-alert border-tangerine-alert/45',
    dot: 'bg-tangerine-alert',
  },
};

export function TelemetryPill({
  tone = 'neutral',
  pulseDot = false,
  dot = false,
  className = '',
  children,
}: TelemetryPillProps) {
  const styles = TONE_STYLES[tone];

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border tabular-nums-mono text-[10px] font-medium ${styles.wrapper} ${className}`}
    >
      {(dot || pulseDot) && (
        <span
          className={`w-1.5 h-1.5 rounded-full shrink-0 ${styles.dot} ${
            pulseDot ? 'animate-pulse' : ''
          }`}
        />
      )}
      {children}
    </span>
  );
}
