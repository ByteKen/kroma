import { useEffect, useRef, useState } from 'react';
import { DarkroomTooltip } from './DarkroomTooltip';

interface DarkroomScrubberProps {
  readonly label: string;
  readonly value: number;
  readonly min: number;
  readonly max: number;
  readonly step?: number;
  readonly defaultValue?: number;
  readonly formatValue?: (value: number) => string;
  readonly onChange: (next: number) => void;
  readonly layout?: 'inline' | 'stacked';
  readonly trackGradient?: string;
  readonly className?: string;
}

function decimalsForStep(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 0;
  const str = String(step);
  const dot = str.indexOf('.');
  return dot === -1 ? 0 : Math.min(4, str.length - dot - 1);
}

/**
 * Parses a user-typed slider string (tolerating optional `+`, `-`, decimals, and trailing unit
 * suffixes such as `4800K`, `+0.35 EV`, `-12%`, `2.5°`), clamps to `[min, max]`, and rounds to
 * step decimal precision. Returns `null` when the input is not a valid finite number.
 */
export function parseSliderInputValue(
  raw: string,
  min: number,
  max: number,
  step = 1,
): number | null {
  const cleaned = raw.trim().replace(/[a-zA-Z%°\s]+$/g, '');
  if (cleaned.length === 0) return null;
  const parsed = Number(cleaned);
  if (!Number.isFinite(parsed)) return null;
  const clamped = Math.min(max, Math.max(min, parsed));
  const decimals = decimalsForStep(step);
  return decimals > 0 ? Number(clamped.toFixed(decimals)) : Math.round(clamped);
}

function formatEditableSeed(value: number, step: number): string {
  const decimals = decimalsForStep(step);
  return decimals > 0 ? value.toFixed(decimals) : String(Math.round(value));
}

/**
 * Precision darkroom slider used across Culling and Lightroom-style Studio Develop panels.
 * - Click the numeric value box to type an exact value (`Enter`/blur commits, `Esc` cancels,
 *   `↑`/`↓` nudges by `step`, `Shift+↑`/`↓` nudges by `10× step`).
 * - Double-click the label or slider track to reset cleanly to `defaultValue`.
 */
export function DarkroomScrubber({
  label,
  value,
  min,
  max,
  step = 1,
  defaultValue,
  formatValue = (v) => String(v),
  onChange,
  layout = 'stacked',
  trackGradient,
  className = '',
}: DarkroomScrubberProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const canReset = defaultValue !== undefined && Math.abs(value - defaultValue) > 1e-6;

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const handleReset = () => {
    if (defaultValue !== undefined) {
      setEditing(false);
      onChange(defaultValue);
    }
  };

  const beginEdit = () => {
    setDraft(formatEditableSeed(value, step));
    setEditing(true);
  };

  const commitDraft = () => {
    const parsed = parseSliderInputValue(draft, min, max, step);
    setEditing(false);
    if (parsed !== null && Math.abs(parsed - value) > 1e-6) {
      onChange(parsed);
    }
  };

  const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    event.stopPropagation();
    if (event.key === 'Enter') {
      event.preventDefault();
      commitDraft();
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      setEditing(false);
      return;
    }
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
      const base = parseSliderInputValue(draft, min, max, step) ?? value;
      const mult = event.shiftKey ? 10 : 1;
      const dir = event.key === 'ArrowUp' ? 1 : -1;
      const decimals = decimalsForStep(step);
      const rawNext = Math.min(max, Math.max(min, base + dir * step * mult));
      const next = decimals > 0 ? Number(rawNext.toFixed(decimals)) : Math.round(rawNext);
      setDraft(formatEditableSeed(next, step));
      onChange(next);
    }
  };

  const valueBoxClass =
    layout === 'inline'
      ? 'tabular-nums-mono text-[11px] text-amber-soft font-semibold min-w-[40px] px-1 py-0.5 rounded bg-surface-low/60 border border-transparent hover:border-hairline-strong text-right cursor-text transition-colors'
      : `tabular-nums-mono text-[11px] px-1.5 py-0.5 rounded bg-canvas border min-w-[52px] text-right cursor-text transition-colors ${
          canReset
            ? 'text-amber-soft border-amber-core/35 hover:border-amber-core/70'
            : 'text-ink-secondary border-hairline hover:border-hairline-strong hover:text-ink-primary'
        }`;

  const valueInputClass =
    layout === 'inline'
      ? 'tabular-nums-mono text-[11px] text-amber-soft font-semibold w-[48px] px-1 py-0.5 rounded bg-canvas border border-amber-core text-right focus:outline-none'
      : 'tabular-nums-mono text-[11px] text-ink-primary w-[66px] px-1.5 py-0.5 rounded bg-surface-lowest border border-amber-core text-right focus:outline-none';

  const valueControl = editing ? (
    <input
      ref={inputRef}
      type="text"
      inputMode="decimal"
      aria-label={`${label} value`}
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commitDraft}
      onKeyDown={handleInputKeyDown}
      className={valueInputClass}
    />
  ) : (
    <DarkroomTooltip content="Click to type exact value">
      <button
        type="button"
        onClick={beginEdit}
        aria-label={`Edit ${label} value (${formatValue(value)})`}
        className={valueBoxClass}
      >
        {formatValue(value)}
      </button>
    </DarkroomTooltip>
  );

  if (layout === 'inline') {
    return (
      <div
        className={`flex items-center gap-2 px-2.5 h-8 rounded-lg bg-canvas border border-hairline ${className}`}
      >
        <DarkroomTooltip content={defaultValue !== undefined ? 'Double-click to reset' : undefined}>
          <button
            type="button"
            onDoubleClick={handleReset}
            className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-ink-primary transition-colors cursor-default"
          >
            {label}
          </button>
        </DarkroomTooltip>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          onDoubleClick={handleReset}
          style={trackGradient ? { background: trackGradient } : undefined}
          className="kroma-slider w-20"
        />
        {valueControl}
      </div>
    );
  }

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <div className="flex items-center justify-between text-[11px]">
        <DarkroomTooltip content={defaultValue !== undefined ? 'Double-click to reset' : undefined}>
          <button
            type="button"
            onDoubleClick={handleReset}
            className="text-ink-secondary hover:text-ink-primary transition-colors text-left cursor-default"
          >
            {label}
          </button>
        </DarkroomTooltip>
        {valueControl}
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        onDoubleClick={handleReset}
        style={trackGradient ? { background: trackGradient } : undefined}
        className="kroma-slider w-full"
      />
    </div>
  );
}
