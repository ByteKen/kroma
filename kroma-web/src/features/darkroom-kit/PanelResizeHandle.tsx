import { useCallback, useRef, useState } from 'react';

export function clampPanelWidth(value: number, minWidth: number, maxWidth: number): number {
  const safeMin = Math.min(minWidth, maxWidth);
  const safeMax = Math.max(minWidth, maxWidth);
  if (!Number.isFinite(value)) return safeMin;
  return Math.round(Math.min(safeMax, Math.max(safeMin, value)));
}

interface UseResizableWidthOptions {
  readonly storageKey: string;
  readonly defaultWidth: number;
  readonly minWidth: number;
  readonly maxWidth: number;
}

export function useResizableWidth({
  storageKey,
  defaultWidth,
  minWidth,
  maxWidth,
}: UseResizableWidthOptions) {
  const fullKey = `kroma.panelWidth.${storageKey}`;

  const [width, setWidthState] = useState<number>(() => {
    if (typeof window === 'undefined' || !window.localStorage) {
      return clampPanelWidth(defaultWidth, minWidth, maxWidth);
    }
    try {
      const saved = window.localStorage.getItem(fullKey);
      if (saved !== null) {
        const parsed = Number(saved);
        if (Number.isFinite(parsed)) {
          return clampPanelWidth(parsed, minWidth, maxWidth);
        }
      }
    } catch {
      // Ignore storage access failures cleanly
    }
    return clampPanelWidth(defaultWidth, minWidth, maxWidth);
  });

  const setWidth = useCallback(
    (next: number) => {
      const clamped = clampPanelWidth(next, minWidth, maxWidth);
      setWidthState(clamped);
      if (typeof window !== 'undefined' && window.localStorage) {
        try {
          window.localStorage.setItem(fullKey, String(clamped));
        } catch {
          // Ignore storage quota restrictions
        }
      }
    },
    [fullKey, minWidth, maxWidth],
  );

  const resetWidth = useCallback(() => {
    setWidth(defaultWidth);
  }, [defaultWidth, setWidth]);

  return { width, setWidth, resetWidth } as const;
}

interface PanelResizeHandleProps {
  /** Which side of the viewport the resized panel is docked on. */
  readonly panelSide: 'left' | 'right';
  readonly width: number;
  readonly minWidth: number;
  readonly maxWidth: number;
  readonly onChangeWidth: (nextWidth: number) => void;
  readonly onResetWidth: () => void;
  readonly label: string;
}

/**
 * Precision 1px darkroom sash divider with a generous 7px hit area, pointer capture,
 * warm amber drag highlight, keyboard nudging, and double-click reset.
 */
export function PanelResizeHandle({
  panelSide,
  width,
  minWidth,
  maxWidth,
  onChangeWidth,
  onResetWidth,
  label,
}: PanelResizeHandleProps) {
  const [dragging, setDragging] = useState(false);
  const dragOriginRef = useRef<{ readonly startX: number; readonly startWidth: number } | null>(
    null,
  );

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={width}
      aria-valuemin={minWidth}
      aria-valuemax={maxWidth}
      tabIndex={0}
      title="Drag to resize • Double-click to reset"
      onDoubleClick={onResetWidth}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        dragOriginRef.current = { startX: event.clientX, startWidth: width };
        setDragging(true);
      }}
      onPointerMove={(event) => {
        const origin = dragOriginRef.current;
        if (!origin) return;
        const deltaX = event.clientX - origin.startX;
        const nextWidth =
          panelSide === 'left' ? origin.startWidth + deltaX : origin.startWidth - deltaX;
        onChangeWidth(nextWidth);
      }}
      onPointerUp={(event) => {
        if (dragOriginRef.current) {
          dragOriginRef.current = null;
          setDragging(false);
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }
      }}
      onPointerCancel={() => {
        dragOriginRef.current = null;
        setDragging(false);
      }}
      onKeyDown={(event) => {
        const step = event.shiftKey ? 32 : 16;
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          onChangeWidth(panelSide === 'left' ? width + step : width - step);
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          onChangeWidth(panelSide === 'left' ? width - step : width + step);
        } else if (event.key === 'Home') {
          event.preventDefault();
          onResetWidth();
        }
      }}
      className="relative w-0 shrink-0 select-none z-30 group focus:outline-none"
    >
      {/* 1px visual seam */}
      <div
        className={`pointer-events-none absolute inset-y-0 left-0 w-px transition-colors ${
          dragging ? 'bg-amber-core w-[2px] -ml-px' : 'bg-hairline group-hover:bg-amber-core/75'
        }`}
      />
      {/* Generous invisible hit strip for effortless grabbing */}
      <div className="absolute inset-y-0 -left-[3px] w-[7px] cursor-col-resize" />
    </div>
  );
}
