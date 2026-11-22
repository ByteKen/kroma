import { AnimatePresence, motion } from 'framer-motion';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

export type TooltipSide = 'top' | 'bottom' | 'left' | 'right' | 'auto';

export interface TooltipTriggerRect {
  readonly left: number;
  readonly right: number;
  readonly top: number;
  readonly bottom: number;
  readonly width: number;
  readonly height: number;
}

export interface ComputedTooltipPlacement {
  readonly side: 'top' | 'bottom' | 'left' | 'right';
  readonly left: number;
  readonly top: number;
  readonly arrowOffsetPx: number;
}

export interface ParsedTooltipContent {
  readonly text: string;
  readonly shortcut: string | null;
}

const TOOLTIP_GAP_PX = 8;
const VIEWPORT_MARGIN_PX = 8;
const ARROW_EDGE_INSET_PX = 12;
const INITIAL_OPEN_DELAY_MS = 180;
const LEAVE_CLOSE_GRACE_MS = 55;
const WARM_GROUP_WINDOW_MS = 280;

const TOOLTIP_SPRING = {
  type: 'spring' as const,
  stiffness: 520,
  damping: 34,
  mass: 0.55,
};

const SHORTCUT_TOKEN_PATTERN =
  /^(?:(?:Ctrl|Cmd|Alt|Shift)\+[^\s)]+|Esc|Space|Enter|Tab|Home|End|←|→|↑|↓|[A-Z0-9])$/;

/**
 * Splits a trailing keyboard shortcut like `"Close (Esc)"` or `"Undo (Ctrl+Z)"`
 * into crisp tooltip text + a compact `<kbd>` pill.
 */
export function parseTooltipContent(raw: string): ParsedTooltipContent {
  const trimmed = raw.trim();
  const match = trimmed.match(/^(.*?)\s*\(([^()]+)\)$/);
  if (!match) {
    return { text: trimmed, shortcut: null };
  }
  const candidate = match[2].trim();
  if (!SHORTCUT_TOKEN_PATTERN.test(candidate)) {
    return { text: trimmed, shortcut: null };
  }
  return {
    text: match[1].trim(),
    shortcut: candidate,
  };
}

/**
 * Computes viewport-safe tooltip placement and exact arrow alignment relative to the trigger.
 */
export function computeTooltipPlacement(
  trigger: TooltipTriggerRect,
  tooltipWidth: number,
  tooltipHeight: number,
  preferredSide: TooltipSide,
  viewportWidth: number,
  viewportHeight: number,
): ComputedTooltipPlacement {
  const fitsTop = trigger.top - tooltipHeight - TOOLTIP_GAP_PX >= VIEWPORT_MARGIN_PX;
  const fitsBottom =
    trigger.bottom + tooltipHeight + TOOLTIP_GAP_PX <= viewportHeight - VIEWPORT_MARGIN_PX;
  const fitsRight =
    trigger.right + tooltipWidth + TOOLTIP_GAP_PX <= viewportWidth - VIEWPORT_MARGIN_PX;
  const fitsLeft = trigger.left - tooltipWidth - TOOLTIP_GAP_PX >= VIEWPORT_MARGIN_PX;

  let side: 'top' | 'bottom' | 'left' | 'right';
  if (preferredSide === 'top' || preferredSide === 'auto') {
    side = fitsTop ? 'top' : 'bottom';
  } else if (preferredSide === 'bottom') {
    side = fitsBottom ? 'bottom' : 'top';
  } else if (preferredSide === 'right') {
    side = fitsRight ? 'right' : 'left';
  } else {
    side = fitsLeft ? 'left' : 'right';
  }

  const triggerCenterX = trigger.left + trigger.width / 2;
  const triggerCenterY = trigger.top + trigger.height / 2;

  if (side === 'top' || side === 'bottom') {
    const rawLeft = triggerCenterX - tooltipWidth / 2;
    const maxLeft = Math.max(VIEWPORT_MARGIN_PX, viewportWidth - tooltipWidth - VIEWPORT_MARGIN_PX);
    const left = Math.round(Math.max(VIEWPORT_MARGIN_PX, Math.min(maxLeft, rawLeft)));
    const top = Math.round(
      side === 'top'
        ? Math.max(VIEWPORT_MARGIN_PX, trigger.top - tooltipHeight - TOOLTIP_GAP_PX)
        : Math.min(viewportHeight - tooltipHeight - VIEWPORT_MARGIN_PX, trigger.bottom + TOOLTIP_GAP_PX),
    );
    const maxArrow = Math.max(ARROW_EDGE_INSET_PX, tooltipWidth - ARROW_EDGE_INSET_PX);
    const arrowOffsetPx = Math.round(
      Math.max(ARROW_EDGE_INSET_PX, Math.min(maxArrow, triggerCenterX - left)),
    );
    return { side, left, top, arrowOffsetPx };
  }

  const rawTop = triggerCenterY - tooltipHeight / 2;
  const maxTop = Math.max(VIEWPORT_MARGIN_PX, viewportHeight - tooltipHeight - VIEWPORT_MARGIN_PX);
  const top = Math.round(Math.max(VIEWPORT_MARGIN_PX, Math.min(maxTop, rawTop)));
  const left = Math.round(
    side === 'right'
      ? Math.min(viewportWidth - tooltipWidth - VIEWPORT_MARGIN_PX, trigger.right + TOOLTIP_GAP_PX)
      : Math.max(VIEWPORT_MARGIN_PX, trigger.left - tooltipWidth - TOOLTIP_GAP_PX),
  );
  const maxArrow = Math.max(ARROW_EDGE_INSET_PX, tooltipHeight - ARROW_EDGE_INSET_PX);
  const arrowOffsetPx = Math.round(
    Math.max(ARROW_EDGE_INSET_PX, Math.min(maxArrow, triggerCenterY - top)),
  );
  return { side, left, top, arrowOffsetPx };
}

interface ActiveTooltipState {
  readonly id: string;
  readonly content: string;
  readonly shortcut: string | null;
  readonly side: TooltipSide;
  readonly rect: TooltipTriggerRect;
}

interface TooltipContextValue {
  readonly showTooltip: (
    id: string,
    content: string,
    rect: TooltipTriggerRect,
    side: TooltipSide,
    shortcut?: string | null,
  ) => void;
  readonly hideTooltip: (id?: string) => void;
}

const DarkroomTooltipContext = createContext<TooltipContextValue | null>(null);

function estimateTooltipWidth(text: string, shortcut: string | null): number {
  const base = text.length * 6.6 + 24;
  const kbd = shortcut ? shortcut.length * 6.5 + 22 : 0;
  return Math.min(260, Math.max(56, Math.round(base + kbd)));
}

function toTriggerRect(el: HTMLElement): TooltipTriggerRect {
  const r = el.getBoundingClientRect();
  return {
    left: r.left,
    right: r.right,
    top: r.top,
    bottom: r.bottom,
    width: r.width,
    height: r.height,
  };
}

/**
 * Shared Motion Base-style Tooltip Provider:
 * - Waits 180ms on initial hover so fast cursor sweeps never flicker.
 * - Switches instantaneously (0ms) and glides smoothly between triggers within a warm group window.
 * - Keeps context actions referentially stable so opening a tooltip causes zero trigger re-renders.
 * - Dismisses immediately on pointer down, scroll, Escape, or window blur.
 */
export function DarkroomTooltipProvider({ children }: { readonly children: ReactNode }) {
  const [active, setActive] = useState<ActiveTooltipState | null>(null);
  const [measured, setMeasured] = useState<{
    readonly key: string;
    readonly width: number;
    readonly height: number;
  } | null>(null);
  const tooltipElRef = useRef<HTMLDivElement | null>(null);
  const openTimerRef = useRef<number | null>(null);
  const closeTimerRef = useRef<number | null>(null);
  const lastClosedAtRef = useRef<number>(0);
  const isOpenRef = useRef(false);

  const clearTimers = useCallback(() => {
    if (openTimerRef.current !== null) {
      window.clearTimeout(openTimerRef.current);
      openTimerRef.current = null;
    }
    if (closeTimerRef.current !== null) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  const hideTooltip = useCallback(
    (id?: string) => {
      if (openTimerRef.current !== null) {
        window.clearTimeout(openTimerRef.current);
        openTimerRef.current = null;
      }

      const commitClose = () => {
        setActive((prev) => {
          if (!prev) return null;
          if (id && prev.id !== id) return prev;
          isOpenRef.current = false;
          lastClosedAtRef.current = performance.now();
          return null;
        });
      };

      // Global dismissals (click, scroll, Escape, blur) close immediately; trigger leave gets a 55ms bridge
      // so moving across small toolbar gaps glides directly to the next tooltip without flickering.
      if (!id) {
        clearTimers();
        commitClose();
        return;
      }

      if (closeTimerRef.current !== null) {
        window.clearTimeout(closeTimerRef.current);
      }
      closeTimerRef.current = window.setTimeout(() => {
        closeTimerRef.current = null;
        commitClose();
      }, LEAVE_CLOSE_GRACE_MS);
    },
    [clearTimers],
  );

  const showTooltip = useCallback(
    (
      id: string,
      content: string,
      rect: TooltipTriggerRect,
      side: TooltipSide,
      explicitShortcut?: string | null,
    ) => {
      const trimmed = content.trim();
      if (!trimmed) {
        hideTooltip(id);
        return;
      }

      clearTimers();
      const parsed = parseTooltipContent(trimmed);
      const nextState: ActiveTooltipState = {
        id,
        content: parsed.text,
        shortcut: explicitShortcut ?? parsed.shortcut,
        side,
        rect,
      };

      const isWarm =
        isOpenRef.current || performance.now() - lastClosedAtRef.current < WARM_GROUP_WINDOW_MS;
      if (isWarm) {
        isOpenRef.current = true;
        setActive(nextState);
        return;
      }

      openTimerRef.current = window.setTimeout(() => {
        openTimerRef.current = null;
        isOpenRef.current = true;
        setActive(nextState);
      }, INITIAL_OPEN_DELAY_MS);
    },
    [clearTimers, hideTooltip],
  );

  useEffect(() => {
    const dismiss = () => hideTooltip();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') hideTooltip();
    };
    window.addEventListener('pointerdown', dismiss, true);
    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('blur', dismiss);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      clearTimers();
      window.removeEventListener('pointerdown', dismiss, true);
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('blur', dismiss);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [clearTimers, hideTooltip]);

  const activeMeasureKey = active ? `${active.content}|${active.shortcut ?? ''}` : '';

  useLayoutEffect(() => {
    const el = tooltipElRef.current;
    if (!el || !active) return;
    const width = el.offsetWidth;
    const height = el.offsetHeight;
    if (width > 0 && height > 0) {
      setMeasured((prev) =>
        prev &&
        prev.key === activeMeasureKey &&
        Math.abs(prev.width - width) < 1 &&
        Math.abs(prev.height - height) < 1
          ? prev
          : { key: activeMeasureKey, width, height },
      );
    }
  }, [active, activeMeasureKey]);

  const placement = useMemo(() => {
    if (!active) return null;
    const vw = typeof window !== 'undefined' ? window.innerWidth : 1280;
    const vh = typeof window !== 'undefined' ? window.innerHeight : 800;
    const width =
      measured && measured.key === activeMeasureKey
        ? measured.width
        : estimateTooltipWidth(active.content, active.shortcut);
    const height = measured && measured.key === activeMeasureKey ? measured.height : 28;
    return computeTooltipPlacement(active.rect, width, height, active.side, vw, vh);
  }, [active, activeMeasureKey, measured]);

  // Keep context value strictly stable so triggers never re-render or cancel on tooltip open!
  const contextValue = useMemo<TooltipContextValue>(
    () => ({
      showTooltip,
      hideTooltip,
    }),
    [showTooltip, hideTooltip],
  );

  const initialOffset =
    placement?.side === 'top'
      ? { x: 0, y: 5 }
      : placement?.side === 'bottom'
        ? { x: 0, y: -5 }
        : placement?.side === 'right'
          ? { x: -5, y: 0 }
          : { x: 5, y: 0 };

  return (
    <DarkroomTooltipContext.Provider value={contextValue}>
      {children}

      {typeof document !== 'undefined' &&
        createPortal(
          <AnimatePresence>
            {active && placement && (
              <motion.div
                key="kroma-darkroom-tooltip"
                ref={tooltipElRef}
                id={active.id}
                role="tooltip"
                initial={{
                  opacity: 0,
                  scale: 0.94,
                  x: placement.left + initialOffset.x,
                  y: placement.top + initialOffset.y,
                }}
                animate={{
                  opacity: 1,
                  scale: 1,
                  x: placement.left,
                  y: placement.top,
                }}
                exit={{
                  opacity: 0,
                  scale: 0.95,
                  x: placement.left + initialOffset.x * 0.6,
                  y: placement.top + initialOffset.y * 0.6,
                  transition: { duration: 0.1, ease: 'easeOut' },
                }}
                transition={TOOLTIP_SPRING}
                style={{ left: 0, top: 0 }}
                className="fixed z-[90] pointer-events-none select-none px-2.5 py-1.5 rounded-lg bg-surface-highest/96 backdrop-blur-md border border-hairline-strong text-ink-primary shadow-[0_10px_28px_rgba(0,0,0,0.72)] flex items-center gap-1.5 whitespace-nowrap max-w-[260px]"
              >
                <span className="font-body text-[11px] font-medium leading-none truncate">
                  {active.content}
                </span>

                {active.shortcut && (
                  <kbd className="tabular-nums-mono text-[9px] leading-none px-1.5 py-0.5 rounded bg-canvas/90 border border-hairline text-amber-soft shrink-0">
                    {active.shortcut}
                  </kbd>
                )}

                {/* Precision Directional Caret */}
                {placement.side === 'top' && (
                  <span
                    style={{ left: `${placement.arrowOffsetPx}px` }}
                    className="absolute -bottom-[5px] -translate-x-1/2 w-2 h-2 rotate-45 bg-surface-highest border-r border-b border-hairline-strong"
                  />
                )}
                {placement.side === 'bottom' && (
                  <span
                    style={{ left: `${placement.arrowOffsetPx}px` }}
                    className="absolute -top-[5px] -translate-x-1/2 w-2 h-2 rotate-45 bg-surface-highest border-l border-t border-hairline-strong"
                  />
                )}
                {placement.side === 'right' && (
                  <span
                    style={{ top: `${placement.arrowOffsetPx}px` }}
                    className="absolute -left-[5px] -translate-y-1/2 w-2 h-2 rotate-45 bg-surface-highest border-l border-b border-hairline-strong"
                  />
                )}
                {placement.side === 'left' && (
                  <span
                    style={{ top: `${placement.arrowOffsetPx}px` }}
                    className="absolute -right-[5px] -translate-y-1/2 w-2 h-2 rotate-45 bg-surface-highest border-r border-t border-hairline-strong"
                  />
                )}
              </motion.div>
            )}
          </AnimatePresence>,
          document.body,
        )}
    </DarkroomTooltipContext.Provider>
  );
}

/**
 * Low-level hook for binding the shared Motion tooltip directly onto any interactive element.
 */
export function useDarkroomTooltip(
  content: string | undefined,
  side: TooltipSide = 'auto',
  shortcut?: string | null,
) {
  const ctx = useContext(DarkroomTooltipContext);
  const tooltipId = useId();

  useEffect(() => {
    return () => {
      ctx?.hideTooltip(tooltipId);
    };
  }, [ctx, tooltipId]);

  const onEnter = useCallback(
    (el: HTMLElement) => {
      if (!ctx || !content) return;
      ctx.showTooltip(tooltipId, content, toTriggerRect(el), side, shortcut);
    },
    [ctx, tooltipId, content, side, shortcut],
  );

  const onLeave = useCallback(() => {
    ctx?.hideTooltip(tooltipId);
  }, [ctx, tooltipId]);

  return {
    tooltipId,
    onEnter,
    onLeave,
  };
}

interface DarkroomTooltipProps {
  readonly content?: string;
  readonly shortcut?: string;
  readonly side?: TooltipSide;
  readonly className?: string;
  readonly children: ReactNode;
}

/**
 * Declarative wrapper for showing the shared Motion tooltip on any element (including disabled controls).
 */
export function DarkroomTooltip({
  content,
  shortcut,
  side = 'auto',
  className = 'inline-flex',
  children,
}: DarkroomTooltipProps) {
  const { onEnter, onLeave } = useDarkroomTooltip(content, side, shortcut);

  if (!content) {
    return <>{children}</>;
  }

  return (
    <span
      className={className}
      onMouseEnter={(e) => onEnter(e.currentTarget)}
      onMouseLeave={onLeave}
      onFocus={(e) => onEnter(e.currentTarget)}
      onBlur={onLeave}
    >
      {children}
    </span>
  );
}
