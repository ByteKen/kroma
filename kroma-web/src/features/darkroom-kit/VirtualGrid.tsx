import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import {
  areGridWindowsEqual,
  areLinearWindowsEqual,
  computeVirtualGridWindow,
  computeVirtualLinearWindow,
  type VirtualGridWindow,
  type VirtualLinearWindow,
} from './virtualWindow';

export interface VirtualGridHandle {
  scrollToIndex: (index: number, smooth?: boolean) => void;
}

interface VirtualGridProps<T> {
  readonly items: readonly T[];
  readonly getKey: (item: T) => string;
  readonly renderItem: (item: T, index: number) => ReactNode;
  /** Minimum card width in px when `fixedColumns` is omitted. */
  readonly minItemWidth?: number;
  /** Optional explicit column count (e.g. `1` for Loupe or `2` for 2-Up Compare). */
  readonly fixedColumns?: number;
  /** Extra height (in px) below the aspect-[3/4] stage for the card footer. */
  readonly footerHeightPx?: number;
  readonly gapPx?: number;
  readonly paddingPx?: number;
  readonly bottomPaddingPx?: number;
  readonly overscanRows?: number;
  readonly maxContentWidthPx?: number;
  readonly className?: string;
}

function VirtualGridInner<T>(
  {
    items,
    getKey,
    renderItem,
    minItemWidth = 195,
    fixedColumns,
    footerHeightPx = 56,
    gapPx = 14,
    paddingPx = 16,
    bottomPaddingPx,
    overscanRows = 2,
    maxContentWidthPx,
    className = '',
  }: VirtualGridProps<T>,
  ref: React.ForwardedRef<VirtualGridHandle>,
) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const [viewport, setViewport] = useState({ width: 960, height: 720 });

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => {
      if (el.clientWidth > 0 && el.clientHeight > 0) {
        setViewport((prev) =>
          prev.width === el.clientWidth && prev.height === el.clientHeight
            ? prev
            : { width: el.clientWidth, height: el.clientHeight },
        );
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const usableWidth = Math.max(
    180,
    (maxContentWidthPx ? Math.min(viewport.width, maxContentWidthPx) : viewport.width) -
      paddingPx * 2,
  );
  const cols = fixedColumns
    ? Math.max(1, fixedColumns)
    : Math.max(2, Math.floor((usableWidth + gapPx) / (minItemWidth + gapPx)));
  const cardWidth = Math.max(110, (usableWidth - gapPx * (cols - 1)) / cols);
  // aspect-[3/4] image stage + footer
  const rowHeight = Math.round((cardWidth * 4) / 3 + footerHeightPx);
  const rowStride = rowHeight + gapPx;
  const totalRows = Math.ceil(items.length / cols);
  const totalHeight = Math.max(0, totalRows * rowStride - (totalRows > 0 ? gapPx : 0));

  const computeCurrentWindow = useCallback((): VirtualGridWindow => {
    const scrollTop = scrollRef.current?.scrollTop ?? 0;
    return computeVirtualGridWindow({
      scrollTop,
      viewportHeight: viewport.height,
      paddingPx,
      rowStride,
      totalRows,
      cols,
      itemCount: items.length,
      overscanRows,
    });
  }, [viewport.height, paddingPx, rowStride, totalRows, cols, items.length, overscanRows]);

  const [windowRange, setWindowRange] = useState<VirtualGridWindow>(() => computeCurrentWindow());

  const syncWindow = useCallback(() => {
    const next = computeCurrentWindow();
    setWindowRange((prev) => (areGridWindowsEqual(prev, next) ? prev : next));
  }, [computeCurrentWindow]);

  useLayoutEffect(() => {
    syncWindow();
  }, [syncWindow]);

  const scheduleScrollSync = useCallback(() => {
    if (rafIdRef.current !== null) return;
    rafIdRef.current = window.requestAnimationFrame(() => {
      rafIdRef.current = null;
      syncWindow();
    });
  }, [syncWindow]);

  useEffect(() => {
    return () => {
      if (rafIdRef.current !== null) {
        window.cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, []);

  useImperativeHandle(
    ref,
    () => ({
      scrollToIndex(index: number, smooth = false) {
        const el = scrollRef.current;
        if (!el || index < 0 || index >= items.length) return;
        const row = Math.floor(index / cols);
        const targetTop = Math.max(
          0,
          paddingPx + row * rowStride - (el.clientHeight - rowHeight) / 2,
        );
        el.scrollTo({ top: targetTop, behavior: smooth ? 'smooth' : 'auto' });
      },
    }),
    [cols, items.length, paddingPx, rowHeight, rowStride],
  );

  // Clamp synchronously on render when item list shrinks between paints
  const clampedStartRow = Math.min(windowRange.startRow, totalRows);
  const startIdx = Math.min(items.length, windowRange.startIdx);
  const endIdx = Math.min(items.length, Math.max(startIdx, windowRange.endIdx));
  const visibleSlice = items.slice(startIdx, endIdx);

  return (
    <div
      ref={scrollRef}
      onScroll={scheduleScrollSync}
      className={`flex-1 overflow-y-auto ${className}`}
    >
      <div
        style={{
          padding: `${paddingPx}px ${paddingPx}px ${bottomPaddingPx ?? paddingPx}px ${paddingPx}px`,
          maxWidth: maxContentWidthPx ? `${maxContentWidthPx}px` : undefined,
          margin: maxContentWidthPx ? '0 auto' : undefined,
        }}
      >
        <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
          <div
            style={{
              position: 'absolute',
              top: `${clampedStartRow * rowStride}px`,
              left: 0,
              right: 0,
              display: 'grid',
              gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
              gap: `${gapPx}px`,
            }}
          >
            {visibleSlice.map((item, offset) => (
              <div key={getKey(item)}>{renderItem(item, startIdx + offset)}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export const VirtualGrid = forwardRef(VirtualGridInner) as <T>(
  props: VirtualGridProps<T> & { ref?: React.Ref<VirtualGridHandle> },
) => ReactNode;

interface VirtualHorizontalStripProps<T> {
  readonly items: readonly T[];
  readonly activeIndex: number;
  readonly itemWidthPx: number;
  readonly gapPx: number;
  readonly getKey: (item: T) => string;
  readonly renderItem: (item: T, index: number) => ReactNode;
  readonly centerActive?: boolean;
  readonly className?: string;
}

/**
 * Windowed horizontal filmstrip for 2,000+ photo Develop & Lightbox sessions (mounts only visible thumbnails).
 */
export function VirtualHorizontalStrip<T>({
  items,
  activeIndex,
  itemWidthPx,
  gapPx,
  getKey,
  renderItem,
  centerActive = false,
  className = '',
}: VirtualHorizontalStripProps<T>) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const [viewportWidth, setViewportWidth] = useState(900);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => {
      if (el.clientWidth > 0) {
        setViewportWidth((prev) => (prev === el.clientWidth ? prev : el.clientWidth));
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const stride = itemWidthPx + gapPx;
  const totalWidth = Math.max(0, items.length * stride - (items.length > 0 ? gapPx : 0));
  const overscan = 6;

  const computeCurrentWindow = useCallback((): VirtualLinearWindow => {
    const scrollLeft = scrollRef.current?.scrollLeft ?? 0;
    return computeVirtualLinearWindow({
      scrollOffset: scrollLeft,
      viewportSize: viewportWidth,
      stride,
      itemCount: items.length,
      overscan,
    });
  }, [viewportWidth, stride, items.length]);

  const [windowRange, setWindowRange] = useState<VirtualLinearWindow>(() => computeCurrentWindow());

  const syncWindow = useCallback(() => {
    const next = computeCurrentWindow();
    setWindowRange((prev) => (areLinearWindowsEqual(prev, next) ? prev : next));
  }, [computeCurrentWindow]);

  useLayoutEffect(() => {
    syncWindow();
  }, [syncWindow]);

  const scheduleScrollSync = useCallback(() => {
    if (rafIdRef.current !== null) return;
    rafIdRef.current = window.requestAnimationFrame(() => {
      rafIdRef.current = null;
      syncWindow();
    });
  }, [syncWindow]);

  useEffect(() => {
    return () => {
      if (rafIdRef.current !== null) {
        window.cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, []);

  // Dedicated non-passive wheel listener: maps standard vertical mouse wheel directly
  // to horizontal filmstrip scrolling (no Shift key required) while preserving native
  // horizontal trackpad swipes and Shift+Wheel.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const onWheel = (event: WheelEvent) => {
      const maxScroll = Math.max(0, el.scrollWidth - el.clientWidth);
      if (maxScroll <= 0) return;

      const scale =
        event.deltaMode === 1 ? 28 : event.deltaMode === 2 ? el.clientWidth : 1;
      const dx = event.deltaX * scale;
      const dy = event.deltaY * scale;

      if (dy !== 0 && Math.abs(dy) >= Math.abs(dx)) {
        event.preventDefault();
        event.stopPropagation();
        el.scrollLeft = Math.max(0, Math.min(maxScroll, el.scrollLeft + dy));
        scheduleScrollSync();
      }
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [totalWidth, scheduleScrollSync]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || activeIndex < 0) return;
    const itemLeft = activeIndex * stride;
    if (centerActive) {
      const targetLeft = Math.max(0, itemLeft - (el.clientWidth - itemWidthPx) / 2);
      el.scrollTo({ left: targetLeft, behavior: 'smooth' });
      return;
    }
    const itemRight = itemLeft + itemWidthPx;
    if (itemLeft < el.scrollLeft + 24) {
      el.scrollTo({ left: Math.max(0, itemLeft - 48), behavior: 'smooth' });
    } else if (itemRight > el.scrollLeft + el.clientWidth - 24) {
      el.scrollTo({ left: itemRight - el.clientWidth + 48, behavior: 'smooth' });
    }
  }, [activeIndex, itemWidthPx, stride, centerActive]);

  const startIdx = Math.min(items.length, windowRange.startIdx);
  const endIdx = Math.min(items.length, Math.max(startIdx, windowRange.endIdx));
  const visibleSlice = items.slice(startIdx, endIdx);
  const shouldCenterShortStrip = centerActive && totalWidth > 0 && totalWidth < viewportWidth;

  return (
    <div
      ref={scrollRef}
      onScroll={scheduleScrollSync}
      className={`overflow-y-hidden ${className}`}
    >
      <div
        style={{
          width: `${totalWidth}px`,
          height: '100%',
          position: 'relative',
          flexShrink: 0,
          margin: shouldCenterShortStrip ? '0 auto' : undefined,
        }}
      >
        <div
          style={{
            position: 'absolute',
            left: `${startIdx * stride}px`,
            top: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            gap: `${gapPx}px`,
          }}
        >
          {visibleSlice.map((item, offset) => (
            <div key={getKey(item)}>{renderItem(item, startIdx + offset)}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface VirtualVerticalListProps<T> {
  readonly items: readonly T[];
  readonly rowHeightPx: number;
  readonly gapPx?: number;
  readonly overscan?: number;
  readonly getKey: (item: T) => string;
  readonly renderItem: (item: T, index: number) => ReactNode;
  readonly className?: string;
  readonly onMouseLeave?: () => void;
  readonly onScroll?: () => void;
}

/**
 * Windowed vertical list for 500+ subject rows in Studio/sidebar panels (mounts only visible rows).
 */
export function VirtualVerticalList<T>({
  items,
  rowHeightPx,
  gapPx = 4,
  overscan = 4,
  getKey,
  renderItem,
  className = '',
  onMouseLeave,
  onScroll,
}: VirtualVerticalListProps<T>) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const [viewportHeight, setViewportHeight] = useState(220);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => {
      if (el.clientHeight > 0) {
        setViewportHeight((prev) => (prev === el.clientHeight ? prev : el.clientHeight));
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const stride = rowHeightPx + gapPx;
  const totalHeight = Math.max(0, items.length * stride - (items.length > 0 ? gapPx : 0));

  const computeCurrentWindow = useCallback((): VirtualLinearWindow => {
    const scrollTop = scrollRef.current?.scrollTop ?? 0;
    return computeVirtualLinearWindow({
      scrollOffset: scrollTop,
      viewportSize: viewportHeight,
      stride,
      itemCount: items.length,
      overscan,
    });
  }, [viewportHeight, stride, items.length, overscan]);

  const [windowRange, setWindowRange] = useState<VirtualLinearWindow>(() => computeCurrentWindow());

  const syncWindow = useCallback(() => {
    const next = computeCurrentWindow();
    setWindowRange((prev) => (areLinearWindowsEqual(prev, next) ? prev : next));
  }, [computeCurrentWindow]);

  useLayoutEffect(() => {
    syncWindow();
  }, [syncWindow]);

  const scheduleScrollSync = useCallback(() => {
    if (rafIdRef.current !== null) return;
    rafIdRef.current = window.requestAnimationFrame(() => {
      rafIdRef.current = null;
      syncWindow();
    });
  }, [syncWindow]);

  useEffect(() => {
    return () => {
      if (rafIdRef.current !== null) {
        window.cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, []);

  const startIdx = Math.min(items.length, windowRange.startIdx);
  const endIdx = Math.min(items.length, Math.max(startIdx, windowRange.endIdx));
  const visibleSlice = items.slice(startIdx, endIdx);

  return (
    <div
      ref={scrollRef}
      onMouseLeave={onMouseLeave}
      onScroll={() => {
        scheduleScrollSync();
        onScroll?.();
      }}
      className={`overflow-y-auto ${className}`}
    >
      <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
        <div
          style={{
            position: 'absolute',
            top: `${startIdx * stride}px`,
            left: 0,
            right: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: `${gapPx}px`,
          }}
        >
          {visibleSlice.map((item, offset) => (
            <div key={getKey(item)} style={{ height: `${rowHeightPx}px` }}>
              {renderItem(item, startIdx + offset)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
