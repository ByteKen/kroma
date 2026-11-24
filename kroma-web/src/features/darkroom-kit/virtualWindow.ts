/**
 * Pure window-range math for Kroma's virtualized contact sheets, vertical lists, and horizontal
 * filmstrips. Quantizing scroll updates to visible index bounds lets scroll handlers skip React
 * re-renders completely while scrolling within the same row/item bucket (~90% fewer renders).
 */

export interface VirtualGridWindow {
  readonly startRow: number;
  readonly endRow: number;
  readonly startIdx: number;
  readonly endIdx: number;
}

export interface VirtualLinearWindow {
  readonly startIdx: number;
  readonly endIdx: number;
}

export function computeVirtualGridWindow(params: {
  readonly scrollTop: number;
  readonly viewportHeight: number;
  readonly paddingPx: number;
  readonly rowStride: number;
  readonly totalRows: number;
  readonly cols: number;
  readonly itemCount: number;
  readonly overscanRows: number;
}): VirtualGridWindow {
  const {
    scrollTop,
    viewportHeight,
    paddingPx,
    rowStride,
    totalRows,
    cols,
    itemCount,
    overscanRows,
  } = params;
  if (itemCount <= 0 || totalRows <= 0 || rowStride <= 0 || cols <= 0) {
    return { startRow: 0, endRow: 0, startIdx: 0, endIdx: 0 };
  }
  const safeScrollTop = Math.max(0, scrollTop);
  const startRow = Math.max(
    0,
    Math.min(totalRows, Math.floor((safeScrollTop - paddingPx) / rowStride) - overscanRows),
  );
  const endRow = Math.min(
    totalRows,
    Math.max(
      startRow,
      Math.ceil((safeScrollTop + Math.max(1, viewportHeight) - paddingPx) / rowStride) +
        overscanRows,
    ),
  );
  const startIdx = Math.min(itemCount, startRow * cols);
  const endIdx = Math.min(itemCount, endRow * cols);
  return { startRow, endRow, startIdx, endIdx };
}

export function computeVirtualLinearWindow(params: {
  readonly scrollOffset: number;
  readonly viewportSize: number;
  readonly stride: number;
  readonly itemCount: number;
  readonly overscan: number;
}): VirtualLinearWindow {
  const { scrollOffset, viewportSize, stride, itemCount, overscan } = params;
  if (itemCount <= 0 || stride <= 0) {
    return { startIdx: 0, endIdx: 0 };
  }
  const safeOffset = Math.max(0, scrollOffset);
  const startIdx = Math.max(
    0,
    Math.min(itemCount, Math.floor(safeOffset / stride) - overscan),
  );
  const endIdx = Math.min(
    itemCount,
    Math.max(
      startIdx,
      Math.ceil((safeOffset + Math.max(1, viewportSize)) / stride) + overscan,
    ),
  );
  return { startIdx, endIdx };
}

export function areGridWindowsEqual(a: VirtualGridWindow, b: VirtualGridWindow): boolean {
  return (
    a.startRow === b.startRow &&
    a.endRow === b.endRow &&
    a.startIdx === b.startIdx &&
    a.endIdx === b.endIdx
  );
}

export function areLinearWindowsEqual(a: VirtualLinearWindow, b: VirtualLinearWindow): boolean {
  return a.startIdx === b.startIdx && a.endIdx === b.endIdx;
}
