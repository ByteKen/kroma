import { motion } from 'framer-motion';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import type { LiveRgbHistogram } from '../../application/ports/studioGateway';
import {
  estimatePreviewHistogramBins,
  type DevelopRecipe,
  type ParametricToneCurve,
} from '../../domain/develop';
import {
  buildToneCurveSvgPaths,
  DEFAULT_TONE_CURVE_POINTS,
  evaluateChannelToneCurve,
  getEffectiveCurvePoints,
  insertToneCurvePoint,
  isToneCurveChannelModified,
  isToneCurveModified,
  moveToneCurvePoint,
  removeToneCurvePoint,
  setCurveChannelPoints,
  TONE_CURVE_SHAPES,
  type ToneCurveChannel,
  type ToneCurvePoint,
} from '../../domain/toneCurve';
import { TACTILE_SPRING } from '../darkroom-kit';

const VIEW_W = 240;
const VIEW_H = 192;
const DRAG_OUT_REMOVE_MARGIN_PX = 38;

const CURVE_CHANNELS: readonly {
  readonly id: ToneCurveChannel;
  readonly label: string;
  readonly stroke: string;
  readonly dotBg: string;
}[] = [
  { id: 'rgb', label: 'RGB', stroke: '#F59E0B', dotBg: '#E5E7EB' },
  { id: 'red', label: 'R', stroke: '#EF4444', dotBg: '#EF4444' },
  { id: 'green', label: 'G', stroke: '#22C55E', dotBg: '#22C55E' },
  { id: 'blue', label: 'B', stroke: '#3B82F6', dotBg: '#3B82F6' },
];

interface InteractiveToneCurveEditorProps {
  readonly curve: ParametricToneCurve;
  readonly recipe: DevelopRecipe;
  readonly liveHistogram: LiveRgbHistogram | null;
  readonly onChange: (nextCurve: ParametricToneCurve) => void;
}

function buildHistogramSilhouettePath(bins: readonly number[]): string {
  if (bins.length < 2) return '';
  const step = VIEW_W / (bins.length - 1);
  const parts: string[] = [`M 0,${VIEW_H}`];
  for (let i = 0; i < bins.length; i++) {
    const x = (i * step).toFixed(1);
    const y = (VIEW_H - Math.min(1, Math.max(0, bins[i])) * (VIEW_H * 0.78)).toFixed(1);
    parts.push(`L ${x},${y}`);
  }
  parts.push(`L ${VIEW_W},${VIEW_H} Z`);
  return parts.join(' ');
}

/**
 * Interactive Lightroom-style Point + Per-Channel (`RGB` / `Red` / `Green` / `Blue`) Tone Curve
 * Editor with live histogram backdrop, ghosted channel overlays, and Motion spring handles.
 */
export const InteractiveToneCurveEditor = memo(function InteractiveToneCurveEditor({
  curve,
  recipe,
  liveHistogram,
  onChange,
}: InteractiveToneCurveEditorProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [channel, setChannel] = useState<ToneCurveChannel>('rgb');
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [dragWillRemove, setDragWillRemove] = useState(false);

  const channelMeta = CURVE_CHANNELS.find((c) => c.id === channel) ?? CURVE_CHANNELS[0];
  const points = useMemo(() => getEffectiveCurvePoints(curve, channel), [curve, channel]);
  const { linePath, areaPath } = useMemo(
    () => buildToneCurveSvgPaths(curve, VIEW_W, VIEW_H, 76, channel),
    [curve, channel],
  );
  const ghostCurves = useMemo(
    () =>
      CURVE_CHANNELS.filter((c) => c.id !== channel && isToneCurveChannelModified(curve, c.id)).map(
        (c) => ({
          id: c.id,
          stroke: c.stroke,
          path: buildToneCurveSvgPaths(curve, VIEW_W, VIEW_H, 56, c.id).linePath,
        }),
      ),
    [curve, channel],
  );
  const anyModified = isToneCurveModified(curve);

  const histogramSilhouette = useMemo(() => {
    const hist = liveHistogram ?? estimatePreviewHistogramBins(recipe);
    if (channel === 'red') return buildHistogramSilhouettePath(hist.red);
    if (channel === 'green') return buildHistogramSilhouettePath(hist.green);
    if (channel === 'blue') return buildHistogramSilhouettePath(hist.blue);
    const luma = hist.red.map(
      (r, i) => 0.2126 * r + 0.7152 * (hist.green[i] ?? 0) + 0.0722 * (hist.blue[i] ?? 0),
    );
    return buildHistogramSilhouettePath(luma);
  }, [liveHistogram, recipe, channel]);

  const commitPoints = useCallback(
    (nextPoints: readonly ToneCurvePoint[]) => {
      onChange(setCurveChannelPoints(curve, channel, nextPoints));
    },
    [curve, channel, onChange],
  );

  const clientToNormalized = useCallback((clientX: number, clientY: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) {
      return { x: 0.5, y: 0.5, outsideMargin: false };
    }
    const rawX = (clientX - rect.left) / rect.width;
    const rawY = 1 - (clientY - rect.top) / rect.height;
    const outsideMargin =
      clientX < rect.left - DRAG_OUT_REMOVE_MARGIN_PX ||
      clientX > rect.right + DRAG_OUT_REMOVE_MARGIN_PX ||
      clientY < rect.top - DRAG_OUT_REMOVE_MARGIN_PX ||
      clientY > rect.bottom + DRAG_OUT_REMOVE_MARGIN_PX;

    return {
      x: Math.min(1, Math.max(0, rawX)),
      y: Math.min(1, Math.max(0, rawY)),
      outsideMargin,
    };
  }, []);

  const handleCanvasPointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const { x, y } = clientToNormalized(event.clientX, event.clientY);

    const rect = svgRef.current?.getBoundingClientRect();
    const hitRadiusX = rect && rect.width > 0 ? 12 / rect.width : 0.05;
    const hitRadiusY = rect && rect.height > 0 ? 12 / rect.height : 0.06;

    let hitIdx = -1;
    let bestDist = Infinity;
    for (let i = 0; i < points.length; i++) {
      const dx = (points[i].x - x) / hitRadiusX;
      const dy = (points[i].y - y) / hitRadiusY;
      const d = Math.hypot(dx, dy);
      if (d <= 1.15 && d < bestDist) {
        bestDist = d;
        hitIdx = i;
      }
    }

    event.currentTarget.setPointerCapture(event.pointerId);
    if (hitIdx !== -1) {
      setActiveIndex(hitIdx);
      setDraggingIndex(hitIdx);
      setDragWillRemove(false);
      return;
    }

    const inserted = insertToneCurvePoint(points, x, y);
    commitPoints(inserted.points);
    setActiveIndex(inserted.index);
    setDraggingIndex(inserted.index);
    setDragWillRemove(false);
  };

  const handleCanvasPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (draggingIndex === null) return;
    const { x, y, outsideMargin } = clientToNormalized(event.clientX, event.clientY);
    const isInterior = draggingIndex > 0 && draggingIndex < points.length - 1;
    setDragWillRemove(isInterior && outsideMargin);
    if (isInterior && outsideMargin) {
      return;
    }
    const next = moveToneCurvePoint(points, draggingIndex, x, y);
    commitPoints(next);
  };

  const handleCanvasPointerUp = (event: React.PointerEvent<SVGSVGElement>) => {
    if (draggingIndex === null) return;
    const idx = draggingIndex;
    const shouldRemove = dragWillRemove && idx > 0 && idx < points.length - 1;
    setDraggingIndex(null);
    setDragWillRemove(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (shouldRemove) {
      commitPoints(removeToneCurvePoint(points, idx));
      setActiveIndex(null);
    }
  };

  const handleResetActiveChannel = () => {
    if (channel === 'rgb') {
      onChange({
        ...curve,
        highlights: 0,
        lights: 0,
        darks: 0,
        shadows: 0,
        points: undefined,
      });
    } else {
      onChange(setCurveChannelPoints(curve, channel, undefined));
    }
    setActiveIndex(null);
  };

  const handleResetAllChannels = () => {
    onChange({
      highlights: 0,
      lights: 0,
      darks: 0,
      shadows: 0,
      points: undefined,
      redPoints: undefined,
      greenPoints: undefined,
      bluePoints: undefined,
    });
    setActiveIndex(null);
  };

  const inspectedIndex = draggingIndex ?? hoveredIndex ?? activeIndex;
  const inspectedPoint =
    inspectedIndex !== null && inspectedIndex < points.length ? points[inspectedIndex] : null;
  const inspectedY = inspectedPoint
    ? evaluateChannelToneCurve(curve, inspectedPoint.x, channel)
    : null;
  const gradId = `kroma-curve-area-${channel}`;

  return (
    <div className="flex flex-col gap-2">
      {/* Header + Per-Channel Selector (RGB / R / G / B) */}
      <div className="flex items-center justify-between gap-2">
        <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
          Tone Curve
        </span>
        <div className="flex items-center gap-1">
          {CURVE_CHANNELS.map((c) => {
            const isSelected = c.id === channel;
            const chEdited = isToneCurveChannelModified(curve, c.id);
            return (
              <button
                key={c.id}
                type="button"
                onClick={() => {
                  setChannel(c.id);
                  setActiveIndex(null);
                }}
                title={`${c.label} tone curve`}
                className={`relative px-1.5 py-0.5 rounded tabular-nums-mono text-[9.5px] border flex items-center gap-1 transition-colors cursor-pointer ${
                  isSelected
                    ? 'bg-surface-active border-amber-core/60 text-ink-primary font-semibold'
                    : 'bg-canvas border-hairline text-ink-telemetry hover:text-ink-primary'
                }`}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: c.dotBg }}
                />
                <span>{c.label}</span>
                {chEdited && (
                  <span className="w-1 h-1 rounded-full bg-amber-core shrink-0" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Quick Curve Presets + Reset */}
      <div className="flex items-center justify-between gap-1">
        <div className="flex items-center gap-1">
          {TONE_CURVE_SHAPES.filter((s) => s.id !== 'linear').map((shape) => (
            <button
              key={shape.id}
              type="button"
              onClick={() => {
                commitPoints(shape.points);
                setActiveIndex(null);
              }}
              className="px-1.5 py-0.5 rounded bg-canvas border border-hairline hover:border-amber-core/50 tabular-nums-mono text-[9px] text-ink-secondary hover:text-amber-soft transition-colors cursor-pointer"
            >
              {shape.label}
            </button>
          ))}
        </div>
        {anyModified && (
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={handleResetActiveChannel}
              className="px-1.5 py-0.5 rounded bg-canvas border border-hairline hover:border-amber-core/50 tabular-nums-mono text-[9px] text-ink-secondary hover:text-amber-soft transition-colors cursor-pointer"
            >
              Reset {channelMeta.label}
            </button>
            <button
              type="button"
              onClick={handleResetAllChannels}
              className="px-1.5 py-0.5 rounded bg-canvas border border-hairline hover:border-amber-core/50 tabular-nums-mono text-[9px] text-ink-telemetry hover:text-amber-soft transition-colors cursor-pointer"
            >
              All
            </button>
          </div>
        )}
      </div>

      {/* Interactive Direct-Manipulation Curve Plot */}
      <div className="relative rounded-md bg-canvas border border-hairline overflow-hidden select-none">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          onPointerDown={handleCanvasPointerDown}
          onPointerMove={handleCanvasPointerMove}
          onPointerUp={handleCanvasPointerUp}
          onPointerCancel={handleCanvasPointerUp}
          className={`w-full aspect-[5/4] block touch-none ${
            draggingIndex !== null ? 'cursor-grabbing' : 'cursor-crosshair'
          }`}
          role="application"
          aria-label={`Interactive ${channelMeta.label} tone curve editor`}
        >
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={channelMeta.stroke} stopOpacity="0.2" />
              <stop offset="100%" stopColor={channelMeta.stroke} stopOpacity="0.01" />
            </linearGradient>
          </defs>

          {/* Channel Histogram Backdrop */}
          {histogramSilhouette && (
            <path d={histogramSilhouette} fill="rgba(255, 255, 255, 0.06)" />
          )}

          {/* 4x4 Exposure Stop Grid */}
          {[0.25, 0.5, 0.75].map((ratio) => (
            <g key={`grid-${ratio}`}>
              <line
                x1={VIEW_W * ratio}
                y1={0}
                x2={VIEW_W * ratio}
                y2={VIEW_H}
                stroke="#252830"
                strokeWidth="1"
              />
              <line
                x1={0}
                y1={VIEW_H * ratio}
                x2={VIEW_W}
                y2={VIEW_H * ratio}
                stroke="#252830"
                strokeWidth="1"
              />
            </g>
          ))}

          {/* 1:1 Linear Reference Diagonal */}
          <line
            x1="0"
            y1={VIEW_H}
            x2={VIEW_W}
            y2="0"
            stroke="#3B3F4A"
            strokeDasharray="3 3"
            strokeWidth="1"
          />

          {/* Ghosted Inactive Modified Channel Curves */}
          {ghostCurves.map((ghost) => (
            <path
              key={ghost.id}
              d={ghost.path}
              fill="none"
              stroke={ghost.stroke}
              strokeOpacity="0.32"
              strokeWidth="1.25"
              className="pointer-events-none"
            />
          ))}

          {/* Precision Crosshair Guides While Dragging */}
          {draggingIndex !== null && points[draggingIndex] && !dragWillRemove && (
            <g className="pointer-events-none">
              <line
                x1={points[draggingIndex].x * VIEW_W}
                y1={0}
                x2={points[draggingIndex].x * VIEW_W}
                y2={VIEW_H}
                stroke={channelMeta.stroke}
                strokeOpacity="0.35"
                strokeDasharray="2 2"
                strokeWidth="1"
              />
              <line
                x1={0}
                y1={(1 - points[draggingIndex].y) * VIEW_H}
                x2={VIEW_W}
                y2={(1 - points[draggingIndex].y) * VIEW_H}
                stroke={channelMeta.stroke}
                strokeOpacity="0.35"
                strokeDasharray="2 2"
                strokeWidth="1"
              />
            </g>
          )}

          {/* Curve Glow Area & Active Spline Stroke */}
          <path d={areaPath} fill={`url(#${gradId})`} className="pointer-events-none" />
          <path
            d={linePath}
            fill="none"
            stroke={channelMeta.stroke}
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="pointer-events-none"
          />

          {/* Interactive Control Points */}
          {points.map((pt, idx) => {
            const cx = pt.x * VIEW_W;
            const cy = (1 - pt.y) * VIEW_H;
            const isEndpoint = idx === 0 || idx === points.length - 1;
            const isDragging = draggingIndex === idx;
            const isActive = activeIndex === idx;
            const isHovered = hoveredIndex === idx;
            const willDeleteThis = isDragging && dragWillRemove;

            return (
              <motion.g
                key={`${channel}-${idx}-${isEndpoint ? 'end' : 'mid'}`}
                tabIndex={0}
                role="slider"
                aria-label={
                  idx === 0
                    ? `${channelMeta.label} black point`
                    : idx === points.length - 1
                      ? `${channelMeta.label} white point`
                      : `${channelMeta.label} control point ${idx}`
                }
                aria-valuemin={0}
                aria-valuemax={255}
                aria-valuenow={Math.round(pt.y * 255)}
                onPointerEnter={() => setHoveredIndex(idx)}
                onPointerLeave={() => setHoveredIndex((prev) => (prev === idx ? null : prev))}
                onFocus={() => setActiveIndex(idx)}
                onDoubleClick={(event) => {
                  event.stopPropagation();
                  commitPoints(removeToneCurvePoint(points, idx));
                  setActiveIndex(null);
                }}
                onContextMenu={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  commitPoints(removeToneCurvePoint(points, idx));
                  setActiveIndex(null);
                }}
                onKeyDown={(event) => {
                  const step = event.shiftKey ? 0.05 : 0.015;
                  if (event.key === 'ArrowUp') {
                    event.preventDefault();
                    commitPoints(moveToneCurvePoint(points, idx, pt.x, pt.y + step));
                  } else if (event.key === 'ArrowDown') {
                    event.preventDefault();
                    commitPoints(moveToneCurvePoint(points, idx, pt.x, pt.y - step));
                  } else if (event.key === 'ArrowRight') {
                    event.preventDefault();
                    commitPoints(moveToneCurvePoint(points, idx, pt.x + step, pt.y));
                  } else if (event.key === 'ArrowLeft') {
                    event.preventDefault();
                    commitPoints(moveToneCurvePoint(points, idx, pt.x - step, pt.y));
                  } else if (event.key === 'Delete' || event.key === 'Backspace') {
                    event.preventDefault();
                    commitPoints(removeToneCurvePoint(points, idx));
                    setActiveIndex(null);
                  }
                }}
                className="cursor-grab active:cursor-grabbing focus:outline-none"
              >
                <circle cx={cx} cy={cy} r={11} fill="transparent" />
                {(isDragging || isHovered || isActive) && (
                  <motion.circle
                    cx={cx}
                    cy={cy}
                    initial={{ r: 6, opacity: 0 }}
                    animate={{
                      r: willDeleteThis ? 11 : isDragging ? 10 : 8.5,
                      opacity: willDeleteThis ? 0.4 : 0.25,
                    }}
                    transition={TACTILE_SPRING}
                    fill={willDeleteThis ? '#EF4444' : channelMeta.stroke}
                  />
                )}
                <motion.circle
                  cx={cx}
                  cy={cy}
                  animate={{
                    r: isDragging ? 5.5 : isHovered || isActive ? 5 : isEndpoint ? 4.2 : 4.4,
                    opacity: willDeleteThis ? 0.45 : 1,
                  }}
                  transition={TACTILE_SPRING}
                  fill={
                    willDeleteThis
                      ? '#EF4444'
                      : isDragging || isActive
                        ? channelMeta.stroke
                        : '#111214'
                  }
                  stroke={willDeleteThis ? '#FCA5A5' : channelMeta.stroke}
                  strokeWidth={isEndpoint ? 1.8 : 2}
                />
              </motion.g>
            );
          })}
        </svg>

        {dragWillRemove && (
          <div className="pointer-events-none absolute top-2 left-2 px-1.5 py-0.5 rounded bg-crimson-clip/90 text-white tabular-nums-mono text-[9px] font-semibold">
            Release to remove point
          </div>
        )}
      </div>

      {/* Live Coordinate Readout */}
      <div className="flex items-center justify-between gap-2 tabular-nums-mono text-[9.5px] text-ink-telemetry">
        {inspectedPoint ? (
          <>
            <span style={{ color: channelMeta.stroke }}>
              {channelMeta.label} IN {Math.round(inspectedPoint.x * 255)} → OUT{' '}
              {Math.round((inspectedY ?? inspectedPoint.y) * 255)}
            </span>
            <span>
              {points.length} {points.length === 1 ? 'pt' : 'pts'}
            </span>
          </>
        ) : (
          <>
            <span className="truncate">Click to add • Drag off or 2× click to delete</span>
            <span className="shrink-0">
              {points.length === DEFAULT_TONE_CURVE_POINTS.length && !isToneCurveChannelModified(curve, channel)
                ? 'Linear'
                : `${points.length} pts`}
            </span>
          </>
        )}
      </div>
    </div>
  );
});
