import { memo, useMemo, useRef, useState } from 'react';
import type { CropTransformSettings } from '../../domain/develop';
import type { YuNetFaceOverlay } from '../../domain/photo';
import {
  createHealSpot,
  faceOverlayToOrientedGuide,
  pickAutoHealSourceUv,
  type HealSpot,
  type InpaintStrokeInput,
} from '../../domain/retouch';
import {
  computeViewportUvTransform,
  ZERO_VIEWPORT_PAN,
  type ViewportPan,
} from '../../domain/viewportZoom';
import type { HealToolOptions } from './RetouchPanel';

interface RetouchCanvasOverlayProps {
  readonly spots: readonly HealSpot[];
  readonly selectedSpotId: string | null;
  readonly showPins: boolean;
  readonly healTool: HealToolOptions;
  readonly pendingRemoveStrokes?: readonly InpaintStrokeInput[];
  readonly busyInpainting?: boolean;
  readonly faceOverlay: YuNetFaceOverlay | null;
  readonly cropTransform: CropTransformSettings;
  readonly applyCrop: boolean;
  readonly fit: 'cover' | 'contain';
  readonly viewportPadPx?: number;
  readonly viewportZoom?: number;
  readonly viewportPan?: ViewportPan;
  readonly naturalWidth: number;
  readonly naturalHeight: number;
  readonly samplePatchStats?: (u: number, v: number, radiusNorm: number) => {
    readonly meanR: number;
    readonly meanG: number;
    readonly meanB: number;
    readonly variance: number;
  } | null;
  readonly onSelectSpot: (spotId: string | null) => void;
  readonly onChangeSpots: (nextSpots: readonly HealSpot[], label: string, coalesce?: boolean) => void;
  readonly onChangePendingRemoveStrokes?: (next: readonly InpaintStrokeInput[]) => void;
}

type DragMode =
  | { readonly kind: 'draw'; readonly spotId: string; readonly startUv: readonly [number, number] }
  | { readonly kind: 'remove_draw'; readonly strokeIdx: number }
  | {
      readonly kind: 'source';
      readonly spotId: string;
    }
  | {
      readonly kind: 'target';
      readonly spotId: string;
      readonly downUv: readonly [number, number];
      readonly origSpot: HealSpot;
    };

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Interactive Lightroom / Evoto-style on-canvas Spot Heal & Clone overlay:
 * - Zoom-calibrated circular feathered brush cursor
 * - Single-click blemish heal with smart auto-picked clean donor pin
 * - Click-and-drag continuous heal/clone stroke for stray hairs, wrinkles, and backdrop scuffs
 * - Draggable emerald donor pin & amber target pin with dashed connector line
 */
export const RetouchCanvasOverlay = memo(function RetouchCanvasOverlay({
  spots,
  selectedSpotId,
  showPins,
  healTool,
  pendingRemoveStrokes = [],
  busyInpainting = false,
  faceOverlay,
  cropTransform,
  applyCrop,
  fit,
  viewportPadPx = 0,
  viewportZoom = 1,
  viewportPan = ZERO_VIEWPORT_PAN,
  naturalWidth,
  naturalHeight,
  samplePatchStats,
  onSelectSpot,
  onChangeSpots,
  onChangePendingRemoveStrokes,
}: RetouchCanvasOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const [sizePx, setSizePx] = useState<{ w: number; h: number }>({ w: 400, h: 533 });
  const [cursorPx, setCursorPx] = useState<{ x: number; y: number } | null>(null);
  const [hoveringPin, setHoveringPin] = useState(false);
  const [hoveredSpotId, setHoveredSpotId] = useState<string | null>(null);
  const dragRef = useRef<DragMode | null>(null);
  const liveSpotsRef = useRef<readonly HealSpot[]>(spots);
  liveSpotsRef.current = spots;
  const liveRemoveStrokesRef = useRef<readonly InpaintStrokeInput[]>(pendingRemoveStrokes);
  liveRemoveStrokesRef.current = pendingRemoveStrokes;

  const qt = (((cropTransform.quarterTurns ?? 0) % 4) + 4) % 4;
  const swapAxes = qt % 2 === 1;
  const orientedW = Math.max(1, swapAxes ? naturalHeight : naturalWidth);
  const orientedH = Math.max(1, swapAxes ? naturalWidth : naturalHeight);
  const minSide = Math.min(orientedW, orientedH);

  const orientedFaceGuide = useMemo(
    () => faceOverlayToOrientedGuide(faceOverlay, cropTransform, orientedW, orientedH, true),
    [faceOverlay, cropTransform, orientedW, orientedH],
  );

  const getUvTransform = (w: number, h: number) =>
    computeViewportUvTransform(
      orientedW,
      orientedH,
      Math.max(1, w),
      Math.max(1, h),
      fit,
      applyCrop ? cropTransform.rect : undefined,
      viewportPadPx,
      viewportZoom,
      viewportPan,
    );

  const clientToPhotoUv = (clientX: number, clientY: number): readonly [number, number] | null => {
    const el = overlayRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    if (Math.abs(rect.width - sizePx.w) > 1 || Math.abs(rect.height - sizePx.h) > 1) {
      setSizePx({ w: rect.width, h: rect.height });
    }
    const nx = (clientX - rect.left) / rect.width;
    const ny = (clientY - rect.top) / rect.height;
    const uv = getUvTransform(rect.width, rect.height);
    return [nx * uv.scaleX + uv.offsetX, ny * uv.scaleY + uv.offsetY] as const;
  };

  const el = overlayRef.current;
  const boxW = el?.clientWidth || sizePx.w;
  const boxH = el?.clientHeight || sizePx.h;
  const uvT = getUvTransform(boxW, boxH);
  const imgPixelW = boxW / Math.max(1e-4, uvT.scaleX);
  const imgPixelH = boxH / Math.max(1e-4, uvT.scaleY);

  const uvToStagePx = (u: number, v: number): { readonly x: number; readonly y: number } => ({
    x: ((u - uvT.offsetX) / Math.max(1e-5, uvT.scaleX)) * boxW,
    y: ((v - uvT.offsetY) / Math.max(1e-5, uvT.scaleY)) * boxH,
  });

  const radiusNormToPx = (radiusNorm: number): number =>
    Math.max(3, radiusNorm * Math.min(imgPixelW, imgPixelH));

  const cursorRadiusPx = radiusNormToPx(healTool.radius);
  const cursorInnerPx = Math.max(1.5, cursorRadiusPx * (1 - healTool.feather * 0.85));

  const handleSurfacePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Let middle-click / right-click / Shift-pan bubble naturally to stage panning
    if (e.button !== 0 || e.altKey || e.shiftKey) return;
    e.stopPropagation();
    const uvPt = clientToPhotoUv(e.clientX, e.clientY);
    if (!uvPt) return;

    // Only place spots/strokes inside the photo frame
    if (uvPt[0] < 0 || uvPt[0] > 1 || uvPt[1] < 0 || uvPt[1] > 1) return;

    if (healTool.mode === 'remove') {
      if (busyInpainting || !onChangePendingRemoveStrokes) return;
      const newStroke: InpaintStrokeInput = {
        radius: Math.max(0.004, healTool.radius),
        points: [uvPt],
      };
      const next = [...liveRemoveStrokesRef.current, newStroke];
      liveRemoveStrokesRef.current = next;
      dragRef.current = { kind: 'remove_draw', strokeIdx: next.length - 1 };
      onChangePendingRemoveStrokes(next);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const [srcU, srcV] = pickAutoHealSourceUv(
      uvPt[0],
      uvPt[1],
      healTool.radius,
      orientedW,
      orientedH,
      orientedFaceGuide,
      samplePatchStats,
    );
    const created = createHealSpot(
      healTool.mode,
      uvPt[0],
      uvPt[1],
      srcU,
      srcV,
      healTool.radius,
      healTool.feather,
      healTool.opacity,
      [uvPt],
    );
    const next = [...liveSpotsRef.current, created];
    liveSpotsRef.current = next;
    dragRef.current = { kind: 'draw', spotId: created.id, startUv: uvPt };
    onSelectSpot(created.id);
    onChangeSpots(next, healTool.mode === 'clone' ? 'Clone Spot' : 'Spot Heal', true);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handleSurfacePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setCursorPx({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    if (Math.abs(rect.width - sizePx.w) > 1 || Math.abs(rect.height - sizePx.h) > 1) {
      setSizePx({ w: rect.width, h: rect.height });
    }

    const drag = dragRef.current;
    if (!drag) return;
    const uvPt = clientToPhotoUv(e.clientX, e.clientY);
    if (!uvPt) return;

    if (drag.kind === 'remove_draw') {
      const target = liveRemoveStrokesRef.current[drag.strokeIdx];
      if (!target || !onChangePendingRemoveStrokes) return;
      const prev = target.points[target.points.length - 1]!;
      const distNorm = Math.hypot(
        ((uvPt[0] - prev[0]) * orientedW) / minSide,
        ((uvPt[1] - prev[1]) * orientedH) / minSide,
      );
      const minStep = Math.max(0.003, target.radius * 0.25);
      if (distNorm < minStep) return;
      const clampedPt: readonly [number, number] = [clamp(uvPt[0], 0, 1), clamp(uvPt[1], 0, 1)];
      const next = liveRemoveStrokesRef.current.map((s, idx) =>
        idx === drag.strokeIdx ? { ...s, points: [...s.points, clampedPt] } : s,
      );
      liveRemoveStrokesRef.current = next;
      onChangePendingRemoveStrokes(next);
      return;
    }

    if (drag.kind === 'draw') {
      const targetSpot = liveSpotsRef.current.find((s) => s.id === drag.spotId);
      if (!targetSpot) return;
      const pts = targetSpot.points ?? [[targetSpot.targetX, targetSpot.targetY] as const];
      const prev = pts[pts.length - 1]!;
      const distNorm = Math.hypot(
        ((uvPt[0] - prev[0]) * orientedW) / minSide,
        ((uvPt[1] - prev[1]) * orientedH) / minSide,
      );
      const minStep = Math.max(0.0025, targetSpot.radius * 0.28);
      if (distNorm < minStep) return;

      const updatedPts = [...pts, uvPt];
      const next = liveSpotsRef.current.map((s) =>
        s.id === drag.spotId ? { ...s, points: updatedPts } : s,
      );
      liveSpotsRef.current = next;
      onChangeSpots(next, targetSpot.mode === 'clone' ? 'Clone Stroke' : 'Heal Stroke', true);
    } else if (drag.kind === 'source') {
      const next = liveSpotsRef.current.map((s) =>
        s.id === drag.spotId
          ? { ...s, sourceX: clamp(uvPt[0], 0, 1), sourceY: clamp(uvPt[1], 0, 1) }
          : s,
      );
      liveSpotsRef.current = next;
      onChangeSpots(next, 'Move Heal Source', true);
    } else if (drag.kind === 'target') {
      const du = uvPt[0] - drag.downUv[0];
      const dv = uvPt[1] - drag.downUv[1];
      const orig = drag.origSpot;
      const movedPts = orig.points?.map(
        (pt) => [clamp(pt[0] + du, 0, 1), clamp(pt[1] + dv, 0, 1)] as const,
      );
      const next = liveSpotsRef.current.map((s) =>
        s.id === drag.spotId
          ? {
              ...s,
              targetX: clamp(orig.targetX + du, 0, 1),
              targetY: clamp(orig.targetY + dv, 0, 1),
              ...(movedPts ? { points: movedPts } : {}),
            }
          : s,
      );
      liveSpotsRef.current = next;
      onChangeSpots(next, 'Move Heal Target', true);
    }
  };

  const handleSurfacePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    const drag = dragRef.current;
    if (!drag) return;
    dragRef.current = null;
    if (drag.kind === 'remove_draw') {
      return;
    }
    const finalLabel =
      drag.kind === 'source'
        ? 'Move Heal Source'
        : drag.kind === 'target'
          ? 'Move Heal Target'
          : healTool.mode === 'clone'
            ? 'Clone Spot'
            : 'Spot Heal';
    onChangeSpots(liveSpotsRef.current, finalLabel, false);
  };

  return (
    <div
      ref={overlayRef}
      onPointerDown={handleSurfacePointerDown}
      onPointerMove={handleSurfacePointerMove}
      onPointerUp={handleSurfacePointerUp}
      onPointerLeave={() => {
        setCursorPx(null);
        setHoveringPin(false);
      }}
      className={`absolute inset-0 z-20 touch-none ${hoveringPin ? 'cursor-grab' : 'cursor-none'}`}
    >
      {/* Painted Generative AI Remove Mask Strokes (Phase 3) */}
      {pendingRemoveStrokes.length > 0 && (
        <svg
          className={`absolute inset-0 w-full h-full pointer-events-none overflow-visible ${
            busyInpainting ? 'animate-pulse' : ''
          }`}
        >
          {pendingRemoveStrokes.map((stroke, idx) => {
            const rPx = radiusNormToPx(stroke.radius);
            if (stroke.points.length === 1) {
              const p = uvToStagePx(stroke.points[0]![0], stroke.points[0]![1]);
              return (
                <circle
                  key={`rem-${idx}`}
                  cx={p.x}
                  cy={p.y}
                  r={rPx}
                  fill="rgba(244, 63, 94, 0.36)"
                  stroke="#FB7185"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
              );
            }
            const poly = stroke.points
              .map((pt) => {
                const px = uvToStagePx(pt[0], pt[1]);
                return `${px.x.toFixed(1)},${px.y.toFixed(1)}`;
              })
              .join(' ');
            return (
              <g key={`rem-${idx}`}>
                <polyline
                  points={poly}
                  fill="none"
                  stroke="rgba(244, 63, 94, 0.38)"
                  strokeWidth={Math.max(4, rPx * 2)}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <polyline
                  points={poly}
                  fill="none"
                  stroke="#FB7185"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeOpacity={0.85}
                />
              </g>
            );
          })}
        </svg>
      )}

      {/* Spot Pins, Stroke Outlines & Donor Connector Lines */}
      {showPins && healTool.mode !== 'remove' && spots.length > 0 && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible">
          {spots.map((spot) => {
            const isSelected = spot.id === selectedSpotId;
            const isHovered = spot.id === hoveredSpotId;
            const showDonor = isSelected || isHovered;
            const tgt = uvToStagePx(spot.targetX, spot.targetY);
            const src = uvToStagePx(spot.sourceX, spot.sourceY);
            const rPx = radiusNormToPx(spot.radius);
            const pts = spot.points && spot.points.length > 1 ? spot.points : null;
            const polylineStr = pts
              ? pts
                  .map((p) => {
                    const px = uvToStagePx(p[0], p[1]);
                    return `${px.x.toFixed(1)},${px.y.toFixed(1)}`;
                  })
                  .join(' ')
              : null;

            const strokeCol = isSelected
              ? '#F59E0B'
              : spot.mode === 'clone'
                ? 'rgba(56,189,248,0.85)'
                : 'rgba(255,255,255,0.75)';

            return (
              <g key={spot.id}>
                {/* Dashed connector line between target blemish and clean donor patch */}
                {showDonor && (
                  <line
                    x1={tgt.x}
                    y1={tgt.y}
                    x2={src.x}
                    y2={src.y}
                    stroke="rgba(255,255,255,0.8)"
                    strokeWidth={1.25}
                    strokeDasharray="4 3"
                  />
                )}

                {/* Stroke centerline ribbon (for stray-hair / wrinkle drag strokes) */}
                {polylineStr && (
                  <polyline
                    points={polylineStr}
                    fill="none"
                    stroke={strokeCol}
                    strokeWidth={Math.max(2, rPx * 2)}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeOpacity={isSelected ? 0.26 : 0.14}
                  />
                )}

                {/* Target circle outline */}
                <circle
                  cx={tgt.x}
                  cy={tgt.y}
                  r={rPx}
                  fill="none"
                  stroke="rgba(0,0,0,0.6)"
                  strokeWidth={2.5}
                />
                <circle
                  cx={tgt.x}
                  cy={tgt.y}
                  r={rPx}
                  fill="none"
                  stroke={strokeCol}
                  strokeWidth={isSelected ? 1.6 : 1.1}
                />

                {/* Donor circle outline */}
                {showDonor && (
                  <>
                    <circle
                      cx={src.x}
                      cy={src.y}
                      r={rPx}
                      fill="none"
                      stroke="rgba(0,0,0,0.65)"
                      strokeWidth={2.5}
                    />
                    <circle
                      cx={src.x}
                      cy={src.y}
                      r={rPx}
                      fill="none"
                      stroke="#34D399"
                      strokeWidth={1.5}
                      strokeDasharray="3 2"
                    />
                  </>
                )}
              </g>
            );
          })}
        </svg>
      )}

      {/* Interactive Draggable Target & Donor Pin Handles */}
      {showPins &&
        healTool.mode !== 'remove' &&
        spots.map((spot) => {
          const isSelected = spot.id === selectedSpotId;
          const isHovered = spot.id === hoveredSpotId;
          const tgt = uvToStagePx(spot.targetX, spot.targetY);
          const src = uvToStagePx(spot.sourceX, spot.sourceY);
          return (
            <div key={`handles-${spot.id}`}>
              {/* Target handle */}
              <button
                type="button"
                aria-label="Move heal target"
                title="Drag to reposition spot (click to select)"
                onPointerEnter={() => {
                  setHoveringPin(true);
                  setHoveredSpotId(spot.id);
                }}
                onPointerLeave={() => {
                  setHoveringPin(false);
                  setHoveredSpotId((prev) => (prev === spot.id ? null : prev));
                }}
                onPointerDown={(e) => {
                  if (e.button !== 0) return;
                  e.stopPropagation();
                  const uvPt = clientToPhotoUv(e.clientX, e.clientY);
                  if (!uvPt) return;
                  onSelectSpot(spot.id);
                  dragRef.current = {
                    kind: 'target',
                    spotId: spot.id,
                    downUv: uvPt,
                    origSpot: spot,
                  };
                  overlayRef.current?.setPointerCapture(e.pointerId);
                }}
                style={{ left: tgt.x, top: tgt.y }}
                className={`absolute -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full border shadow-md transition-transform cursor-move ${
                  isSelected
                    ? 'bg-amber-core border-white scale-110'
                    : 'bg-surface/90 border-white/80 hover:scale-110'
                }`}
              />

              {/* Donor source handle */}
              {(isSelected || isHovered) && (
                <button
                  type="button"
                  aria-label="Move heal donor source"
                  title="Drag to pick clean donor skin/backdrop area"
                  onPointerEnter={() => {
                    setHoveringPin(true);
                    setHoveredSpotId(spot.id);
                  }}
                  onPointerLeave={() => {
                    setHoveringPin(false);
                    setHoveredSpotId((prev) => (prev === spot.id ? null : prev));
                  }}
                  onPointerDown={(e) => {
                    if (e.button !== 0) return;
                    e.stopPropagation();
                    onSelectSpot(spot.id);
                    dragRef.current = { kind: 'source', spotId: spot.id };
                    overlayRef.current?.setPointerCapture(e.pointerId);
                  }}
                  style={{ left: src.x, top: src.y }}
                  className="absolute -translate-x-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-emerald-500 border-2 border-white shadow-md hover:scale-110 transition-transform cursor-grab flex items-center justify-center"
                >
                  <span className="w-1 h-1 rounded-full bg-white" />
                </button>
              )}
            </div>
          );
        })}

      {/* Live Zoom-Calibrated Brush Cursor (WYSIWYG exact painted diameter in Remove mode; feathered inner ring in Heal/Clone) */}
      {cursorPx && !hoveringPin && (
        healTool.mode === 'remove' ? (
          <div
            style={{
              left: cursorPx.x,
              top: cursorPx.y,
              width: cursorRadiusPx * 2,
              height: cursorRadiusPx * 2,
            }}
            className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none rounded-full border border-rose-400/95 bg-rose-500/18 shadow-[0_0_0_1px_rgba(0,0,0,0.7)] flex items-center justify-center"
          >
            <span className="w-1 h-1 rounded-full bg-white/90 shadow" />
          </div>
        ) : (
          <div
            style={{
              left: cursorPx.x,
              top: cursorPx.y,
              width: cursorRadiusPx * 2,
              height: cursorRadiusPx * 2,
            }}
            className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none rounded-full border border-white/85 shadow-[0_0_0_1px_rgba(0,0,0,0.65)] flex items-center justify-center"
          >
            <div
              style={{ width: cursorInnerPx * 2, height: cursorInnerPx * 2 }}
              className={`rounded-full border border-dashed flex items-center justify-center ${
                healTool.mode === 'clone' ? 'border-sky-400/90' : 'border-amber-core/90'
              }`}
            >
              <span className="w-1 h-1 rounded-full bg-white/90 shadow" />
            </div>
          </div>
        )
      )}
    </div>
  );
});
