import { memo, useMemo, useRef, useState } from 'react';
import type { CropTransformSettings } from '../../domain/develop';
import {
  computeViewportUvTransform,
  ZERO_VIEWPORT_PAN,
  type ViewportPan,
} from '../../domain/viewportZoom';
import type {
  BrushComponent,
  BrushStroke,
  LinearGradientComponent,
  MaskComponent,
  MaskLayer,
  RadialGradientComponent,
} from '../../domain/masking';

export interface BrushToolOptions {
  readonly size: number; // 0.01..0.25 (normalized to shorter image side)
  readonly feather: number; // 0..1
  readonly flow: number; // 0.05..1
  readonly erase: boolean;
}

interface MaskCanvasOverlayProps {
  readonly activeLayer: MaskLayer | null;
  readonly activeComponentId: string | null;
  readonly cropTransform: CropTransformSettings;
  readonly applyCrop: boolean;
  readonly fit: 'cover' | 'contain';
  readonly viewportPadPx?: number;
  readonly viewportZoom?: number;
  readonly viewportPan?: ViewportPan;
  readonly naturalWidth: number;
  readonly naturalHeight: number;
  readonly brushOptions: BrushToolOptions;
  readonly colorPickActive: boolean;
  readonly onUpdateComponent: (layerId: string, nextComp: MaskComponent, coalesce?: boolean) => void;
  readonly onEnsureBrushStroke: (
    layerId: string,
    mode: 'add' | 'subtract',
    stroke: BrushStroke,
    isFinal: boolean,
  ) => void;
}

type DragHandle =
  | { readonly kind: 'lin-start' }
  | { readonly kind: 'lin-end' }
  | { readonly kind: 'lin-center'; readonly origStart: [number, number]; readonly origEnd: [number, number]; readonly downUv: [number, number] }
  | { readonly kind: 'rad-center'; readonly origCenter: [number, number]; readonly downUv: [number, number] }
  | { readonly kind: 'rad-rx' }
  | { readonly kind: 'rad-ry' }
  | { readonly kind: 'rad-rot' };

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Interactive Lightroom-style on-canvas masking overlay for:
 * - Linear Gradient draggable Start / Center / End pins & guide lines
 * - Radial Gradient draggable Center / Radius-X / Radius-Y / Rotation handles & feather ellipse
 * - Live Freehand Brush cursor + smooth Add/Erase stroke painting
 */
export const MaskCanvasOverlay = memo(function MaskCanvasOverlay({
  activeLayer,
  activeComponentId,
  cropTransform,
  applyCrop,
  fit,
  viewportPadPx = 0,
  viewportZoom = 1,
  viewportPan = ZERO_VIEWPORT_PAN,
  naturalWidth,
  naturalHeight,
  brushOptions,
  colorPickActive,
  onUpdateComponent,
  onEnsureBrushStroke,
}: MaskCanvasOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const [sizePx, setSizePx] = useState<{ w: number; h: number }>({ w: 400, h: 533 });
  const [drag, setDrag] = useState<DragHandle | null>(null);
  const [cursorPx, setCursorPx] = useState<{ x: number; y: number } | null>(null);
  const activeStrokeRef = useRef<Array<[number, number]> | null>(null);
  const lastSampledRef = useRef<[number, number] | null>(null);

  const qt = ((cropTransform.quarterTurns ?? 0) % 4 + 4) % 4;
  const swapAxes = qt % 2 === 1;
  const orientedW = Math.max(1, swapAxes ? naturalHeight : naturalWidth);
  const orientedH = Math.max(1, swapAxes ? naturalWidth : naturalHeight);

  const activeComponent = useMemo<MaskComponent | null>(() => {
    if (!activeLayer || activeLayer.components.length === 0) return null;
    return (
      activeLayer.components.find((c) => c.id === activeComponentId) ??
      activeLayer.components[activeLayer.components.length - 1] ??
      null
    );
  }, [activeLayer, activeComponentId]);

  const isBrushActive =
    !colorPickActive &&
    activeLayer !== null &&
    activeComponent !== null &&
    activeComponent.type === 'brush';

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

  const clientToPhotoUv = (clientX: number, clientY: number): [number, number] | null => {
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
    return [nx * uv.scaleX + uv.offsetX, ny * uv.scaleY + uv.offsetY];
  };

  const photoUvToPercent = (u: number, v: number): { xPct: number; yPct: number } => {
    const el = overlayRef.current;
    const w = el?.clientWidth || sizePx.w;
    const h = el?.clientHeight || sizePx.h;
    const uv = getUvTransform(w, h);
    const nx = (u - uv.offsetX) / Math.max(1e-5, uv.scaleX);
    const ny = (v - uv.offsetY) / Math.max(1e-5, uv.scaleY);
    return { xPct: nx * 100, yPct: ny * 100 };
  };

  if (!activeLayer || !activeComponent || colorPickActive) {
    return null;
  }

  // ── 1. Freehand Brush Painting Surface ──
  if (isBrushActive) {
    const el = overlayRef.current;
    const boxW = el?.clientWidth || sizePx.w;
    const boxH = el?.clientHeight || sizePx.h;
    const uv = getUvTransform(boxW, boxH);
    const imgPixelW = boxW / Math.max(1e-4, uv.scaleX);
    const imgPixelH = boxH / Math.max(1e-4, uv.scaleY);
    const brushRadiusPx = Math.max(4, brushOptions.size * Math.min(imgPixelW, imgPixelH));
    const innerRadiusPx = Math.max(2, brushRadiusPx * (1 - brushOptions.feather));

    const emitStroke = (pts: Array<[number, number]>, isFinal: boolean) => {
      const stroke: BrushStroke = {
        size: brushOptions.size,
        feather: brushOptions.feather,
        flow: brushOptions.flow,
        points: pts.map((p) => [p[0], p[1]] as const),
      };
      onEnsureBrushStroke(
        activeLayer.id,
        brushOptions.erase ? 'subtract' : 'add',
        stroke,
        isFinal,
      );
    };

    return (
      <div
        ref={overlayRef}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          e.stopPropagation();
          e.currentTarget.setPointerCapture(e.pointerId);
          const uvPt = clientToPhotoUv(e.clientX, e.clientY);
          if (!uvPt) return;
          activeStrokeRef.current = [uvPt];
          lastSampledRef.current = uvPt;
          emitStroke([uvPt], false);
        }}
        onPointerMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          setCursorPx({ x: e.clientX - rect.left, y: e.clientY - rect.top });
          if (Math.abs(rect.width - sizePx.w) > 1 || Math.abs(rect.height - sizePx.h) > 1) {
            setSizePx({ w: rect.width, h: rect.height });
          }
          if (!activeStrokeRef.current) return;
          const uvPt = clientToPhotoUv(e.clientX, e.clientY);
          if (!uvPt) return;
          const prev = lastSampledRef.current;
          const minStep = Math.max(0.004, brushOptions.size * 0.14);
          if (prev && Math.hypot(uvPt[0] - prev[0], uvPt[1] - prev[1]) < minStep) {
            return;
          }
          activeStrokeRef.current.push(uvPt);
          lastSampledRef.current = uvPt;
          emitStroke(activeStrokeRef.current, false);
        }}
        onPointerUp={(e) => {
          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId);
          }
          if (activeStrokeRef.current) {
            emitStroke(activeStrokeRef.current, true);
            activeStrokeRef.current = null;
            lastSampledRef.current = null;
          }
        }}
        onPointerLeave={() => {
          setCursorPx(null);
        }}
        className="absolute inset-0 z-20 cursor-none touch-none"
      >
        {cursorPx && (
          <div
            style={{
              left: cursorPx.x,
              top: cursorPx.y,
              width: brushRadiusPx * 2,
              height: brushRadiusPx * 2,
            }}
            className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none rounded-full border border-white/85 shadow-[0_0_0_1px_rgba(0,0,0,0.65)] flex items-center justify-center"
          >
            <div
              style={{ width: innerRadiusPx * 2, height: innerRadiusPx * 2 }}
              className="rounded-full border border-dashed border-amber-core/90 flex items-center justify-center"
            >
              <span className="text-[10px] font-bold text-white drop-shadow-[0_1px_2px_rgba(0,0,0,0.9)] leading-none">
                {brushOptions.erase ? '−' : '+'}
              </span>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ── 2. Linear Gradient Interactive Handles ──
  if (activeComponent.type === 'linear_gradient') {
    const lin: LinearGradientComponent = activeComponent;
    const pStart = photoUvToPercent(lin.startX, lin.startY);
    const pEnd = photoUvToPercent(lin.endX, lin.endY);
    const midX = (lin.startX + lin.endX) * 0.5;
    const midY = (lin.startY + lin.endY) * 0.5;
    const pMid = photoUvToPercent(midX, midY);

    const handleMove = (e: React.PointerEvent<HTMLDivElement>) => {
      if (!drag) return;
      const pt = clientToPhotoUv(e.clientX, e.clientY);
      if (!pt) return;
      if (drag.kind === 'lin-start') {
        onUpdateComponent(
          activeLayer.id,
          { ...lin, startX: clamp(pt[0], -0.3, 1.3), startY: clamp(pt[1], -0.3, 1.3) },
          true,
        );
      } else if (drag.kind === 'lin-end') {
        onUpdateComponent(
          activeLayer.id,
          { ...lin, endX: clamp(pt[0], -0.3, 1.3), endY: clamp(pt[1], -0.3, 1.3) },
          true,
        );
      } else if (drag.kind === 'lin-center') {
        const dx = pt[0] - drag.downUv[0];
        const dy = pt[1] - drag.downUv[1];
        onUpdateComponent(
          activeLayer.id,
          {
            ...lin,
            startX: clamp(drag.origStart[0] + dx, -0.3, 1.3),
            startY: clamp(drag.origStart[1] + dy, -0.3, 1.3),
            endX: clamp(drag.origEnd[0] + dx, -0.3, 1.3),
            endY: clamp(drag.origEnd[1] + dy, -0.3, 1.3),
          },
          true,
        );
      }
    };

    const handleEnd = (e: React.PointerEvent<HTMLDivElement>) => {
      if (!drag) return;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      setDrag(null);
    };

    return (
      <div
        ref={overlayRef}
        onPointerMove={handleMove}
        onPointerUp={handleEnd}
        className={`absolute inset-0 z-20 ${drag ? 'pointer-events-auto' : 'pointer-events-none'}`}
      >
        <svg className="w-full h-full overflow-visible">
          <line
            x1={`${pStart.xPct}%`}
            y1={`${pStart.yPct}%`}
            x2={`${pEnd.xPct}%`}
            y2={`${pEnd.yPct}%`}
            stroke="rgba(245, 158, 11, 0.9)"
            strokeWidth="1.5"
            strokeDasharray="4 3"
          />
        </svg>

        {/* Start Pin (100% Selected Side) */}
        <button
          type="button"
          title="Drag gradient start (100% mask)"
          onPointerDown={(e) => {
            e.stopPropagation();
            overlayRef.current?.setPointerCapture(e.pointerId);
            setDrag({ kind: 'lin-start' });
          }}
          style={{ left: `${pStart.xPct}%`, top: `${pStart.yPct}%` }}
          className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-amber-core border-2 border-white shadow-md cursor-move"
        />

        {/* Center Move Pin */}
        <button
          type="button"
          title="Drag to move gradient"
          onPointerDown={(e) => {
            e.stopPropagation();
            const pt = clientToPhotoUv(e.clientX, e.clientY);
            if (!pt) return;
            overlayRef.current?.setPointerCapture(e.pointerId);
            setDrag({
              kind: 'lin-center',
              origStart: [lin.startX, lin.startY],
              origEnd: [lin.endX, lin.endY],
              downUv: pt,
            });
          }}
          style={{ left: `${pMid.xPct}%`, top: `${pMid.yPct}%` }}
          className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-surface-active/90 border-2 border-amber-core text-amber-soft shadow-lg flex items-center justify-center cursor-grab"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-amber-core" />
        </button>

        {/* End Pin (0% Fade Out Side) */}
        <button
          type="button"
          title="Drag gradient end (fade out)"
          onPointerDown={(e) => {
            e.stopPropagation();
            overlayRef.current?.setPointerCapture(e.pointerId);
            setDrag({ kind: 'lin-end' });
          }}
          style={{ left: `${pEnd.xPct}%`, top: `${pEnd.yPct}%` }}
          className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-surface-lowest border-2 border-white shadow-md cursor-move"
        />
      </div>
    );
  }

  // ── 3. Radial Gradient Interactive Ellipse & Handles ──
  if (activeComponent.type === 'radial_gradient') {
    const rad: RadialGradientComponent = activeComponent;
    const radAngle = (rad.rotationDeg * Math.PI) / 180;
    const cosR = Math.cos(radAngle);
    const sinR = Math.sin(radAngle);

    const pCenter = photoUvToPercent(rad.centerX, rad.centerY);
    const rxU = rad.centerX + rad.radiusX * cosR;
    const rxV = rad.centerY + rad.radiusX * sinR;
    const ryU = rad.centerX - rad.radiusY * sinR;
    const ryV = rad.centerY + rad.radiusY * cosR;
    const rotU = rad.centerX - (rad.radiusY + 0.065) * sinR;
    const rotV = rad.centerY - (rad.radiusY + 0.065) * cosR;

    const pRx = photoUvToPercent(rxU, rxV);
    const pRy = photoUvToPercent(ryU, ryV);
    const pRot = photoUvToPercent(rotU, rotV);

    // Sample 48 ellipse perimeter points in photo space so aspect + crop transform stay exact
    const buildEllipsePath = (scale: number): string => {
      const pts: string[] = [];
      for (let i = 0; i < 48; i++) {
        const theta = (i / 48) * Math.PI * 2;
        const lx = Math.cos(theta) * rad.radiusX * scale;
        const ly = Math.sin(theta) * rad.radiusY * scale;
        const u = rad.centerX + lx * cosR - ly * sinR;
        const v = rad.centerY + lx * sinR + ly * cosR;
        const p = photoUvToPercent(u, v);
        pts.push(`${i === 0 ? 'M' : 'L'} ${p.xPct.toFixed(2)} ${p.yPct.toFixed(2)}`);
      }
      return `${pts.join(' ')} Z`;
    };

    const innerScale = Math.max(0.05, 1 - rad.feather / 100);

    const handleMove = (e: React.PointerEvent<HTMLDivElement>) => {
      if (!drag) return;
      const pt = clientToPhotoUv(e.clientX, e.clientY);
      if (!pt) return;
      if (drag.kind === 'rad-center') {
        const dx = pt[0] - drag.downUv[0];
        const dy = pt[1] - drag.downUv[1];
        onUpdateComponent(
          activeLayer.id,
          {
            ...rad,
            centerX: clamp(drag.origCenter[0] + dx, -0.25, 1.25),
            centerY: clamp(drag.origCenter[1] + dy, -0.25, 1.25),
          },
          true,
        );
      } else if (drag.kind === 'rad-rx') {
        const dx = pt[0] - rad.centerX;
        const dy = pt[1] - rad.centerY;
        const proj = Math.abs(dx * cosR + dy * sinR);
        onUpdateComponent(activeLayer.id, { ...rad, radiusX: clamp(proj, 0.03, 1.2) }, true);
      } else if (drag.kind === 'rad-ry') {
        const dx = pt[0] - rad.centerX;
        const dy = pt[1] - rad.centerY;
        const proj = Math.abs(-dx * sinR + dy * cosR);
        onUpdateComponent(activeLayer.id, { ...rad, radiusY: clamp(proj, 0.03, 1.2) }, true);
      } else if (drag.kind === 'rad-rot') {
        const dx = pt[0] - rad.centerX;
        const dy = pt[1] - rad.centerY;
        const deg = (Math.atan2(dx, -dy) * 180) / Math.PI;
        onUpdateComponent(activeLayer.id, { ...rad, rotationDeg: clamp(Math.round(deg), -180, 180) }, true);
      }
    };

    const handleEnd = (e: React.PointerEvent<HTMLDivElement>) => {
      if (!drag) return;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      setDrag(null);
    };

    return (
      <div
        ref={overlayRef}
        onPointerMove={handleMove}
        onPointerUp={handleEnd}
        className={`absolute inset-0 z-20 ${drag ? 'pointer-events-auto' : 'pointer-events-none'}`}
      >
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full overflow-visible">
          <path
            d={buildEllipsePath(1)}
            fill="none"
            stroke="rgba(245, 158, 11, 0.92)"
            strokeWidth="0.35"
          />
          <path
            d={buildEllipsePath(innerScale)}
            fill="none"
            stroke="rgba(255, 255, 255, 0.65)"
            strokeWidth="0.25"
            strokeDasharray="1.2 1.2"
          />
        </svg>

        {/* Center Pin */}
        <button
          type="button"
          title="Drag to move radial mask"
          onPointerDown={(e) => {
            e.stopPropagation();
            const pt = clientToPhotoUv(e.clientX, e.clientY);
            if (!pt) return;
            overlayRef.current?.setPointerCapture(e.pointerId);
            setDrag({ kind: 'rad-center', origCenter: [rad.centerX, rad.centerY], downUv: pt });
          }}
          style={{ left: `${pCenter.xPct}%`, top: `${pCenter.yPct}%` }}
          className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-surface-active/90 border-2 border-amber-core shadow-lg flex items-center justify-center cursor-grab"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-amber-core" />
        </button>

        {/* Radius X Handle */}
        <button
          type="button"
          title="Drag to resize horizontal radius"
          onPointerDown={(e) => {
            e.stopPropagation();
            overlayRef.current?.setPointerCapture(e.pointerId);
            setDrag({ kind: 'rad-rx' });
          }}
          style={{ left: `${pRx.xPct}%`, top: `${pRx.yPct}%` }}
          className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-white border-2 border-amber-core shadow cursor-ew-resize"
        />

        {/* Radius Y Handle */}
        <button
          type="button"
          title="Drag to resize vertical radius"
          onPointerDown={(e) => {
            e.stopPropagation();
            overlayRef.current?.setPointerCapture(e.pointerId);
            setDrag({ kind: 'rad-ry' });
          }}
          style={{ left: `${pRy.xPct}%`, top: `${pRy.yPct}%` }}
          className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-white border-2 border-amber-core shadow cursor-ns-resize"
        />

        {/* Rotation Handle */}
        <button
          type="button"
          title="Drag to rotate radial mask"
          onPointerDown={(e) => {
            e.stopPropagation();
            overlayRef.current?.setPointerCapture(e.pointerId);
            setDrag({ kind: 'rad-rot' });
          }}
          style={{ left: `${pRot.xPct}%`, top: `${pRot.yPct}%` }}
          className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-amber-core/95 border-2 border-white shadow flex items-center justify-center cursor-alias"
        >
          <span className="material-symbols-outlined text-[10px] text-amber-ink">refresh</span>
        </button>
      </div>
    );
  }

  return <div ref={overlayRef} className="hidden" />;
});

/** Type guard helper for callers checking whether a component is a Brush. */
export function isBrushComponent(c: MaskComponent | null | undefined): c is BrushComponent {
  return Boolean(c && c.type === 'brush');
}
