import type { CropRect } from './develop';
import { computeObjectFitUvTransform, type ObjectFitUvTransform } from './developShaderMath';

export interface ViewportPan {
  readonly x: number;
  readonly y: number;
}

export interface ViewportRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface ViewportUvResult extends ObjectFitUvTransform {
  /** Oriented full-frame `[minU, minV, maxU, maxV]` visible crop bounds outside which fragments render transparent. */
  readonly cropBounds: readonly [number, number, number, number];
}

export const ZERO_VIEWPORT_PAN: ViewportPan = { x: 0, y: 0 };

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function sanitizeCropRect(cropRect?: CropRect): {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
} {
  if (!cropRect) {
    return { left: 0, top: 0, width: 1, height: 1 };
  }
  const left = clamp(cropRect.left, 0, 0.95);
  const top = clamp(cropRect.top, 0, 0.95);
  const width = clamp(cropRect.width, 0.05, 1 - left);
  const height = clamp(cropRect.height, 0.05, 1 - top);
  return { left, top, width, height };
}

/**
 * Computes the centered Fit-to-Stage rectangle for a photo of `photoAspect` (`width / height`)
 * inside a full darkroom stage of `stageW × stageH` CSS pixels with breathing margin `padPx`.
 */
export function computeFitPhotoRect(
  stageW: number,
  stageH: number,
  photoAspect: number,
  padPx: number,
): ViewportRect {
  const safeStageW = Math.max(1, stageW);
  const safeStageH = Math.max(1, stageH);
  const safeAspect = Number.isFinite(photoAspect) && photoAspect > 0.02 ? photoAspect : 2 / 3;
  const safePad = Math.max(0, Math.min(padPx, Math.min(safeStageW, safeStageH) * 0.4));
  const availW = Math.max(1, safeStageW - safePad * 2);
  const availH = Math.max(1, safeStageH - safePad * 2);

  let width = availH * safeAspect;
  let height = availH;
  if (width > availW) {
    width = availW;
    height = width / safeAspect;
  }

  return {
    x: (safeStageW - width) * 0.5,
    y: (safeStageH - height) * 0.5,
    width,
    height,
  };
}

/**
 * Computes the photo's on-screen CSS pixel rectangle inside the stage at `(zoom, pan)`,
 * where `zoom = 1` matches `fitRect` and `pan = (0, 0)` centers the zoomed image in the stage.
 */
export function computeDisplayedPhotoRect(
  stageW: number,
  stageH: number,
  fitRect: ViewportRect,
  zoom: number,
  pan: ViewportPan,
): ViewportRect {
  const safeZoom = Math.max(1, Number.isFinite(zoom) ? zoom : 1);
  const width = fitRect.width * safeZoom;
  const height = fitRect.height * safeZoom;
  return {
    x: (stageW - width) * 0.5 + pan.x,
    y: (stageH - height) * 0.5 + pan.y,
    width,
    height,
  };
}

/**
 * Computes the exact `zoom` multiplier relative to `Fit` where **1 native image pixel equals
 * 1 physical display pixel** (`100%` / `1:1` loupe inspection).
 *
 * When a small image already renders at >= 100% in Fit view, returns a minimum useful loupe
 * magnification (`1.5×`) so toggling `100%` still zooms cleanly.
 */
export function computeNativeOneToOneZoom(
  nativeCropW: number,
  nativeCropH: number,
  fitRectW: number,
  fitRectH: number,
  dpr: number,
): number {
  if (nativeCropW <= 0 || nativeCropH <= 0 || fitRectW <= 0 || fitRectH <= 0) {
    return 2;
  }
  const safeDpr = clamp(Number.isFinite(dpr) ? dpr : 1, 1, 3);
  const physFitW = fitRectW * safeDpr;
  const physFitH = fitRectH * safeDpr;
  const ratio = Math.max(nativeCropW / physFitW, nativeCropH / physFitH);
  return Math.max(1.5, ratio);
}

/**
 * Clamps pan offsets for free-form 2D darkroom panning whenever zoomed in (`zoom > 1`):
 * - When an axis is narrower than the stage (e.g. a moderately zoomed portrait on a widescreen
 *   monitor), allows free horizontal/vertical sliding anywhere inside the stage bounds rather than
 *   locking movement to a single rail.
 * - When an axis exceeds the stage, allows panning cleanly to every photo edge plus `edgePadPx`.
 * - At `Fit` (`zoom <= 1`), locks cleanly to `(0, 0)`.
 */
export function clampViewportPan(
  pan: ViewportPan,
  stageW: number,
  stageH: number,
  fitRect: ViewportRect,
  zoom: number,
  edgePadPx = 0,
): ViewportPan {
  const safeZoom = Math.max(1, Number.isFinite(zoom) ? zoom : 1);
  if (safeZoom <= 1.0001) {
    return ZERO_VIEWPORT_PAN;
  }
  const safePad = Math.max(0, edgePadPx);
  const dispW = fitRect.width * safeZoom;
  const dispH = fitRect.height * safeZoom;
  const maxX =
    dispW > stageW
      ? (dispW - stageW) * 0.5 + safePad
      : Math.max(0, (stageW - dispW) * 0.5 - safePad);
  const maxY =
    dispH > stageH
      ? (dispH - stageH) * 0.5 + safePad
      : Math.max(0, (stageH - dispH) * 0.5 - safePad);
  return {
    x: clamp(pan.x, -maxX, maxX),
    y: clamp(pan.y, -maxY, maxY),
  };
}

/**
 * Computes the new clamped pan offset when zooming from `currentZoom` to `nextZoom` while keeping
 * the image point directly under `(cursorStageX, cursorStageY)` locked stationary on screen.
 */
export function zoomAroundStagePoint(
  cursorStageX: number,
  cursorStageY: number,
  currentZoom: number,
  nextZoom: number,
  currentPan: ViewportPan,
  stageW: number,
  stageH: number,
  fitRect: ViewportRect,
  edgePadPx = 0,
): ViewportPan {
  const safeCurZoom = Math.max(1, Number.isFinite(currentZoom) ? currentZoom : 1);
  const safeNextZoom = Math.max(1, Number.isFinite(nextZoom) ? nextZoom : 1);
  if (safeNextZoom <= 1.0001) {
    return ZERO_VIEWPORT_PAN;
  }

  const stageCx = stageW * 0.5;
  const stageCy = stageH * 0.5;
  const scaleRatio = safeNextZoom / safeCurZoom;
  // Photo center at currentZoom is (stageCx + pan.x, stageCy + pan.y).
  // Preserve vector from photo center to cursor scaled by `scaleRatio`:
  //   cursor - nextCenter = (cursor - curCenter) * scaleRatio
  const rawPanX = cursorStageX - stageCx - (cursorStageX - (stageCx + currentPan.x)) * scaleRatio;
  const rawPanY = cursorStageY - stageCy - (cursorStageY - (stageCy + currentPan.y)) * scaleRatio;

  return clampViewportPan(
    { x: rawPanX, y: rawPanY },
    stageW,
    stageH,
    fitRect,
    safeNextZoom,
    edgePadPx,
  );
}

/**
 * Unified UV transform mapping normalized canvas coordinates `(nx, ny) ∈ [0..1]²` into oriented
 * full-frame photo UV `(u, v) ∈ [0..1]²`, supporting both standard thumbnail `cover`/`contain`
 * rendering and full-stage darkroom GPU zoom + pan + crop bounds clipping.
 */
export function computeViewportUvTransform(
  orientedW: number,
  orientedH: number,
  canvasW: number,
  canvasH: number,
  fit: 'cover' | 'contain',
  cropRect?: CropRect,
  padPx = 0,
  zoom = 1,
  pan: ViewportPan = ZERO_VIEWPORT_PAN,
): ViewportUvResult {
  const cr = sanitizeCropRect(cropRect);
  const cropBounds: readonly [number, number, number, number] = [
    cr.left,
    cr.top,
    cr.left + cr.width,
    cr.top + cr.height,
  ];

  if (orientedW <= 0 || orientedH <= 0 || canvasW <= 0 || canvasH <= 0) {
    return { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0, cropBounds };
  }

  const safeZoom = Math.max(1, Number.isFinite(zoom) ? zoom : 1);
  const isDefaultFit =
    fit === 'cover' ||
    (padPx <= 0 && Math.abs(safeZoom - 1) <= 1e-5 && Math.abs(pan.x) <= 1e-4 && Math.abs(pan.y) <= 1e-4);

  if (isDefaultFit) {
    const base = computeObjectFitUvTransform(
      orientedW,
      orientedH,
      canvasW,
      canvasH,
      fit,
      cropRect,
    );
    return { ...base, cropBounds };
  }

  const effectiveW = orientedW * cr.width;
  const effectiveH = orientedH * cr.height;
  const croppedAspect = effectiveW / Math.max(1e-5, effectiveH);
  const fitRect = computeFitPhotoRect(canvasW, canvasH, croppedAspect, padPx);
  const disp = computeDisplayedPhotoRect(canvasW, canvasH, fitRect, safeZoom, pan);

  const safeDispW = Math.max(1e-4, disp.width);
  const safeDispH = Math.max(1e-4, disp.height);
  return {
    scaleX: (canvasW / safeDispW) * cr.width,
    scaleY: (canvasH / safeDispH) * cr.height,
    offsetX: cr.left + (-disp.x / safeDispW) * cr.width,
    offsetY: cr.top + (-disp.y / safeDispH) * cr.height,
    cropBounds,
  };
}

/**
 * Formats a concise darkroom zoom readout (`Fit`, `50%`, `100%`, `200%`, etc.) relative to the
 * photo's true `1:1` native pixel ratio.
 */
export function formatViewportZoomLabel(zoom: number, oneToOneZoom: number): string {
  if (!Number.isFinite(zoom) || zoom <= 1.01) {
    return 'Fit';
  }
  const safe1x = Math.max(1, Number.isFinite(oneToOneZoom) ? oneToOneZoom : 2);
  const pct = Math.max(10, Math.round((zoom / safe1x) * 100));
  return `${pct}%`;
}

/**
 * Returns `true` when the photo's current physical display size on screen exceeds the active
 * working texture resolution enough to benefit from upgrading to the full native resolution source.
 */
export function shouldUpgradeToNativeZoomResolution(
  dispWidthCss: number,
  dispHeightCss: number,
  dpr: number,
  activeTexLongEdge: number,
  nativeLongEdge: number,
): boolean {
  if (activeTexLongEdge <= 0 || nativeLongEdge <= 0) return false;
  // Already at (or practically at) native resolution
  if (nativeLongEdge <= activeTexLongEdge * 1.12) return false;
  const safeDpr = clamp(Number.isFinite(dpr) ? dpr : 1, 1, 3);
  const requiredPhysicalLongEdge = Math.max(dispWidthCss, dispHeightCss) * safeDpr;
  return requiredPhysicalLongEdge > activeTexLongEdge * 1.12;
}
