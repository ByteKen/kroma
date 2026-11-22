export interface RectLike {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

export interface FittedFrameSize {
  readonly width: number;
  readonly height: number;
}

export interface OpeningTransform {
  readonly x: number;
  readonly y: number;
  readonly scale: number;
}

export interface PanOffset {
  readonly x: number;
  readonly y: number;
}

const DEFAULT_FALLBACK_ASPECT = 3 / 4;
const SWIPE_DISTANCE_RATIO = 0.16;
const MIN_SWIPE_DISTANCE_PX = 64;
const MAX_SWIPE_DISTANCE_PX = 140;
const SWIPE_VELOCITY_THRESHOLD_PX_PER_SEC = 420;
const RUBBER_BAND_RESISTANCE = 0.24;

/**
 * Computes the exact pixel box for an `object-contain` photograph inside the carousel stage
 * so rounded borders, shadows, and normalized `[0..100%]` YuNet face landmarks hug the real image edges.
 */
export function fitPhotoToStage(
  naturalWidth: number | null,
  naturalHeight: number | null,
  maxWidth: number,
  maxHeight: number,
): FittedFrameSize {
  const safeMaxW = Math.max(120, maxWidth);
  const safeMaxH = Math.max(120, maxHeight);
  const aspect =
    naturalWidth && naturalHeight && naturalWidth > 0 && naturalHeight > 0
      ? naturalWidth / naturalHeight
      : DEFAULT_FALLBACK_ASPECT;

  const stageAspect = safeMaxW / safeMaxH;
  if (aspect > stageAspect) {
    return {
      width: Math.round(safeMaxW),
      height: Math.round(safeMaxW / aspect),
    };
  }
  return {
    width: Math.round(safeMaxH * aspect),
    height: Math.round(safeMaxH),
  };
}

/**
 * Applies smooth iOS/Motion-style rubber-banding when dragging past the first or last slide.
 */
export function applyRubberBandProgress(rawProgress: number, maxIndex: number): number {
  const clampedMax = Math.max(0, maxIndex);
  if (rawProgress < 0) {
    return rawProgress * RUBBER_BAND_RESISTANCE;
  }
  if (rawProgress > clampedMax) {
    return clampedMax + (rawProgress - clampedMax) * RUBBER_BAND_RESISTANCE;
  }
  return rawProgress;
}

/**
 * Resolves the target slide index after a horizontal pointer drag or flick gesture.
 * Positive `dragOffsetPx` means dragging right (toward previous slide); negative means dragging left (next).
 */
export function resolveCarouselTargetIndex(
  currentIndex: number,
  dragOffsetPx: number,
  velocityPxPerSec: number,
  slideStridePx: number,
  itemCount: number,
): number {
  if (itemCount <= 1) return 0;
  const maxIndex = itemCount - 1;
  const distanceThreshold = Math.min(
    MAX_SWIPE_DISTANCE_PX,
    Math.max(MIN_SWIPE_DISTANCE_PX, slideStridePx * SWIPE_DISTANCE_RATIO),
  );

  let step = 0;
  if (dragOffsetPx <= -distanceThreshold || velocityPxPerSec <= -SWIPE_VELOCITY_THRESHOLD_PX_PER_SEC) {
    step = 1;
  } else if (
    dragOffsetPx >= distanceThreshold ||
    velocityPxPerSec >= SWIPE_VELOCITY_THRESHOLD_PX_PER_SEC
  ) {
    step = -1;
  }

  return Math.max(0, Math.min(maxIndex, currentIndex + step));
}

/**
 * Computes the GPU-composited initial offset & scale so the active lightbox photo blooms
 * directly out of the double-clicked grid card without distorting aspect ratios.
 */
export function computeOpeningOffset(
  originRect: RectLike | null,
  viewportWidth: number,
  viewportHeight: number,
  targetFrameWidth: number,
): OpeningTransform {
  if (!originRect || originRect.width <= 0 || viewportWidth <= 0 || viewportHeight <= 0) {
    return { x: 0, y: 12, scale: 0.92 };
  }

  const originCenterX = originRect.left + originRect.width / 2;
  const originCenterY = originRect.top + originRect.height / 2;
  const x = originCenterX - viewportWidth / 2;
  const y = originCenterY - viewportHeight / 2;
  const rawScale = targetFrameWidth > 0 ? originRect.width / targetFrameWidth : 0.4;
  const scale = Math.max(0.22, Math.min(0.92, rawScale));

  return { x, y, scale };
}

/**
 * Clamps drag-to-pan offsets when inspecting a photo at > 1x zoom so the frame never leaves the stage.
 */
export function clampZoomPan(
  panX: number,
  panY: number,
  frameWidth: number,
  frameHeight: number,
  zoom: number,
): PanOffset {
  if (zoom <= 1) return { x: 0, y: 0 };
  const maxX = Math.max(0, ((zoom - 1) * frameWidth) / 2);
  const maxY = Math.max(0, ((zoom - 1) * frameHeight) / 2);
  return {
    x: Math.max(-maxX, Math.min(maxX, panX)),
    y: Math.max(-maxY, Math.min(maxY, panY)),
  };
}
