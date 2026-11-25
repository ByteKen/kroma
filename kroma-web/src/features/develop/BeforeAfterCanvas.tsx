import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useStudio } from '../../app/StudioContext';
import { resolveCanvasBackdropHex } from '../../domain/preferences';
import {
  createDefaultDevelopRecipe,
  type CropTransformSettings,
} from '../../domain/develop';
import {
  compressRec2020ToLinearSrgb,
  linearToSrgb,
  softClip,
} from '../../domain/developShaderMath';
import type { BrushStroke, MaskComponent, MaskLayer } from '../../domain/masking';
import {
  type FaceParseMapData,
  type HealSpot,
  type InpaintStrokeInput,
} from '../../domain/retouch';
import {
  isRawFileName,
  shouldProbeHighBitDepthBuffer,
  type DevelopWorkingBuffer,
  type WorkingBufferPrecision,
} from '../../domain/developWorkingBuffer';
import {
  clampViewportPan,
  computeDisplayedPhotoRect,
  computeFitPhotoRect,
  computeNativeOneToOneZoom,
  shouldUpgradeToNativeZoomResolution,
  zoomAroundStagePoint,
  ZERO_VIEWPORT_PAN,
  type ViewportPan,
} from '../../domain/viewportZoom';
import { detectWebGpuAvailable } from '../../domain/wgslUniformPacker';
import type { PhotoAsset } from '../../domain/photo';
import { TelemetryPill } from '../darkroom-kit';
import { CropOverlay } from './CropOverlay';
import { peekDevelopBitmap } from './developBitmapCache';
import {
  DevelopViewportCanvas,
  type ActiveViewportBackend,
} from './DevelopViewportCanvas';
import { MaskCanvasOverlay, type BrushToolOptions } from './MaskCanvasOverlay';
import { RetouchCanvasOverlay } from './RetouchCanvasOverlay';
import type { HealToolOptions } from './RetouchPanel';

export type CompareViewportMode = 'single' | 'split' | 'before_after';

const NEUTRAL_DEVELOP_RECIPE = createDefaultDevelopRecipe();
const EMPTY_HEAL_SPOTS: readonly HealSpot[] = [];
const DEFAULT_HEAL_TOOL: HealToolOptions = {
  mode: 'heal',
  radius: 0.018,
  feather: 0.55,
  opacity: 1,
};

function f16BitsToF32(h: number): number {
  const sign = (h & 0x8000) !== 0 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 31) return frac === 0 ? sign * 65504 : 0;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function orientSourceUv(
  u: number,
  v: number,
  flipH: boolean,
  flipV: boolean,
  quarterTurns: number,
): readonly [number, number] {
  const px = flipH ? 1 - u : u;
  const py = flipV ? 1 - v : v;
  const qt = ((quarterTurns % 4) + 4) % 4;
  if (qt === 1) return [py, 1 - px] as const;
  if (qt === 2) return [1 - px, 1 - py] as const;
  if (qt === 3) return [1 - py, px] as const;
  return [px, py] as const;
}
export const DEVELOP_WORKING_MAX_DIM = 1600;
const ACTIVE_RAW_SETTLE_MS = 45;
const HI_RES_DEBOUNCE_MS = 120;
const WHEEL_ZOOM_STEP = 1.25;
const EDGE_PAN_CUSHION_PX = 20;

interface BeforeAfterCanvasProps {
  readonly photo: PhotoAsset;
  readonly compareMode: CompareViewportMode;
  /** Continuous zoom multiplier relative to `Fit` (`1` = Fit-to-stage, `oneToOneZoom` = true 100% 1:1). */
  readonly zoomFactor?: number;
  readonly onChangeZoomFactor?: (nextZoom: number) => void;
  readonly onOneToOneZoomChange?: (oneToOneZoom: number) => void;
  readonly showShadowClip?: boolean;
  readonly showHighlightClip?: boolean;
  readonly cropModeActive?: boolean;
  readonly onChangeCrop?: (next: CropTransformSettings) => void;
  readonly onNaturalAspectChange?: (aspect: number) => void;
  readonly eyedropperActive?: boolean;
  readonly onSampleNeutralRgb?: (r: number, g: number, b: number) => void;
  readonly maskingActive?: boolean;
  readonly activeMaskLayer?: MaskLayer | null;
  readonly activeMaskComponentId?: string | null;
  readonly showMaskOverlay?: boolean;
  readonly brushOptions?: BrushToolOptions;
  readonly colorPickActive?: boolean;
  readonly onSampleColorRangeRgb?: (r: number, g: number, b: number) => void;
  readonly onUpdateMaskComponent?: (
    layerId: string,
    nextComp: MaskComponent,
    coalesce?: boolean,
  ) => void;
  readonly onEnsureBrushStroke?: (
    layerId: string,
    mode: 'add' | 'subtract',
    stroke: BrushStroke,
    isFinal: boolean,
  ) => void;
  readonly retouchActive?: boolean;
  readonly healTool?: HealToolOptions;
  readonly selectedSpotId?: string | null;
  readonly showSpotPins?: boolean;
  readonly faceParseMap?: FaceParseMapData | null;
  readonly pendingRemoveStrokes?: readonly InpaintStrokeInput[];
  readonly busyInpainting?: boolean;
  readonly onSelectSpot?: (spotId: string | null) => void;
  readonly onChangeHealSpots?: (
    nextSpots: readonly HealSpot[],
    label: string,
    coalesce?: boolean,
  ) => void;
  readonly onChangePendingRemoveStrokes?: (next: readonly InpaintStrokeInput[]) => void;
  readonly floatingOverlay?: ReactNode;
  readonly focusMode?: boolean;
}

interface LoadedWorkingState {
  readonly photoId: string;
  readonly maxDim: number;
  readonly buffer: DevelopWorkingBuffer;
}

export const BeforeAfterCanvas = memo(function BeforeAfterCanvas({
  photo,
  compareMode,
  zoomFactor = 1,
  onChangeZoomFactor,
  onOneToOneZoomChange,
  showShadowClip = false,
  showHighlightClip = false,
  cropModeActive = false,
  onChangeCrop,
  onNaturalAspectChange,
  eyedropperActive = false,
  onSampleNeutralRgb,
  maskingActive = false,
  activeMaskLayer = null,
  activeMaskComponentId = null,
  showMaskOverlay = false,
  brushOptions = { size: 0.065, feather: 0.55, flow: 1, erase: false },
  colorPickActive = false,
  onSampleColorRangeRgb,
  onUpdateMaskComponent,
  onEnsureBrushStroke,
  retouchActive = false,
  healTool = DEFAULT_HEAL_TOOL,
  selectedSpotId = null,
  showSpotPins = true,
  faceParseMap = null,
  pendingRemoveStrokes,
  busyInpainting = false,
  onSelectSpot,
  onChangeHealSpots,
  onChangePendingRemoveStrokes,
  floatingOverlay,
  focusMode = false,
}: BeforeAfterCanvasProps) {
  const {
    isDesktopRuntime,
    preferences,
    peekDevelopWorkingBuffer,
    loadDevelopWorkingBuffer,
    getPhotoNativeDimensions,
    setDevelopViewportStatus,
  } = useStudio();
  const workingMaxDim = preferences.studioPreviewMaxDim ?? DEVELOP_WORKING_MAX_DIM;
  const stageBackdropStyle = {
    backgroundColor: resolveCanvasBackdropHex(preferences.canvasBackdropTone),
  };

  const [splitPct, setSplitPct] = useState(50);
  const [draggingSplit, setDraggingSplit] = useState(false);
  const [pan, setPan] = useState<ViewportPan>(ZERO_VIEWPORT_PAN);
  const [panning, setPanning] = useState(false);
  const panStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number }>({ w: 2, h: 3 });
  const [nativeDimsState, setNativeDimsState] = useState<{
    readonly photoId: string;
    readonly w: number;
    readonly h: number;
  } | null>(null);
  const [stageSize, setStageSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const [gpuMaxTexDim, setGpuMaxTexDim] = useState<number>(8192);
  const [loadedWorking, setLoadedWorking] = useState<LoadedWorkingState | null>(null);
  const [loadedHiRes, setLoadedHiRes] = useState<LoadedWorkingState | null>(null);
  const [decodingRawPhotoId, setDecodingRawPhotoId] = useState<string | null>(null);
  const [upgradingHiResPhotoId, setUpgradingHiResPhotoId] = useState<string | null>(null);
  const [jpegHiResActive, setJpegHiResActive] = useState(false);

  const preferWebGpu = preferences.enableGpuAcceleration !== false;
  const [viewportBackend, setViewportBackend] = useState<ActiveViewportBackend>(() =>
    preferWebGpu && detectWebGpuAvailable() ? 'webgpu' : 'webgl2',
  );
  const handleBackendChange = useCallback((backend: ActiveViewportBackend) => {
    setViewportBackend(backend);
  }, []);
  const handleMaxTextureDim = useCallback((maxDim: number) => {
    if (maxDim > 0) {
      setGpuMaxTexDim(maxDim);
    }
  }, []);
  const handleNaturalSize = useCallback((w: number, h: number) => {
    setNaturalSize((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
  }, []);

  const requestedPrecision: WorkingBufferPrecision = 'f16';
  const stageRef = useRef<HTMLDivElement>(null);

  // Measure the full darkroom stage (edge-to-edge) so zoomed photos expand across the entire monitor
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const update = () => {
      setStageSize((prev) =>
        prev.w === el.clientWidth && prev.h === el.clientHeight
          ? prev
          : { w: Math.max(120, el.clientWidth), h: Math.max(120, el.clientHeight) },
      );
    };
    update();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [compareMode, cropModeActive, focusMode]);

  const shouldProbe = isDesktopRuntime && shouldProbeHighBitDepthBuffer(photo.fileName);
  const isRawSource = isRawFileName(photo.fileName);
  const syncCachedBuffer = shouldProbe
    ? peekDevelopWorkingBuffer(photo.id, workingMaxDim, requestedPrecision)
    : null;

  // Query true native sensor pixel dimensions (<1ms header read, cached) so `100%` (1:1) is exact
  useEffect(() => {
    if (!shouldProbe) return;
    let cancelled = false;
    void getPhotoNativeDimensions(photo.id).then((dims) => {
      if (cancelled || !dims) return;
      setNativeDimsState({ photoId: photo.id, w: dims.width, h: dims.height });
    });
    return () => {
      cancelled = true;
    };
  }, [photo.id, shouldProbe, getPhotoNativeDimensions]);

  // Reset pan & jpeg hi-res flag when returning to Fit or switching photo
  useEffect(() => {
    if (zoomFactor <= 1.001) {
      setPan(ZERO_VIEWPORT_PAN);
      setJpegHiResActive(false);
    }
  }, [zoomFactor, photo.id]);

  // Clear stale hi-res JPEG upgrade when switching photos
  useEffect(() => {
    setJpegHiResActive(false);
    setUpgradingHiResPhotoId(null);
  }, [photo.id]);

  // Load standard Fit-view 16-bit/32-bit working buffer (instant on LRU / SSD hit)
  useEffect(() => {
    if (!shouldProbe) {
      setDecodingRawPhotoId(null);
      return;
    }

    if (syncCachedBuffer) {
      setDecodingRawPhotoId((prev) => (prev === photo.id ? null : prev));
      setLoadedWorking((prev) =>
        prev?.photoId === photo.id && prev.maxDim === workingMaxDim && prev.buffer === syncCachedBuffer
          ? prev
          : { photoId: photo.id, maxDim: workingMaxDim, buffer: syncCachedBuffer },
      );
      return;
    }

    let cancelled = false;
    // Brief 45ms settle window for uncached RAWs: rapid clicks or arrow-key scrubbing skip wasted
    // IPC/demosaic work while 8-bit previews render immediately; RAM LRU hits stay 0ms above.
    const timer = window.setTimeout(() => {
      if (cancelled) return;
      if (isRawSource) {
        setDecodingRawPhotoId(photo.id);
      }

      void loadDevelopWorkingBuffer(photo.id, workingMaxDim, requestedPrecision, 'active')
        .then((buffer) => {
          if (cancelled) return;
          setDecodingRawPhotoId((prev) => (prev === photo.id ? null : prev));
          if (buffer) {
            setLoadedWorking({ photoId: photo.id, maxDim: workingMaxDim, buffer });
          }
        })
        .catch(() => {
          if (!cancelled) {
            setDecodingRawPhotoId((prev) => (prev === photo.id ? null : prev));
          }
        });
    }, ACTIVE_RAW_SETTLE_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      setDecodingRawPhotoId((prev) => (prev === photo.id ? null : prev));
    };
  }, [
    photo.id,
    shouldProbe,
    isRawSource,
    workingMaxDim,
    requestedPrecision,
    syncCachedBuffer,
    loadDevelopWorkingBuffer,
  ]);

  const baseWorkingBuffer =
    syncCachedBuffer ??
    (loadedWorking && loadedWorking.photoId === photo.id && loadedWorking.maxDim === workingMaxDim
      ? loadedWorking.buffer
      : null);

  const swapNatural = (photo.developRecipe.cropTransform.quarterTurns ?? 0) % 2 === 1;
  const effectiveZoom = cropModeActive ? 1 : Math.max(1, zoomFactor);
  const padPx = focusMode ? 16 : 24;

  // True native dimensions (sensor dimensions for RAW/TIFF via header probe; naturalWidth/Height for JPEG/PNG)
  const resolvedNativeW =
    shouldProbe && nativeDimsState?.photoId === photo.id
      ? nativeDimsState.w
      : naturalSize.w;
  const resolvedNativeH =
    shouldProbe && nativeDimsState?.photoId === photo.id
      ? nativeDimsState.h
      : naturalSize.h;

  const orientedNativeW = swapNatural ? resolvedNativeH : resolvedNativeW;
  const orientedNativeH = swapNatural ? resolvedNativeW : resolvedNativeH;
  const orientedNaturalAspect = swapNatural
    ? naturalSize.h / Math.max(1, naturalSize.w)
    : naturalSize.w / Math.max(1, naturalSize.h);

  useEffect(() => {
    onNaturalAspectChange?.(orientedNaturalAspect);
  }, [orientedNaturalAspect, onNaturalAspectChange]);

  const cropRect = photo.developRecipe.cropTransform.rect;
  const croppedAspect =
    orientedNaturalAspect * (Math.max(0.05, cropRect.width) / Math.max(0.05, cropRect.height));
  const frameAspect = cropModeActive ? orientedNaturalAspect : croppedAspect;

  const isBeforeAfter = compareMode === 'before_after' && !cropModeActive;
  const paneW = isBeforeAfter ? Math.max(120, (stageSize.w - 1) * 0.5) : Math.max(120, stageSize.w);
  const paneH = Math.max(120, stageSize.h);

  const fitRect = useMemo(
    () => computeFitPhotoRect(paneW, paneH, frameAspect, padPx),
    [paneW, paneH, frameAspect, padPx],
  );

  const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;
  const nativeCropW = orientedNativeW * (cropModeActive ? 1 : Math.max(0.05, cropRect.width));
  const nativeCropH = orientedNativeH * (cropModeActive ? 1 : Math.max(0.05, cropRect.height));
  const oneToOneZoom = useMemo(
    () => computeNativeOneToOneZoom(nativeCropW, nativeCropH, fitRect.width, fitRect.height, dpr),
    [nativeCropW, nativeCropH, fitRect.width, fitRect.height, dpr],
  );
  const maxZoom = Math.min(Math.max(4, oneToOneZoom * 3), 12);

  useEffect(() => {
    onOneToOneZoomChange?.(oneToOneZoom);
  }, [oneToOneZoom, onOneToOneZoomChange]);

  const clampedPan = useMemo(
    () => clampViewportPan(pan, paneW, paneH, fitRect, effectiveZoom, EDGE_PAN_CUSHION_PX),
    [pan, paneW, paneH, fitRect, effectiveZoom],
  );

  const displayedRect = useMemo(
    () => computeDisplayedPhotoRect(paneW, paneH, fitRect, effectiveZoom, clampedPan),
    [paneW, paneH, fitRect, effectiveZoom, clampedPan],
  );

  // Safe 4K+ hi-res ceiling keeps every WebGPU texture strictly under Dawn's 256 MB allocation
  // limit (4096px = ~89 MB srcTex + ~179 MB rgba32float workTexA; dual compare capped at 3072px).
  const hiResMaxDim = Math.min(
    gpuMaxTexDim,
    compareMode === 'single' ? 4096 : 3072,
  );

  const nativeLongEdge = Math.max(resolvedNativeW, resolvedNativeH);
  const baseTexLongEdge = baseWorkingBuffer
    ? Math.max(baseWorkingBuffer.width, baseWorkingBuffer.height)
    : workingMaxDim;
  const wantsHiRes =
    effectiveZoom > 1.02 &&
    hiResMaxDim > workingMaxDim &&
    shouldUpgradeToNativeZoomResolution(
      displayedRect.width,
      displayedRect.height,
      dpr,
      baseTexLongEdge,
      Math.min(nativeLongEdge, hiResMaxDim),
    );

  // On-demand native 1:1 upgrade for RAW / 16-bit TIFF / 16-bit PNG (debounced 120ms, zero flicker)
  const syncHiResBuffer =
    shouldProbe && wantsHiRes
      ? peekDevelopWorkingBuffer(photo.id, hiResMaxDim, requestedPrecision)
      : null;

  useEffect(() => {
    if (!shouldProbe || !wantsHiRes || !baseWorkingBuffer) {
      setUpgradingHiResPhotoId((prev) => (prev === photo.id ? null : prev));
      return;
    }
    if (syncHiResBuffer) {
      setUpgradingHiResPhotoId((prev) => (prev === photo.id ? null : prev));
      setLoadedHiRes((prev) =>
        prev?.photoId === photo.id && prev.maxDim === hiResMaxDim && prev.buffer === syncHiResBuffer
          ? prev
          : { photoId: photo.id, maxDim: hiResMaxDim, buffer: syncHiResBuffer },
      );
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(() => {
      if (cancelled) return;
      setUpgradingHiResPhotoId(photo.id);
      void loadDevelopWorkingBuffer(photo.id, hiResMaxDim, requestedPrecision, 'active')
        .then((buf) => {
          if (cancelled) return;
          setUpgradingHiResPhotoId((prev) => (prev === photo.id ? null : prev));
          if (buf) {
            setLoadedHiRes({ photoId: photo.id, maxDim: hiResMaxDim, buffer: buf });
          }
        })
        .catch(() => {
          if (!cancelled) {
            setUpgradingHiResPhotoId((prev) => (prev === photo.id ? null : prev));
          }
        });
    }, HI_RES_DEBOUNCE_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    photo.id,
    shouldProbe,
    wantsHiRes,
    baseWorkingBuffer,
    hiResMaxDim,
    requestedPrecision,
    syncHiResBuffer,
    loadDevelopWorkingBuffer,
  ]);

  // On-demand native 1:1 upgrade for 8-bit JPEG / PNG / WebP (debounced 120ms, zero flash)
  useEffect(() => {
    if (shouldProbe) return;
    if (!wantsHiRes) {
      setJpegHiResActive(false);
      return;
    }
    const timer = window.setTimeout(() => {
      setJpegHiResActive(true);
    }, HI_RES_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [shouldProbe, wantsHiRes]);

  const activeHiResBuffer =
    wantsHiRes
      ? (syncHiResBuffer ??
          (loadedHiRes && loadedHiRes.photoId === photo.id && loadedHiRes.maxDim === hiResMaxDim
            ? loadedHiRes.buffer
            : null))
      : null;

  const activeWorkingBuffer = activeHiResBuffer ?? baseWorkingBuffer;
  const isDecodingRaw = decodingRawPhotoId === photo.id && !activeWorkingBuffer;
  const isUpgradingHiRes = upgradingHiResPhotoId === photo.id && !activeHiResBuffer;
  const effectivePreviewMaxDim = !shouldProbe && jpegHiResActive ? hiResMaxDim : workingMaxDim;

  // Report live 32-bit/16-bit RAW decode & active GPU buffer telemetry to the bottom status strip
  useEffect(() => {
    if (activeWorkingBuffer) {
      const displayWidth = swapNatural ? activeWorkingBuffer.height : activeWorkingBuffer.width;
      const displayHeight = swapNatural ? activeWorkingBuffer.width : activeWorkingBuffer.height;
      setDevelopViewportStatus({
        state: 'ready',
        precision: viewportBackend === 'webgpu' ? 'f32' : activeWorkingBuffer.precision,
        isRawDemosaic: activeWorkingBuffer.isRawDemosaic,
        backend: viewportBackend,
        width: displayWidth,
        height: displayHeight,
        upgradingHiRes: isUpgradingHiRes,
      });
    } else if (isDecodingRaw) {
      setDevelopViewportStatus({
        state: 'decoding',
        precision: viewportBackend === 'webgpu' ? 'f32' : requestedPrecision,
        isRawDemosaic: true,
        backend: viewportBackend,
      });
    } else {
      setDevelopViewportStatus(null);
    }
  }, [
    activeWorkingBuffer,
    isDecodingRaw,
    isUpgradingHiRes,
    viewportBackend,
    requestedPrecision,
    swapNatural,
    setDevelopViewportStatus,
  ]);

  useEffect(() => {
    return () => {
      setDevelopViewportStatus(null);
    };
  }, [setDevelopViewportStatus]);

  const updateSplitFromClientX = (clientX: number) => {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSplitPct(Math.min(95, Math.max(5, pct)));
  };

  // Non-passive native wheel listener guarantees Ctrl+Scroll (or Cmd+Scroll) zooms the GPU
  // darkroom viewport smoothly toward the cursor without triggering WebView2 page zoom.
  const wheelStateRef = useRef({
    onChangeZoomFactor,
    cropModeActive,
    isBeforeAfter,
    effectiveZoom,
    maxZoom,
    clampedPan,
    fitRect,
    paneW,
    paneH,
  });
  wheelStateRef.current = {
    onChangeZoomFactor,
    cropModeActive,
    isBeforeAfter,
    effectiveZoom,
    maxZoom,
    clampedPan,
    fitRect,
    paneW,
    paneH,
  };

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onNativeWheel = (e: WheelEvent) => {
      const s = wheelStateRef.current;
      if (!s.onChangeZoomFactor || s.cropModeActive || (!e.ctrlKey && !e.metaKey)) return;
      e.preventDefault();
      const stageRect = el.getBoundingClientRect();
      let cursorX = e.clientX - stageRect.left;
      const cursorY = e.clientY - stageRect.top;
      if (s.isBeforeAfter && cursorX > s.paneW) {
        cursorX -= s.paneW + 1;
      }
      const rawNext =
        e.deltaY < 0
          ? Math.min(s.maxZoom, s.effectiveZoom * WHEEL_ZOOM_STEP)
          : s.effectiveZoom / WHEEL_ZOOM_STEP;
      const nextZoom = rawNext < 1.06 ? 1 : Number(rawNext.toFixed(4));
      const nextPan = zoomAroundStagePoint(
        cursorX,
        cursorY,
        s.effectiveZoom,
        nextZoom,
        s.clampedPan,
        s.paneW,
        s.paneH,
        s.fitRect,
        EDGE_PAN_CUSHION_PX,
      );
      setPan(nextPan);
      s.onChangeZoomFactor(nextZoom);
    };
    el.addEventListener('wheel', onNativeWheel, { passive: false });
    return () => el.removeEventListener('wheel', onNativeWheel);
  }, []);

  // Lightroom-style double-click toggles Fit <-> true 100% (1:1) centered right on the clicked pixel
  const handleStageDoubleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onChangeZoomFactor || cropModeActive || eyedropperActive || colorPickActive) return;
    if (retouchActive) return;
    if (maskingActive && activeMaskLayer) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const cursorX = e.clientX - rect.left;
    const cursorY = e.clientY - rect.top;
    if (effectiveZoom > 1.02) {
      setPan(ZERO_VIEWPORT_PAN);
      onChangeZoomFactor(1);
    } else {
      const targetZoom = Number(oneToOneZoom.toFixed(4));
      const nextPan = zoomAroundStagePoint(
        cursorX,
        cursorY,
        1,
        targetZoom,
        ZERO_VIEWPORT_PAN,
        rect.width,
        rect.height,
        fitRect,
        EDGE_PAN_CUSHION_PX,
      );
      setPan(nextPan);
      onChangeZoomFactor(targetZoom);
    }
  };

  const handleBeginPan = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    if (effectiveZoom <= 1.01 || eyedropperActive || colorPickActive || cropModeActive || draggingSplit)
      return;
    if (maskingActive && activeMaskLayer) return;
    if (retouchActive && !e.shiftKey) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setPanning(true);
    panStartRef.current = { x: e.clientX, y: e.clientY, panX: clampedPan.x, panY: clampedPan.y };
  };

  const handleMovePan = (e: React.PointerEvent<HTMLDivElement>) => {
    if (draggingSplit) {
      updateSplitFromClientX(e.clientX);
      return;
    }
    if (!panning || !panStartRef.current) return;
    const rawNext: ViewportPan = {
      x: panStartRef.current.panX + (e.clientX - panStartRef.current.x),
      y: panStartRef.current.panY + (e.clientY - panStartRef.current.y),
    };
    setPan(clampViewportPan(rawNext, paneW, paneH, fitRect, effectiveZoom, EDGE_PAN_CUSHION_PX));
  };

  const handleEndPan = (e: React.PointerEvent<HTMLDivElement>) => {
    setDraggingSplit(false);
    setPanning(false);
    panStartRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const toolHintBadge = colorPickActive ? (
    <div className="absolute top-2.5 left-1/2 -translate-x-1/2 z-30 pointer-events-none">
      <TelemetryPill tone="amber" dot className="bg-surface-lowest/90 backdrop-blur">
        Color Range Picker — Click any color on the photo (Esc to cancel)
      </TelemetryPill>
    </div>
  ) : eyedropperActive ? (
    <div className="absolute top-2.5 left-1/2 -translate-x-1/2 z-30 pointer-events-none">
      <TelemetryPill tone="emerald" dot className="bg-surface-lowest/90 backdrop-blur">
        WB Eyedropper — Click a neutral white/gray area (Esc to cancel)
      </TelemetryPill>
    </div>
  ) : null;

  const beforeRecipe = useMemo(
    () => ({
      ...NEUTRAL_DEVELOP_RECIPE,
      cropTransform: photo.developRecipe.cropTransform,
    }),
    [photo.developRecipe.cropTransform],
  );

  const developSrc = photo.studioSourceUrl ?? photo.imageUrl;
  const retouchSpots = photo.developRecipe.retouch?.healSpots ?? EMPTY_HEAL_SPOTS;

  // Lightweight cached pixel grid for 8-bit previews so auto-donor selection scores real skin patches in <0.2ms
  const patchGridCacheRef = useRef<{
    readonly key: string;
    readonly w: number;
    readonly h: number;
    readonly data: Uint8ClampedArray;
  } | null>(null);

  const samplePatchStats = useCallback(
    (u: number, v: number, radiusNorm: number) => {
      const ct = photo.developRecipe.cropTransform;
      const qt = ((ct.quarterTurns ?? 0) % 4 + 4) % 4;
      const swap = qt % 2 === 1;
      const flipH = Boolean(ct.flipHorizontal);
      const flipV = Boolean(ct.flipVertical);

      const wb = activeWorkingBuffer;
      if (wb && (wb.float32Rgba || wb.halfFloatRgba)) {
        const oriW = swap ? wb.height : wb.width;
        const oriH = swap ? wb.width : wb.height;
        const minSide = Math.min(oriW, oriH);
        const rx = (radiusNorm * minSide) / Math.max(1, oriW);
        const ry = (radiusNorm * minSide) / Math.max(1, oriH);
        let sumR = 0;
        let sumG = 0;
        let sumB = 0;
        let sumL = 0;
        let sumL2 = 0;
        let n = 0;
        for (let gy = -2; gy <= 2; gy++) {
          const sv = Math.min(1, Math.max(0, v + (gy * 0.5) * ry));
          for (let gx = -2; gx <= 2; gx++) {
            const su = Math.min(1, Math.max(0, u + (gx * 0.5) * rx));
            const [rawU, rawV] = orientSourceUv(su, sv, flipH, flipV, qt);
            const px = Math.min(wb.width - 1, Math.max(0, Math.round(rawU * (wb.width - 1))));
            const py = Math.min(wb.height - 1, Math.max(0, Math.round(rawV * (wb.height - 1))));
            const idx = (py * wb.width + px) * 4;
            let r = wb.float32Rgba
              ? (wb.float32Rgba[idx] ?? 0)
              : f16BitsToF32(wb.halfFloatRgba![idx] ?? 0);
            let g = wb.float32Rgba
              ? (wb.float32Rgba[idx + 1] ?? 0)
              : f16BitsToF32(wb.halfFloatRgba![idx + 1] ?? 0);
            let b = wb.float32Rgba
              ? (wb.float32Rgba[idx + 2] ?? 0)
              : f16BitsToF32(wb.halfFloatRgba![idx + 2] ?? 0);
            if (wb.isLinear) {
              const lin709 = wb.isWideGamut
                ? compressRec2020ToLinearSrgb([r, g, b])
                : ([r, g, b] as [number, number, number]);
              r = linearToSrgb(softClip(lin709[0]));
              g = linearToSrgb(softClip(lin709[1]));
              b = linearToSrgb(softClip(lin709[2]));
            }
            const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sumR += r;
            sumG += g;
            sumB += b;
            sumL += l;
            sumL2 += l * l;
            n += 1;
          }
        }
        if (n === 0) return null;
        const meanL = sumL / n;
        return {
          meanR: sumR / n,
          meanG: sumG / n,
          meanB: sumB / n,
          variance: Math.max(0, sumL2 / n - meanL * meanL),
        };
      }

      const cachedBitmap = peekDevelopBitmap(developSrc, effectivePreviewMaxDim);
      if (!cachedBitmap || cachedBitmap.width <= 0 || cachedBitmap.height <= 0) return null;

      const cacheKey = `${developSrc}:${cachedBitmap.width}x${cachedBitmap.height}`;
      let grid = patchGridCacheRef.current;
      if (!grid || grid.key !== cacheKey) {
        try {
          const maxGrid = 192;
          const scale = Math.min(1, maxGrid / Math.max(cachedBitmap.width, cachedBitmap.height));
          const gw = Math.max(16, Math.round(cachedBitmap.width * scale));
          const gh = Math.max(16, Math.round(cachedBitmap.height * scale));
          const c = document.createElement('canvas');
          c.width = gw;
          c.height = gh;
          const ctx = c.getContext('2d');
          if (!ctx) return null;
          ctx.drawImage(cachedBitmap.bitmap, 0, 0, gw, gh);
          const data = ctx.getImageData(0, 0, gw, gh).data;
          grid = { key: cacheKey, w: gw, h: gh, data };
          patchGridCacheRef.current = grid;
        } catch {
          return null;
        }
      }

      const oriW = swap ? grid.h : grid.w;
      const oriH = swap ? grid.w : grid.h;
      const minSide = Math.min(oriW, oriH);
      const rx = (radiusNorm * minSide) / Math.max(1, oriW);
      const ry = (radiusNorm * minSide) / Math.max(1, oriH);
      let sumR = 0;
      let sumG = 0;
      let sumB = 0;
      let sumL = 0;
      let sumL2 = 0;
      let n = 0;
      for (let gy = -2; gy <= 2; gy++) {
        const sv = Math.min(1, Math.max(0, v + (gy * 0.5) * ry));
        for (let gx = -2; gx <= 2; gx++) {
          const su = Math.min(1, Math.max(0, u + (gx * 0.5) * rx));
          const [rawU, rawV] = orientSourceUv(su, sv, flipH, flipV, qt);
          const px = Math.min(grid.w - 1, Math.max(0, Math.round(rawU * (grid.w - 1))));
          const py = Math.min(grid.h - 1, Math.max(0, Math.round(rawV * (grid.h - 1))));
          const idx = (py * grid.w + px) * 4;
          const r = (grid.data[idx] ?? 0) / 255;
          const g = (grid.data[idx + 1] ?? 0) / 255;
          const b = (grid.data[idx + 2] ?? 0) / 255;
          const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          sumR += r;
          sumG += g;
          sumB += b;
          sumL += l;
          sumL2 += l * l;
          n += 1;
        }
      }
      if (n === 0) return null;
      const meanL = sumL / n;
      return {
        meanR: sumR / n,
        meanG: sumG / n,
        meanB: sumB / n,
        variance: Math.max(0, sumL2 / n - meanL * meanL),
      };
    },
    [photo.developRecipe.cropTransform, activeWorkingBuffer, developSrc, effectivePreviewMaxDim],
  );

  const panCursorClass =
    effectiveZoom > 1.01 && !eyedropperActive && !colorPickActive && !cropModeActive && (!retouchActive || panning)
      ? panning
        ? 'cursor-grabbing'
        : 'cursor-grab'
      : '';

  // Photo perimeter frame overlay: clean darkroom border at Fit; hides once zoomed past stage edges
  // so Chromium never rasterizes a massive offscreen DOM shadow layer during high-zoom panning.
  const photoCoversEntirePane =
    displayedRect.x <= 1 &&
    displayedRect.y <= 1 &&
    displayedRect.x + displayedRect.width >= paneW - 1 &&
    displayedRect.y + displayedRect.height >= paneH - 1;
  const showFrameOutline = !photoCoversEntirePane;
  const frameOutlineStyle = {
    left: `${Math.round(displayedRect.x)}px`,
    top: `${Math.round(displayedRect.y)}px`,
    width: `${Math.round(displayedRect.width)}px`,
    height: `${Math.round(displayedRect.height)}px`,
  };
  const frameShadowClass = effectiveZoom <= 1.05 ? 'shadow-[0_10px_36px_rgba(0,0,0,0.62)]' : '';

  if (isBeforeAfter) {
    return (
      <div
        ref={stageRef}
        style={stageBackdropStyle}
        className="flex-1 min-h-0 flex overflow-hidden relative select-none transition-colors duration-200"
      >
        {toolHintBadge}
        {floatingOverlay}

        {/* Left Full-Height Pane: ORIGINAL */}
        <div
          onDoubleClick={handleStageDoubleClick}
          onPointerDown={handleBeginPan}
          onPointerMove={handleMovePan}
          onPointerUp={handleEndPan}
          onPointerLeave={handleEndPan}
          className={`flex-1 min-w-0 h-full relative overflow-hidden border-r border-hairline ${panCursorClass}`}
        >
          <div className="absolute inset-0">
            <DevelopViewportCanvas
              src={developSrc}
              alt={`${photo.fileName} original`}
              recipe={beforeRecipe}
              workingBuffer={activeWorkingBuffer}
              previewMaxDim={effectivePreviewMaxDim}
              preferWebGpu={preferWebGpu}
              fit="contain"
              viewportPadPx={padPx}
              viewportZoom={effectiveZoom}
              viewportPan={clampedPan}
            />
          </div>
          {showFrameOutline && (
            <div
              style={frameOutlineStyle}
              className={`absolute pointer-events-none border border-hairline/80 ${frameShadowClass}`}
            />
          )}
          <div className="absolute top-3 left-3 z-20 pointer-events-none">
            <TelemetryPill tone="neutral" className="bg-surface-lowest/85 backdrop-blur">
              ORIGINAL
            </TelemetryPill>
          </div>
        </div>

        {/* Right Full-Height Pane: EDITED */}
        <div
          onDoubleClick={handleStageDoubleClick}
          onPointerDown={handleBeginPan}
          onPointerMove={handleMovePan}
          onPointerUp={handleEndPan}
          onPointerLeave={handleEndPan}
          className={`flex-1 min-w-0 h-full relative overflow-hidden ${panCursorClass}`}
        >
          <div className="absolute inset-0">
            <DevelopViewportCanvas
              src={developSrc}
              alt={`${photo.fileName} edited`}
              recipe={photo.developRecipe}
              faceOverlay={photo.faceOverlay}
              faceParseMap={faceParseMap}
              workingBuffer={activeWorkingBuffer}
              previewMaxDim={effectivePreviewMaxDim}
              preferWebGpu={preferWebGpu}
              onBackendChange={handleBackendChange}
              onMaxTextureDim={handleMaxTextureDim}
              fit="contain"
              viewportPadPx={padPx}
              viewportZoom={effectiveZoom}
              viewportPan={clampedPan}
              showShadowClip={showShadowClip}
              showHighlightClip={showHighlightClip}
              eyedropperActive={eyedropperActive}
              onSampleNeutralRgb={onSampleNeutralRgb}
              activeMaskId={activeMaskLayer?.id ?? null}
              showMaskOverlay={showMaskOverlay && maskingActive}
              colorPickActive={colorPickActive}
              onSampleColorRangeRgb={onSampleColorRangeRgb}
              onNaturalSize={handleNaturalSize}
            />
            {retouchActive &&
              !cropModeActive &&
              !eyedropperActive &&
              onSelectSpot &&
              onChangeHealSpots && (
                <RetouchCanvasOverlay
                  spots={retouchSpots}
                  selectedSpotId={selectedSpotId}
                  showPins={showSpotPins}
                  healTool={healTool}
                  pendingRemoveStrokes={pendingRemoveStrokes}
                  busyInpainting={busyInpainting}
                  faceOverlay={photo.faceOverlay}
                  cropTransform={photo.developRecipe.cropTransform}
                  applyCrop={!cropModeActive}
                  fit="contain"
                  viewportPadPx={padPx}
                  viewportZoom={effectiveZoom}
                  viewportPan={clampedPan}
                  naturalWidth={naturalSize.w}
                  naturalHeight={naturalSize.h}
                  samplePatchStats={samplePatchStats}
                  onSelectSpot={onSelectSpot}
                  onChangeSpots={onChangeHealSpots}
                  onChangePendingRemoveStrokes={onChangePendingRemoveStrokes}
                />
              )}
          </div>
          {showFrameOutline && (
            <div
              style={frameOutlineStyle}
              className={`absolute pointer-events-none border border-amber-core/45 ${frameShadowClass}`}
            />
          )}
          <div className="absolute top-3 left-3 z-20 pointer-events-none">
            <TelemetryPill tone="amber" className="bg-surface-lowest/85 backdrop-blur">
              EDITED
            </TelemetryPill>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={stageRef}
      onDoubleClick={handleStageDoubleClick}
      onPointerDown={handleBeginPan}
      onPointerMove={handleMovePan}
      onPointerUp={handleEndPan}
      onPointerLeave={handleEndPan}
      style={stageBackdropStyle}
      className={`flex-1 min-h-0 overflow-hidden relative select-none transition-colors duration-200 ${panCursorClass}`}
    >
      {toolHintBadge}
      {floatingOverlay}

      {/* Full-Stage Unified 32-Bit Wide-Gamut WebGPU / WebGL2 Canvas (Zero CSS scale digital zoom!) */}
      <div className="absolute inset-0">
        <DevelopViewportCanvas
          src={developSrc}
          alt={`${photo.fileName} graded`}
          recipe={photo.developRecipe}
          faceOverlay={photo.faceOverlay}
          faceParseMap={faceParseMap}
          workingBuffer={activeWorkingBuffer}
          previewMaxDim={effectivePreviewMaxDim}
          preferWebGpu={preferWebGpu}
          onBackendChange={handleBackendChange}
          onMaxTextureDim={handleMaxTextureDim}
          fit="contain"
          viewportPadPx={padPx}
          viewportZoom={effectiveZoom}
          viewportPan={clampedPan}
          applyCrop={!cropModeActive}
          showShadowClip={showShadowClip}
          showHighlightClip={showHighlightClip}
          eyedropperActive={eyedropperActive}
          onSampleNeutralRgb={onSampleNeutralRgb}
          activeMaskId={activeMaskLayer?.id ?? null}
          showMaskOverlay={showMaskOverlay && maskingActive}
          colorPickActive={colorPickActive}
          onSampleColorRangeRgb={onSampleColorRangeRgb}
          onNaturalSize={handleNaturalSize}
        />

        {/* Interactive Spot Heal & Clone Pin/Stroke Overlay aligned 1:1 across the full stage */}
        {retouchActive &&
          !cropModeActive &&
          !eyedropperActive &&
          onSelectSpot &&
          onChangeHealSpots && (
            <RetouchCanvasOverlay
              spots={retouchSpots}
              selectedSpotId={selectedSpotId}
              showPins={showSpotPins}
              healTool={healTool}
              pendingRemoveStrokes={pendingRemoveStrokes}
              busyInpainting={busyInpainting}
              faceOverlay={photo.faceOverlay}
              cropTransform={photo.developRecipe.cropTransform}
              applyCrop={!cropModeActive}
              fit="contain"
              viewportPadPx={padPx}
              viewportZoom={effectiveZoom}
              viewportPan={clampedPan}
              naturalWidth={naturalSize.w}
              naturalHeight={naturalSize.h}
              samplePatchStats={samplePatchStats}
              onSelectSpot={onSelectSpot}
              onChangeSpots={onChangeHealSpots}
              onChangePendingRemoveStrokes={onChangePendingRemoveStrokes}
            />
          )}

        {/* Interactive Local Mask Handles & Freehand Brush Painter aligned 1:1 across the full stage */}
        {maskingActive &&
          !cropModeActive &&
          !eyedropperActive &&
          onUpdateMaskComponent &&
          onEnsureBrushStroke && (
            <MaskCanvasOverlay
              activeLayer={activeMaskLayer}
              activeComponentId={activeMaskComponentId}
              cropTransform={photo.developRecipe.cropTransform}
              applyCrop={!cropModeActive}
              fit="contain"
              viewportPadPx={padPx}
              viewportZoom={effectiveZoom}
              viewportPan={clampedPan}
              naturalWidth={naturalSize.w}
              naturalHeight={naturalSize.h}
              brushOptions={brushOptions}
              colorPickActive={colorPickActive}
              onUpdateComponent={onUpdateMaskComponent}
              onEnsureBrushStroke={onEnsureBrushStroke}
            />
          )}
      </div>

      {/* Subtle Darkroom Frame Border hugging displayed photo perimeter (auto-hides once zoomed edge-to-edge) */}
      {!cropModeActive && showFrameOutline && (
        <div
          style={frameOutlineStyle}
          className={`absolute pointer-events-none border border-hairline/75 ${frameShadowClass}`}
        />
      )}

      {/* Interactive Rule-of-Thirds Crop Overlay (R) hugging exact photo bounds on the full stage */}
      {cropModeActive && onChangeCrop && (
        <div
          style={{
            left: `${Math.round(fitRect.x)}px`,
            top: `${Math.round(fitRect.y)}px`,
            width: `${Math.round(fitRect.width)}px`,
            height: `${Math.round(fitRect.height)}px`,
          }}
          className="absolute z-30"
        >
          <CropOverlay
            crop={photo.developRecipe.cropTransform}
            naturalAspect={orientedNaturalAspect}
            onChange={onChangeCrop}
          />
        </div>
      )}

      {/* Full-Stage Split Wipe Overlay: Original Untouched RAW on Left */}
      {compareMode === 'split' && !cropModeActive && (
        <>
          <div
            className="absolute inset-0 overflow-hidden pointer-events-none"
            style={{ clipPath: `inset(0 ${(100 - splitPct).toFixed(2)}% 0 0)` }}
          >
            <DevelopViewportCanvas
              src={developSrc}
              alt={`${photo.fileName} original`}
              recipe={beforeRecipe}
              workingBuffer={activeWorkingBuffer}
              previewMaxDim={effectivePreviewMaxDim}
              preferWebGpu={preferWebGpu}
              fit="contain"
              viewportPadPx={padPx}
              viewportZoom={effectiveZoom}
              viewportPan={clampedPan}
            />
          </div>

          {/* Interactive Amber Split Handle spanning the full stage */}
          <div
            role="slider"
            aria-label="Before and after split wipe"
            aria-valuemin={5}
            aria-valuemax={95}
            aria-valuenow={Math.round(splitPct)}
            tabIndex={0}
            onDoubleClick={(e) => e.stopPropagation()}
            onPointerDown={(e) => {
              e.stopPropagation();
              e.currentTarget.setPointerCapture(e.pointerId);
              setDraggingSplit(true);
              updateSplitFromClientX(e.clientX);
            }}
            style={{ left: `${splitPct}%` }}
            className="absolute top-0 bottom-0 -translate-x-1/2 w-6 flex items-center justify-center cursor-ew-resize z-20"
          >
            <div className="w-[2px] h-full bg-amber-core shadow-[0_0_8px_rgba(245,158,11,0.75)]" />
            <div className="absolute w-7 h-7 rounded-full bg-surface-active border-2 border-amber-core text-amber-soft flex items-center justify-center shadow-lg">
              <span className="material-symbols-outlined text-[15px]">code</span>
            </div>
          </div>

          <div className="absolute top-3 left-3 z-20 pointer-events-none">
            <TelemetryPill tone="neutral" className="bg-surface-lowest/85 backdrop-blur">
              ORIGINAL
            </TelemetryPill>
          </div>
          <div className="absolute top-3 right-3 z-20 pointer-events-none">
            <TelemetryPill tone="amber" className="bg-surface-lowest/85 backdrop-blur">
              EDITED
            </TelemetryPill>
          </div>
        </>
      )}

      {/* Active LUT Badge */}
      {photo.developRecipe.lut3d && !cropModeActive && !focusMode && (
        <div className="absolute bottom-3 left-3 z-20 pointer-events-none">
          <TelemetryPill tone="emerald" dot className="bg-surface-lowest/85 backdrop-blur">
            3D LUT: {photo.developRecipe.lut3d.name} ({photo.developRecipe.lut3d.intensity}%)
          </TelemetryPill>
        </div>
      )}
    </div>
  );
});
