import { memo, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { useStudio } from '../../app/StudioContext';
import { resolveCanvasBackdropHex } from '../../domain/preferences';
import {
  createDefaultDevelopRecipe,
  type CropTransformSettings,
} from '../../domain/develop';
import type { BrushStroke, MaskComponent, MaskLayer } from '../../domain/masking';
import {
  isRawFileName,
  shouldProbeHighBitDepthBuffer,
  type DevelopWorkingBuffer,
  type WorkingBufferPrecision,
} from '../../domain/developWorkingBuffer';
import { detectWebGpuAvailable } from '../../domain/wgslUniformPacker';
import type { PhotoAsset } from '../../domain/photo';
import { TelemetryPill } from '../darkroom-kit';
import { CropOverlay } from './CropOverlay';
import {
  DevelopViewportCanvas,
  type ActiveViewportBackend,
} from './DevelopViewportCanvas';
import { MaskCanvasOverlay, type BrushToolOptions } from './MaskCanvasOverlay';

export type CompareViewportMode = 'single' | 'split' | 'before_after';
export type ViewportZoomLevel = 1 | 2 | 3;

const NEUTRAL_DEVELOP_RECIPE = createDefaultDevelopRecipe();
const DEVELOP_WORKING_MAX_DIM = 1600;

interface BeforeAfterCanvasProps {
  readonly photo: PhotoAsset;
  readonly compareMode: CompareViewportMode;
  readonly zoomLevel?: ViewportZoomLevel;
  readonly onChangeZoom?: (next: ViewportZoomLevel) => void;
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
  readonly floatingOverlay?: ReactNode;
  readonly focusMode?: boolean;
}

interface LoadedWorkingState {
  readonly photoId: string;
  readonly buffer: DevelopWorkingBuffer;
}

export const BeforeAfterCanvas = memo(function BeforeAfterCanvas({
  photo,
  compareMode,
  zoomLevel = 1,
  onChangeZoom,
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
  floatingOverlay,
  focusMode = false,
}: BeforeAfterCanvasProps) {
  const { isDesktopRuntime, preferences, loadDevelopWorkingBuffer } = useStudio();
  const workingMaxDim = preferences.studioPreviewMaxDim ?? DEVELOP_WORKING_MAX_DIM;
  const stageBackdropStyle = {
    backgroundColor: resolveCanvasBackdropHex(preferences.canvasBackdropTone),
  };
  const [splitPct, setSplitPct] = useState(50);
  const [draggingSplit, setDraggingSplit] = useState(false);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [panning, setPanning] = useState(false);
  const panStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number }>({ w: 2, h: 3 });
  const [stageSize, setStageSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const [loadedWorking, setLoadedWorking] = useState<LoadedWorkingState | null>(null);
  const [decodingRawPhotoId, setDecodingRawPhotoId] = useState<string | null>(null);
  const preferWebGpu = preferences.enableGpuAcceleration !== false;
  const [viewportBackend, setViewportBackend] = useState<ActiveViewportBackend>(() =>
    preferWebGpu && detectWebGpuAvailable() ? 'webgpu' : 'webgl2',
  );
  const handleBackendChange = useCallback((backend: ActiveViewportBackend) => {
    setViewportBackend(backend);
  }, []);
  const requestedPrecision: WorkingBufferPrecision =
    viewportBackend === 'webgpu' ? 'f32' : 'f16';
  const stageRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const pad = focusMode ? 32 : 40;
    const update = () => {
      setStageSize({
        w: Math.max(120, el.clientWidth - pad),
        h: Math.max(120, el.clientHeight - pad),
      });
    };
    update();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [compareMode, cropModeActive, focusMode]);

  const shouldProbe = isDesktopRuntime && shouldProbeHighBitDepthBuffer(photo.fileName);
  const isRawSource = isRawFileName(photo.fileName);

  // Reset pan cleanly when returning to Fit zoom or switching photo
  useEffect(() => {
    if (zoomLevel <= 1) {
      setPan({ x: 0, y: 0 });
    }
  }, [zoomLevel, photo.id]);

  useEffect(() => {
    if (!shouldProbe) {
      setDecodingRawPhotoId(null);
      return;
    }

    let cancelled = false;
    if (isRawSource) {
      setDecodingRawPhotoId(photo.id);
    }

    void loadDevelopWorkingBuffer(photo.id, workingMaxDim, requestedPrecision)
      .then((buffer) => {
        if (cancelled) return;
        setDecodingRawPhotoId((prev) => (prev === photo.id ? null : prev));
        if (buffer) {
          setLoadedWorking({ photoId: photo.id, buffer });
        }
      })
      .catch(() => {
        if (!cancelled) {
          setDecodingRawPhotoId((prev) => (prev === photo.id ? null : prev));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [photo.id, shouldProbe, isRawSource, workingMaxDim, requestedPrecision, loadDevelopWorkingBuffer]);

  const activeWorkingBuffer =
    loadedWorking && loadedWorking.photoId === photo.id ? loadedWorking.buffer : null;
  const isDecodingRaw = decodingRawPhotoId === photo.id && !activeWorkingBuffer;

  const rotation = photo.developRecipe.cropTransform.rotationDeg;
  const effectiveZoom = cropModeActive ? 1 : zoomLevel;
  const zoomStyle =
    effectiveZoom > 1
      ? { transform: `translate(${pan.x}px, ${pan.y}px) scale(${effectiveZoom})` }
      : undefined;

  const updateSplitFromClientX = (clientX: number) => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSplitPct(Math.min(95, Math.max(5, pct)));
  };

  const handleStageWheel = (e: React.WheelEvent) => {
    if (!onChangeZoom || cropModeActive || (!e.ctrlKey && !e.metaKey)) return;
    e.preventDefault();
    if (e.deltaY < 0) {
      onChangeZoom(zoomLevel === 1 ? 2 : 3);
    } else {
      onChangeZoom(zoomLevel === 3 ? 2 : 1);
    }
  };

  const handleBeginPan = (e: React.PointerEvent<HTMLDivElement>) => {
    if (effectiveZoom <= 1 || eyedropperActive || colorPickActive || cropModeActive || draggingSplit)
      return;
    if (maskingActive && activeMaskLayer) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setPanning(true);
    panStartRef.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
  };

  const handleMovePan = (e: React.PointerEvent<HTMLDivElement>) => {
    if (draggingSplit) {
      updateSplitFromClientX(e.clientX);
      return;
    }
    if (!panning || !panStartRef.current) return;
    const rect = frameRef.current?.getBoundingClientRect();
    const maxOffset = rect ? Math.max(rect.width, rect.height) * ((effectiveZoom - 1) * 0.5) : 320;
    const nextX = Math.min(
      maxOffset,
      Math.max(-maxOffset, panStartRef.current.panX + (e.clientX - panStartRef.current.x)),
    );
    const nextY = Math.min(
      maxOffset,
      Math.max(-maxOffset, panStartRef.current.panY + (e.clientY - panStartRef.current.y)),
    );
    setPan({ x: nextX, y: nextY });
  };

  const handleEndPan = (e: React.PointerEvent<HTMLDivElement>) => {
    setDraggingSplit(false);
    setPanning(false);
    panStartRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const backendTag = viewportBackend === 'webgpu' ? 'WEBGPU' : 'WEBGL2';
  const precisionBadge = activeWorkingBuffer ? (
    <div className="absolute bottom-2.5 right-2.5 z-10 pointer-events-none">
      <TelemetryPill
        tone={activeWorkingBuffer.isRawDemosaic ? 'emerald' : 'amber'}
        dot
        className="bg-surface-lowest/85 backdrop-blur"
      >
        {activeWorkingBuffer.precision === 'f32'
          ? activeWorkingBuffer.isRawDemosaic
            ? `32-BIT SCENE-LINEAR RAW • ${backendTag} (${activeWorkingBuffer.width}×${activeWorkingBuffer.height})`
            : `32-BIT FLOAT • ${backendTag} (${activeWorkingBuffer.width}×${activeWorkingBuffer.height})`
          : activeWorkingBuffer.isRawDemosaic
            ? `16-BIT LINEAR RAW • ${backendTag} (${activeWorkingBuffer.width}×${activeWorkingBuffer.height})`
            : `16-BIT FLOAT • ${backendTag} (${activeWorkingBuffer.width}×${activeWorkingBuffer.height})`}
      </TelemetryPill>
    </div>
  ) : isDecodingRaw ? (
    <div className="absolute bottom-2.5 right-2.5 z-10 pointer-events-none">
      <TelemetryPill tone="neutral" className="bg-surface-lowest/85 backdrop-blur">
        {requestedPrecision === 'f32' ? 'Decoding 32-bit RAW…' : 'Decoding 16-bit RAW…'}
      </TelemetryPill>
    </div>
  ) : null;

  const toolHintBadge = colorPickActive ? (
    <div className="absolute top-2.5 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
      <TelemetryPill tone="amber" dot className="bg-surface-lowest/90 backdrop-blur">
        Color Range Picker — Click any color on the photo (Esc to cancel)
      </TelemetryPill>
    </div>
  ) : eyedropperActive ? (
    <div className="absolute top-2.5 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
      <TelemetryPill tone="emerald" dot className="bg-surface-lowest/90 backdrop-blur">
        WB Eyedropper — Click a neutral white/gray area (Esc to cancel)
      </TelemetryPill>
    </div>
  ) : null;

  const swapNatural = (photo.developRecipe.cropTransform.quarterTurns ?? 0) % 2 === 1;
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

  const computeFittedStyle = (
    aspect: number,
    maxW: number,
    maxH: number,
  ): { width?: string; height: string; aspectRatio?: string; maxWidth?: string; maxHeight?: string } => {
    const safeAspect = Number.isFinite(aspect) && aspect > 0.05 ? aspect : 2 / 3;
    if (maxW <= 0 || maxH <= 0) {
      return {
        aspectRatio: String(safeAspect),
        height: '100%',
        maxHeight: '660px',
        maxWidth: '100%',
      };
    }
    const boundedH = focusMode ? maxH : Math.min(maxH, 760);
    let w = boundedH * safeAspect;
    let h = boundedH;
    if (w > maxW) {
      w = maxW;
      h = w / safeAspect;
    }
    return { width: `${Math.round(w)}px`, height: `${Math.round(h)}px` };
  };

  const beforeRecipe = {
    ...NEUTRAL_DEVELOP_RECIPE,
    cropTransform: photo.developRecipe.cropTransform,
  };

  const singleFrameStyle = computeFittedStyle(frameAspect, stageSize.w, stageSize.h);
  const pairFrameStyle = computeFittedStyle(
    frameAspect,
    Math.max(120, (stageSize.w - 16) * 0.5),
    stageSize.h,
  );

  if (compareMode === 'before_after' && !cropModeActive) {
    return (
      <div
        ref={stageRef}
        onWheel={handleStageWheel}
        style={stageBackdropStyle}
        className="flex-1 min-h-0 flex items-center justify-center gap-4 p-5 overflow-hidden relative transition-colors duration-200"
      >
        {toolHintBadge}
        {floatingOverlay}
        <div
          style={pairFrameStyle}
          className="relative rounded-lg overflow-hidden border border-hairline bg-surface-lowest"
        >
          <div className="w-full h-full" style={zoomStyle}>
            <DevelopViewportCanvas
              src={photo.imageUrl}
              alt={`${photo.fileName} original`}
              recipe={beforeRecipe}
              workingBuffer={activeWorkingBuffer}
              preferWebGpu={preferWebGpu}
              fit="contain"
              rotationDeg={rotation}
              rotationScale={1.06}
            />
          </div>
          <div className="absolute top-2.5 left-2.5">
            <TelemetryPill tone="neutral" className="bg-surface-lowest/85 backdrop-blur">
              ORIGINAL
            </TelemetryPill>
          </div>
        </div>

        <div
          style={pairFrameStyle}
          className="relative rounded-lg overflow-hidden border border-amber-core/50 bg-surface-lowest"
        >
          <div className="w-full h-full" style={zoomStyle}>
            <DevelopViewportCanvas
              src={photo.imageUrl}
              alt={`${photo.fileName} edited`}
              recipe={photo.developRecipe}
              workingBuffer={activeWorkingBuffer}
              preferWebGpu={preferWebGpu}
              onBackendChange={handleBackendChange}
              fit="contain"
              rotationDeg={rotation}
              rotationScale={1.06}
              showShadowClip={showShadowClip}
              showHighlightClip={showHighlightClip}
              eyedropperActive={eyedropperActive}
              onSampleNeutralRgb={onSampleNeutralRgb}
              activeMaskId={activeMaskLayer?.id ?? null}
              showMaskOverlay={showMaskOverlay && maskingActive}
              colorPickActive={colorPickActive}
              onSampleColorRangeRgb={onSampleColorRangeRgb}
              onNaturalSize={(w, h) => setNaturalSize({ w, h })}
              className="transition-transform duration-100"
            />
          </div>
          <div className="absolute top-2.5 left-2.5">
            <TelemetryPill tone="amber" className="bg-surface-lowest/85 backdrop-blur">
              EDITED
            </TelemetryPill>
          </div>
          {!focusMode && precisionBadge}
        </div>
      </div>
    );
  }

  return (
    <div
      ref={stageRef}
      onWheel={handleStageWheel}
      style={stageBackdropStyle}
      className={`flex-1 min-h-0 flex items-center justify-center ${
        focusMode ? 'p-4' : 'p-5'
      } overflow-hidden relative transition-colors duration-200`}
    >
      {toolHintBadge}
      {floatingOverlay}
      <div
        ref={frameRef}
        style={singleFrameStyle}
        onPointerDown={handleBeginPan}
        onPointerMove={handleMovePan}
        onPointerUp={handleEndPan}
        onPointerLeave={handleEndPan}
        className={`relative rounded-lg ${cropModeActive ? 'overflow-visible' : 'overflow-hidden'} border border-hairline bg-surface-lowest shadow-[0_8px_32px_rgba(0,0,0,0.65)] select-none ${
          effectiveZoom > 1 && !eyedropperActive && !cropModeActive
            ? panning
              ? 'cursor-grabbing'
              : 'cursor-grab'
            : ''
        }`}
      >
        {/* Base Layer: Unified 32-Bit Wide-Gamut WebGPU / WebGL2 Kroma Develop Grade */}
        <div className="w-full h-full relative rounded-lg overflow-hidden" style={zoomStyle}>
          <DevelopViewportCanvas
            src={photo.imageUrl}
            alt={`${photo.fileName} graded`}
            recipe={photo.developRecipe}
            workingBuffer={activeWorkingBuffer}
            preferWebGpu={preferWebGpu}
            onBackendChange={handleBackendChange}
            fit="contain"
            applyCrop={!cropModeActive}
            rotationDeg={rotation}
            rotationScale={1.06}
            showShadowClip={showShadowClip}
            showHighlightClip={showHighlightClip}
            eyedropperActive={eyedropperActive}
            onSampleNeutralRgb={onSampleNeutralRgb}
            activeMaskId={activeMaskLayer?.id ?? null}
            showMaskOverlay={showMaskOverlay && maskingActive}
            colorPickActive={colorPickActive}
            onSampleColorRangeRgb={onSampleColorRangeRgb}
            onNaturalSize={(w, h) => setNaturalSize({ w, h })}
          />

          {/* Interactive Local Mask Handles & Freehand Brush Painter */}
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
                naturalWidth={naturalSize.w}
                naturalHeight={naturalSize.h}
                brushOptions={brushOptions}
                colorPickActive={colorPickActive}
                onUpdateComponent={onUpdateMaskComponent}
                onEnsureBrushStroke={onEnsureBrushStroke}
              />
            )}
        </div>

        {/* Interactive Rule-of-Thirds Crop Overlay (R) hugging exact photo edges */}
        {cropModeActive && onChangeCrop && (
          <CropOverlay
            crop={photo.developRecipe.cropTransform}
            naturalAspect={orientedNaturalAspect}
            onChange={onChangeCrop}
          />
        )}

        {/* Split Wipe Overlay: Original Untouched RAW on Left */}
        {compareMode === 'split' && !cropModeActive && (
          <>
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none"
              style={{ clipPath: `inset(0 ${(100 - splitPct).toFixed(2)}% 0 0)` }}
            >
              <div className="w-full h-full" style={zoomStyle}>
                <DevelopViewportCanvas
                  src={photo.imageUrl}
                  alt={`${photo.fileName} original`}
                  recipe={beforeRecipe}
                  workingBuffer={activeWorkingBuffer}
                  preferWebGpu={preferWebGpu}
                  fit="contain"
                  rotationDeg={rotation}
                  rotationScale={1.06}
                />
              </div>
            </div>

            {/* Interactive Amber Split Handle */}
            <div
              role="slider"
              aria-label="Before and after split wipe"
              aria-valuemin={5}
              aria-valuemax={95}
              aria-valuenow={Math.round(splitPct)}
              tabIndex={0}
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

            <div className="absolute top-2.5 left-2.5 z-10">
              <TelemetryPill tone="neutral" className="bg-surface-lowest/85 backdrop-blur">
                ORIGINAL
              </TelemetryPill>
            </div>
            <div className="absolute top-2.5 right-2.5 z-10">
              <TelemetryPill tone="amber" className="bg-surface-lowest/85 backdrop-blur">
                EDITED
              </TelemetryPill>
            </div>
          </>
        )}

        {/* Active LUT Badge */}
        {photo.developRecipe.lut3d && !cropModeActive && !focusMode && (
          <div className="absolute bottom-2.5 left-2.5 z-10">
            <TelemetryPill tone="emerald" dot className="bg-surface-lowest/85 backdrop-blur">
              3D LUT: {photo.developRecipe.lut3d.name} ({photo.developRecipe.lut3d.intensity}%)
            </TelemetryPill>
          </div>
        )}

        {!cropModeActive && !focusMode && precisionBadge}
      </div>
    </div>
  );
});
