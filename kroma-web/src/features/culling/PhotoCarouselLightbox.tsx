import {
  AnimatePresence,
  animate,
  motion,
  useMotionValue,
  useTransform,
  type MotionValue,
} from 'framer-motion';
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react';
import { createPortal } from 'react-dom';
import { isRecipeModified } from '../../domain/develop';
import { GradedThumbnailImg } from '../develop/GradedThumbnailImg';
import { WebGlDevelopCanvas } from '../develop/WebGlDevelopCanvas';
import { formatSimilarityPercent, needsVisualMatchReview } from '../../domain/identity';
import type { PhotoAsset } from '../../domain/photo';
import {
  DarkroomButton,
  DarkroomTooltip,
  FAST_FADE,
  TACTILE_SPRING,
  TelemetryPill,
  VirtualHorizontalStrip,
} from '../darkroom-kit';
import {
  applyRubberBandProgress,
  clampZoomPan,
  computeOpeningOffset,
  fitPhotoToStage,
  resolveCarouselTargetIndex,
  type RectLike,
} from './lightboxGeometry';

const CAROUSEL_SPRING = {
  type: 'spring' as const,
  stiffness: 350,
  damping: 35,
  mass: 0.82,
};

const OPEN_SPRING = {
  type: 'spring' as const,
  stiffness: 380,
  damping: 34,
  mass: 0.78,
};

const ZOOM_LEVEL = 2;
const DRAG_DEADZONE_PX = 6;
const WHEEL_COOLDOWN_MS = 240;

interface PhotoCarouselLightboxProps {
  readonly open: boolean;
  readonly photos: readonly PhotoAsset[];
  readonly activeIndex: number;
  readonly originRect: RectLike | null;
  readonly subjectName: string;
  readonly isClustered: boolean;
  readonly isSpecialBucket: boolean;
  readonly reviewWarningThreshold: number;
  readonly approvedReviewPhotoIds: ReadonlySet<string>;
  readonly showLandmarksOverlay: boolean;
  readonly selectedPhotoIds: ReadonlySet<string>;
  readonly onChangeIndex: (nextIndex: number) => void;
  readonly onClose: () => void;
  readonly onToggleSelect: (photoId: string) => void;
  readonly onToggleLandmarks: () => void;
  readonly onOpenDevelop: (photoId: string) => void;
  readonly onOpenManualMatch?: () => void;
}

interface CarouselSlideItemProps {
  readonly photo: PhotoAsset;
  readonly index: number;
  readonly isActive: boolean;
  readonly trackProgress: MotionValue<number>;
  readonly slideStridePx: number;
  readonly maxFrameWidth: number;
  readonly maxFrameHeight: number;
  readonly reviewWarningThreshold: number;
  readonly isReviewApproved: boolean;
  readonly showLandmarksOverlay: boolean;
  readonly isZoomed: boolean;
  readonly zoomPan: { readonly x: number; readonly y: number };
  readonly onSelectSideSlide: (index: number) => void;
  readonly onToggleZoom: () => void;
  readonly onNaturalSize: (photoId: string, w: number, h: number) => void;
  readonly naturalSize?: { readonly w: number; readonly h: number };
}

function CarouselSlideItem({
  photo,
  index,
  isActive,
  trackProgress,
  slideStridePx,
  maxFrameWidth,
  maxFrameHeight,
  reviewWarningThreshold,
  isReviewApproved,
  showLandmarksOverlay,
  isZoomed,
  zoomPan,
  onSelectSideSlide,
  onToggleZoom,
  onNaturalSize,
  naturalSize,
}: CarouselSlideItemProps) {
  const x = useTransform(trackProgress, (p) => (index - p) * slideStridePx);
  const scale = useTransform(
    trackProgress,
    [index - 2, index - 1, index, index + 1, index + 2],
    [0.78, 0.86, 1, 0.86, 0.78],
  );
  const opacity = useTransform(
    trackProgress,
    [index - 1.65, index - 1, index, index + 1, index + 1.65],
    [0, 0.44, 1, 0.44, 0],
  );

  const fitted = fitPhotoToStage(
    naturalSize?.w ?? null,
    naturalSize?.h ?? null,
    maxFrameWidth,
    maxFrameHeight,
  );
  const rotation = photo.developRecipe.cropTransform.rotationDeg;
  const isAmbiguous =
    !isReviewApproved && needsVisualMatchReview(photo, reviewWarningThreshold);
  const edited = isRecipeModified(photo.developRecipe);

  return (
    <motion.div
      style={{ x, scale, opacity }}
      className={`absolute inset-0 flex items-center justify-center pointer-events-none ${
        isActive ? 'z-20' : 'z-10'
      }`}
    >
      <div
        data-lightbox-card="true"
        onClick={(e) => {
          e.stopPropagation();
          if (!isActive) {
            onSelectSideSlide(index);
          }
        }}
        onDoubleClick={(e) => {
          e.stopPropagation();
          if (isActive) {
            onToggleZoom();
          }
        }}
        style={{ width: `${fitted.width}px`, height: `${fitted.height}px` }}
        className={`relative rounded-2xl overflow-hidden bg-surface-lowest border shadow-[0_24px_70px_rgba(0,0,0,0.82)] pointer-events-auto select-none transition-[border-color,opacity] duration-150 ${
          isActive
            ? isZoomed
              ? 'border-amber-core/60 cursor-move'
              : 'border-white/15 cursor-grab active:cursor-grabbing'
            : 'border-white/10 hover:border-white/30 hover:opacity-85 cursor-pointer'
        }`}
      >
        {/* Zoomable + Pannable Photo Layer */}
        <motion.div
          animate={{
            scale: isActive && isZoomed ? ZOOM_LEVEL : 1,
            x: isActive && isZoomed ? zoomPan.x : 0,
            y: isActive && isZoomed ? zoomPan.y : 0,
          }}
          transition={isZoomed ? { type: 'spring', stiffness: 420, damping: 38 } : CAROUSEL_SPRING}
          className="relative w-full h-full"
        >
          {isActive && edited ? (
            <WebGlDevelopCanvas
              src={photo.imageUrl}
              alt={photo.fileName}
              recipe={photo.developRecipe}
              fit="contain"
              rotationDeg={rotation}
              rotationScale={1}
              onNaturalSize={(w, h) => onNaturalSize(photo.id, w, h)}
              className="select-none"
            />
          ) : (
            <GradedThumbnailImg
              src={photo.imageUrl}
              alt={photo.fileName}
              recipe={photo.developRecipe}
              draggable={false}
              loading="eager"
              decoding="async"
              onLoad={(e) => {
                const img = e.currentTarget;
                if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                  onNaturalSize(photo.id, img.naturalWidth, img.naturalHeight);
                }
              }}
              style={{
                transform: rotation !== 0 ? `rotate(${rotation}deg)` : undefined,
              }}
              className="w-full h-full object-contain select-none"
            />
          )}

          {/* Pixel-Exact YuNet Face Box & 5-Point Landmarks Overlay */}
          <AnimatePresence>
            {isActive && showLandmarksOverlay && photo.faceOverlay && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={FAST_FADE}
                className="absolute inset-0 pointer-events-none"
              >
                <div
                  className="absolute"
                  style={{
                    top: `${photo.faceOverlay.topPct}%`,
                    left: `${photo.faceOverlay.leftPct}%`,
                    width: `${photo.faceOverlay.widthPct}%`,
                    height: `${photo.faceOverlay.heightPct}%`,
                  }}
                >
                  <div
                    className={`absolute top-0 left-0 w-3.5 h-3.5 border-t-2 border-l-2 ${
                      isAmbiguous ? 'border-tangerine-alert' : 'border-emerald-focus'
                    }`}
                  />
                  <div
                    className={`absolute top-0 right-0 w-3.5 h-3.5 border-t-2 border-r-2 ${
                      isAmbiguous ? 'border-tangerine-alert' : 'border-emerald-focus'
                    }`}
                  />
                  <div
                    className={`absolute bottom-0 left-0 w-3.5 h-3.5 border-b-2 border-l-2 ${
                      isAmbiguous ? 'border-tangerine-alert' : 'border-emerald-focus'
                    }`}
                  />
                  <div
                    className={`absolute bottom-0 right-0 w-3.5 h-3.5 border-b-2 border-r-2 ${
                      isAmbiguous ? 'border-tangerine-alert' : 'border-emerald-focus'
                    }`}
                  />

                  {[
                    photo.faceOverlay.rightEye,
                    photo.faceOverlay.leftEye,
                    photo.faceOverlay.noseTip,
                    photo.faceOverlay.rightMouth,
                    photo.faceOverlay.leftMouth,
                  ].map((pt, idx) => (
                    <span
                      key={idx}
                      style={{ left: `${pt.x}%`, top: `${pt.y}%` }}
                      className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full ${
                        idx === 2 ? 'w-1.5 h-1.5' : 'w-2 h-2'
                      } ${isAmbiguous ? 'bg-tangerine-alert' : 'bg-emerald-bright'}`}
                    />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Subtle Corner Badges on Active Slide */}
        {isActive && photo.isSlateAnchor && (
          <div className="absolute top-3 left-3 z-10 pointer-events-none">
            <span className="px-2 py-0.5 rounded bg-amber-core text-amber-ink font-display text-[10px] font-bold tracking-wider uppercase shadow">
              SLATE
            </span>
          </div>
        )}

        {isActive && edited && (
          <div className="absolute bottom-3 right-3 z-10 pointer-events-none px-2 py-0.5 rounded bg-surface-lowest/85 backdrop-blur border border-amber-core/50 tabular-nums-mono text-[10px] text-amber-soft">
            EDITED
          </div>
        )}
      </div>
    </motion.div>
  );
}

/**
 * Motion-powered Carousel Lightbox:
 * - Blooms smoothly from the double-clicked photo card with spring physics.
 * - Continuous 5-slide spring track with peeking adjacent frames, pointer drag/swipe with momentum,
 *   keyboard navigation, 2x sharpness inspection zoom, and a virtualized bottom thumbnail scrubber.
 */
export function PhotoCarouselLightbox({
  open,
  photos,
  activeIndex,
  originRect,
  subjectName,
  isClustered,
  isSpecialBucket,
  reviewWarningThreshold,
  approvedReviewPhotoIds,
  showLandmarksOverlay,
  selectedPhotoIds,
  onChangeIndex,
  onClose,
  onToggleSelect,
  onToggleLandmarks,
  onOpenDevelop,
  onOpenManualMatch,
}: PhotoCarouselLightboxProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 1100, height: 640 });
  const [naturalSizes, setNaturalSizes] = useState<
    Record<string, { readonly w: number; readonly h: number }>
  >({});
  const [isZoomed, setIsZoomed] = useState(false);
  const [zoomPan, setZoomPan] = useState({ x: 0, y: 0 });

  const clampedIndex = photos.length > 0 ? Math.max(0, Math.min(photos.length - 1, activeIndex)) : 0;
  const activePhoto = photos[clampedIndex] ?? null;

  const trackProgress = useMotionValue(clampedIndex);
  const activeAnimRef = useRef<ReturnType<typeof animate> | null>(null);
  const pendingVelocityRef = useRef<number | null>(null);
  const wasOpenRef = useRef(false);
  const lastWheelNavRef = useRef(0);

  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    lastX: number;
    lastTime: number;
    velocityX: number;
    startProgress: number;
    startPanX: number;
    startPanY: number;
    isDragging: boolean;
    mode: 'carousel' | 'pan';
  } | null>(null);
  const suppressNextBackdropClickRef = useRef(false);

  // Observe available stage dimensions for exact aspect-fitted photo frames and peeking side stride
  useEffect(() => {
    if (!open) return;
    const el = stageRef.current;
    if (!el) return;
    const measure = () => {
      setStageSize({ width: el.clientWidth, height: el.clientHeight });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [open]);

  // Snap immediately on initial open; spring-glide on every subsequent slide change
  useEffect(() => {
    if (!open) {
      wasOpenRef.current = false;
      activeAnimRef.current?.stop();
      setIsZoomed(false);
      setZoomPan({ x: 0, y: 0 });
      return;
    }

    if (!wasOpenRef.current) {
      wasOpenRef.current = true;
      pendingVelocityRef.current = null;
      activeAnimRef.current?.stop();
      trackProgress.set(clampedIndex);
    } else {
      const velocity = pendingVelocityRef.current;
      pendingVelocityRef.current = null;
      activeAnimRef.current?.stop();
      activeAnimRef.current = animate(
        trackProgress,
        clampedIndex,
        velocity !== null ? { ...CAROUSEL_SPRING, velocity } : CAROUSEL_SPRING,
      );
    }
    setIsZoomed(false);
    setZoomPan({ x: 0, y: 0 });
  }, [open, clampedIndex, trackProgress]);

  const handleNaturalSize = useCallback((photoId: string, w: number, h: number) => {
    setNaturalSizes((prev) => {
      const existing = prev[photoId];
      if (existing && existing.w === w && existing.h === h) return prev;
      return { ...prev, [photoId]: { w, h } };
    });
  }, []);

  const goToIndex = useCallback(
    (nextIndex: number) => {
      if (photos.length === 0) return;
      const clamped = Math.max(0, Math.min(photos.length - 1, nextIndex));
      if (clamped === clampedIndex) {
        activeAnimRef.current?.stop();
        activeAnimRef.current = animate(trackProgress, clamped, CAROUSEL_SPRING);
        return;
      }
      onChangeIndex(clamped);
    },
    [photos.length, clampedIndex, onChangeIndex, trackProgress],
  );

  const handleSelectSideSlide = useCallback(
    (index: number) => {
      if (suppressNextBackdropClickRef.current) return;
      goToIndex(index);
    },
    [goToIndex],
  );

  const toggleZoom = useCallback(() => {
    setIsZoomed((prev) => {
      if (prev) setZoomPan({ x: 0, y: 0 });
      return !prev;
    });
  }, []);

  // Full photographer keyboard shortcuts while lightbox is open
  useEffect(() => {
    if (!open || !activePhoto) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      ) {
        return;
      }
      switch (event.key) {
        case 'Escape':
          event.preventDefault();
          if (isZoomed) {
            setIsZoomed(false);
            setZoomPan({ x: 0, y: 0 });
          } else {
            onClose();
          }
          break;
        case 'ArrowRight':
        case 'd':
        case 'D':
          event.preventDefault();
          goToIndex(clampedIndex + 1);
          break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
          event.preventDefault();
          goToIndex(clampedIndex - 1);
          break;
        case 'Home':
          event.preventDefault();
          goToIndex(0);
          break;
        case 'End':
          event.preventDefault();
          goToIndex(photos.length - 1);
          break;
        case ' ':
          event.preventDefault();
          if (!activePhoto.isSlateAnchor) {
            onToggleSelect(activePhoto.id);
          }
          break;
        case 'z':
        case 'Z':
          event.preventDefault();
          toggleZoom();
          break;
        case 'f':
        case 'F':
          event.preventDefault();
          onToggleLandmarks();
          break;
        case 'e':
        case 'E':
          event.preventDefault();
          onClose();
          onOpenDevelop(activePhoto.id);
          break;
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    open,
    activePhoto,
    clampedIndex,
    photos.length,
    isZoomed,
    goToIndex,
    onClose,
    onToggleSelect,
    onToggleLandmarks,
    onOpenDevelop,
    toggleZoom,
  ]);

  // Responsive stage frame budget (leaves room on left & right so adjacent slides peek cleanly)
  const maxFrameWidth = Math.max(240, Math.min(stageSize.width * 0.72, 1080));
  const maxFrameHeight = Math.max(220, stageSize.height - 28);
  const activeNatural = activePhoto ? naturalSizes[activePhoto.id] : undefined;
  const activeFitted = fitPhotoToStage(
    activeNatural?.w ?? null,
    activeNatural?.h ?? null,
    maxFrameWidth,
    maxFrameHeight,
  );
  const slideStridePx = Math.max(activeFitted.width + 44, Math.min(stageSize.width * 0.62, maxFrameWidth + 44));

  const openingTransform = useMemo(() => {
    const vw = typeof window !== 'undefined' ? window.innerWidth : 1280;
    const vh = typeof window !== 'undefined' ? window.innerHeight : 800;
    return computeOpeningOffset(originRect, vw, vh, activeFitted.width);
  }, [originRect, activeFitted.width]);

  // Only mount a tiny 5-slide window around `clampedIndex` (constant O(1) DOM cost even on 2,000+ photos)
  const visibleSlideIndices = useMemo(() => {
    const out: number[] = [];
    const start = Math.max(0, clampedIndex - 2);
    const end = Math.min(photos.length - 1, clampedIndex + 2);
    for (let i = start; i <= end; i++) {
      out.push(i);
    }
    return out;
  }, [clampedIndex, photos.length]);

  const handleStagePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    if ((e.target as HTMLElement).closest('button')) return;

    activeAnimRef.current?.stop();
    dragRef.current = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      lastX: e.clientX,
      lastTime: performance.now(),
      velocityX: 0,
      startProgress: trackProgress.get(),
      startPanX: zoomPan.x,
      startPanY: zoomPan.y,
      isDragging: false,
      mode: isZoomed ? 'pan' : 'carousel',
    };
  };

  const handleStagePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;

    const deltaX = e.clientX - drag.startX;
    const deltaY = e.clientY - drag.startY;
    if (!drag.isDragging) {
      if (Math.hypot(deltaX, deltaY) < DRAG_DEADZONE_PX) return;
      drag.isDragging = true;
      e.currentTarget.setPointerCapture(e.pointerId);
    }

    if (drag.mode === 'pan') {
      const clamped = clampZoomPan(
        drag.startPanX + deltaX,
        drag.startPanY + deltaY,
        activeFitted.width,
        activeFitted.height,
        ZOOM_LEVEL,
      );
      setZoomPan(clamped);
      return;
    }

    const now = performance.now();
    const dt = Math.max(1, now - drag.lastTime);
    const instantVelocity = ((e.clientX - drag.lastX) / dt) * 1000;
    drag.velocityX = drag.velocityX * 0.45 + instantVelocity * 0.55;
    drag.lastX = e.clientX;
    drag.lastTime = now;

    const rawProgress = drag.startProgress - deltaX / slideStridePx;
    trackProgress.set(applyRubberBandProgress(rawProgress, photos.length - 1));
  };

  const finishPointerDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    dragRef.current = null;

    if (!drag.isDragging) return;
    suppressNextBackdropClickRef.current = true;
    window.setTimeout(() => {
      suppressNextBackdropClickRef.current = false;
    }, 80);

    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }

    if (drag.mode === 'pan') return;

    const deltaX = e.clientX - drag.startX;
    const nextIndex = resolveCarouselTargetIndex(
      clampedIndex,
      deltaX,
      drag.velocityX,
      slideStridePx,
      photos.length,
    );

    const springVelocity = Math.max(-18, Math.min(18, -drag.velocityX / slideStridePx));
    if (nextIndex !== clampedIndex) {
      pendingVelocityRef.current = springVelocity;
      onChangeIndex(nextIndex);
    } else {
      activeAnimRef.current?.stop();
      activeAnimRef.current = animate(trackProgress, clampedIndex, {
        ...CAROUSEL_SPRING,
        velocity: springVelocity,
      });
    }
  };

  const handleStageWheel = (e: ReactWheelEvent<HTMLDivElement>) => {
    if (isZoomed || photos.length <= 1) return;
    const horizontalDelta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.shiftKey ? e.deltaY : 0;
    if (Math.abs(horizontalDelta) < 24) return;
    const now = performance.now();
    if (now - lastWheelNavRef.current < WHEEL_COOLDOWN_MS) return;
    lastWheelNavRef.current = now;
    goToIndex(clampedIndex + (horizontalDelta > 0 ? 1 : -1));
  };

  if (typeof document === 'undefined') return null;

  const isSelected = activePhoto ? selectedPhotoIds.has(activePhoto.id) : false;
  const activeAmbiguous = Boolean(
    activePhoto &&
      !approvedReviewPhotoIds.has(activePhoto.id) &&
      needsVisualMatchReview(activePhoto, reviewWarningThreshold),
  );

  return createPortal(
    <AnimatePresence>
      {open && activePhoto && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Photo Carousel Lightbox: ${activePhoto.fileName}`}
          className="fixed inset-0 z-[70] flex flex-col select-none"
        >
          {/* Frosted Obsidian Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={FAST_FADE}
            onClick={() => {
              if (!suppressNextBackdropClickRef.current) onClose();
            }}
            className="absolute inset-0 bg-canvas/92 backdrop-blur-xl"
          />

          {/* Top Floating Darkroom HUD */}
          <motion.header
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={FAST_FADE}
            className="relative z-30 h-[54px] shrink-0 px-5 bg-surface-lowest/75 backdrop-blur-md border-b border-hairline/70 flex items-center justify-between gap-4"
          >
            {/* Left: Counter, Filename, Subject & Telemetry */}
            <div className="flex items-center gap-3 min-w-0">
              <span className="tabular-nums-mono text-[11px] px-2 py-0.5 rounded-md bg-surface border border-hairline text-ink-secondary shrink-0">
                {clampedIndex + 1} / {photos.length}
              </span>

              <div className="min-w-0 flex flex-col">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="tabular-nums-mono text-[13px] font-semibold text-ink-primary truncate">
                    {activePhoto.fileName}
                  </span>
                  <TelemetryPill tone="neutral" className="shrink-0 hidden sm:inline-flex">
                    {subjectName}
                  </TelemetryPill>
                  {activePhoto.faceOverlay && !activePhoto.isSlateAnchor && (
                    <TelemetryPill
                      tone={activeAmbiguous ? 'tangerine' : 'emerald'}
                      dot
                      className="shrink-0"
                    >
                      {formatSimilarityPercent(activePhoto.sfaceCosineSimilarity)}
                    </TelemetryPill>
                  )}
                </div>
                <span className="tabular-nums-mono text-[10px] text-ink-telemetry truncate">
                  {[
                    activePhoto.exif.shutterSpeed,
                    activePhoto.exif.aperture,
                    activePhoto.exif.iso > 0 ? `ISO ${activePhoto.exif.iso}` : null,
                    activePhoto.capturedAtLabel,
                  ]
                    .filter(Boolean)
                    .join(' • ')}
                </span>
              </div>
            </div>

            {/* Right: Focused Inspection & Workflow Actions */}
            <div className="flex items-center gap-2 shrink-0">
              {!activePhoto.isSlateAnchor && (
                <DarkroomButton
                  variant={isSelected ? 'primary' : 'subtle'}
                  size="sm"
                  tooltipSide="bottom"
                  icon={isSelected ? 'check_box' : 'check_box_outline_blank'}
                  onClick={() => onToggleSelect(activePhoto.id)}
                  title="Select photo (Space)"
                >
                  <span className="hidden md:inline">{isSelected ? 'Selected' : 'Select'}</span>
                </DarkroomButton>
              )}

              {activePhoto.faceOverlay && (
                <DarkroomButton
                  variant={showLandmarksOverlay ? 'emerald' : 'subtle'}
                  size="sm"
                  tooltipSide="bottom"
                  icon="face"
                  onClick={onToggleLandmarks}
                  title="Face boxes (F)"
                >
                  <span className="hidden lg:inline">Face Box</span>
                </DarkroomButton>
              )}

              <DarkroomButton
                variant={isZoomed ? 'primary' : 'subtle'}
                size="sm"
                tooltipSide="bottom"
                icon={isZoomed ? 'zoom_out' : 'zoom_in'}
                onClick={toggleZoom}
                title={isZoomed ? 'Reset zoom (Z)' : 'Zoom 2× (Z)'}
              >
                <span className="hidden md:inline">{isZoomed ? '2×' : '1×'}</span>
              </DarkroomButton>

              {isClustered && !isSpecialBucket && onOpenManualMatch && (
                <DarkroomButton
                  variant="secondary"
                  size="sm"
                  tooltipSide="bottom"
                  icon="compare"
                  onClick={() => {
                    onClose();
                    onOpenManualMatch();
                  }}
                  title="Manual match"
                >
                  <span className="hidden lg:inline">Match</span>
                </DarkroomButton>
              )}

              <DarkroomButton
                variant="secondary"
                size="sm"
                tooltipSide="bottom"
                icon="tune"
                iconClassName="text-amber-soft"
                onClick={() => {
                  onClose();
                  onOpenDevelop(activePhoto.id);
                }}
                title="Edit in Develop (E)"
              >
                <span>Edit</span>
              </DarkroomButton>

              <DarkroomTooltip content="Close (Esc)" side="bottom">
                <button
                  type="button"
                  onClick={onClose}
                  aria-label="Close lightbox"
                  className="w-8 h-8 rounded-lg bg-surface border border-hairline hover:border-hairline-strong text-ink-secondary hover:text-ink-primary flex items-center justify-center transition-colors cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[18px]">close</span>
                </button>
              </DarkroomTooltip>
            </div>
          </motion.header>

          {/* Main Interactive Carousel Stage */}
          <motion.div
            ref={stageRef}
            initial={{
              opacity: 0,
              x: openingTransform.x,
              y: openingTransform.y,
              scale: openingTransform.scale,
            }}
            animate={{ opacity: 1, x: 0, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.92 }}
            transition={OPEN_SPRING}
            onPointerDown={handleStagePointerDown}
            onPointerMove={handleStagePointerMove}
            onPointerUp={finishPointerDrag}
            onPointerCancel={finishPointerDrag}
            onWheel={handleStageWheel}
            onClick={(e) => {
              if (suppressNextBackdropClickRef.current) return;
              if (!(e.target as HTMLElement).closest('[data-lightbox-card="true"]')) {
                onClose();
              }
            }}
            className="relative z-20 flex-1 min-h-0 overflow-hidden touch-none"
          >
            {/* Windowed Continuous Spring Slides */}
            {visibleSlideIndices.map((idx) => {
              const photo = photos[idx];
              if (!photo) return null;
              return (
                <CarouselSlideItem
                  key={photo.id}
                  photo={photo}
                  index={idx}
                  isActive={idx === clampedIndex}
                  trackProgress={trackProgress}
                  slideStridePx={slideStridePx}
                  maxFrameWidth={maxFrameWidth}
                  maxFrameHeight={maxFrameHeight}
                  reviewWarningThreshold={reviewWarningThreshold}
                  isReviewApproved={approvedReviewPhotoIds.has(photo.id)}
                  showLandmarksOverlay={showLandmarksOverlay}
                  isZoomed={isZoomed}
                  zoomPan={zoomPan}
                  onSelectSideSlide={handleSelectSideSlide}
                  onToggleZoom={toggleZoom}
                  onNaturalSize={handleNaturalSize}
                  naturalSize={naturalSizes[photo.id]}
                />
              );
            })}

            {/* Floating Glass Previous Button */}
            {photos.length > 1 && (
              <div className="absolute left-5 top-1/2 -translate-y-1/2 z-30">
                <DarkroomTooltip content="Previous (←)" side="right">
                  <motion.button
                    type="button"
                    whileHover={{ scale: 1.06 }}
                    whileTap={{ scale: 0.92 }}
                    transition={TACTILE_SPRING}
                    disabled={clampedIndex <= 0}
                    onClick={(e) => {
                      e.stopPropagation();
                      goToIndex(clampedIndex - 1);
                    }}
                    aria-label="Previous photo"
                    className="w-11 h-11 rounded-full bg-surface-lowest/75 hover:bg-surface-active/90 backdrop-blur-md border border-white/15 hover:border-amber-core/60 text-ink-primary disabled:opacity-25 disabled:pointer-events-none flex items-center justify-center shadow-[0_8px_24px_rgba(0,0,0,0.55)] cursor-pointer transition-colors"
                  >
                    <span className="material-symbols-outlined text-[22px]">chevron_left</span>
                  </motion.button>
                </DarkroomTooltip>
              </div>
            )}

            {/* Floating Glass Next Button */}
            {photos.length > 1 && (
              <div className="absolute right-5 top-1/2 -translate-y-1/2 z-30">
                <DarkroomTooltip content="Next (→)" side="left">
                  <motion.button
                    type="button"
                    whileHover={{ scale: 1.06 }}
                    whileTap={{ scale: 0.92 }}
                    transition={TACTILE_SPRING}
                    disabled={clampedIndex >= photos.length - 1}
                    onClick={(e) => {
                      e.stopPropagation();
                      goToIndex(clampedIndex + 1);
                    }}
                    aria-label="Next photo"
                    className="w-11 h-11 rounded-full bg-surface-lowest/75 hover:bg-surface-active/90 backdrop-blur-md border border-white/15 hover:border-amber-core/60 text-ink-primary disabled:opacity-25 disabled:pointer-events-none flex items-center justify-center shadow-[0_8px_24px_rgba(0,0,0,0.55)] cursor-pointer transition-colors"
                  >
                    <span className="material-symbols-outlined text-[22px]">chevron_right</span>
                  </motion.button>
                </DarkroomTooltip>
              </div>
            )}
          </motion.div>

          {/* Synchronized Virtualized Bottom Thumbnail Carousel Strip */}
          <motion.footer
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={FAST_FADE}
            className="relative z-30 h-[92px] shrink-0 bg-surface-lowest/80 backdrop-blur-md border-t border-hairline/70 flex flex-col justify-center px-4"
          >
            <VirtualHorizontalStrip
              items={photos}
              activeIndex={clampedIndex}
              itemWidthPx={56}
              gapPx={8}
              centerActive
              getKey={(photo) => photo.id}
              className="w-full h-[68px] overflow-x-auto overflow-y-hidden"
              renderItem={(photo, idx) => {
                const isThumbActive = idx === clampedIndex;
                const isThumbSelected = selectedPhotoIds.has(photo.id);
                return (
                  <button
                    type="button"
                    onClick={() => goToIndex(idx)}
                    title={`${photo.fileName} (${idx + 1}/${photos.length})`}
                    className={`relative w-[56px] h-[58px] rounded-lg overflow-hidden shrink-0 border transition-all duration-150 cursor-pointer bg-surface ${
                      isThumbActive
                        ? 'border-amber-core scale-[1.06] ring-2 ring-amber-core/35 opacity-100 z-10'
                        : 'border-hairline hover:border-hairline-strong opacity-55 hover:opacity-95'
                    }`}
                  >
                    <GradedThumbnailImg
                      src={photo.imageUrl}
                      alt={photo.fileName}
                      recipe={photo.developRecipe}
                      loading="lazy"
                      decoding="async"
                      draggable={false}
                      className="w-full h-full object-cover"
                    />
                    {isThumbSelected && !photo.isSlateAnchor && (
                      <span className="absolute top-1 left-1 w-2 h-2 rounded-full bg-amber-core shadow" />
                    )}
                    {isThumbActive && (
                      <motion.span
                        layoutId="lightbox-active-thumb-bar"
                        transition={TACTILE_SPRING}
                        className="absolute bottom-0 inset-x-1 h-0.5 rounded-full bg-amber-core"
                      />
                    )}
                  </button>
                );
              }}
            />

            <div className="flex items-center justify-center gap-4 pb-1 tabular-nums-mono text-[10px] text-ink-telemetry">
              <span>← / → Navigate</span>
              <span>Drag to Swipe</span>
              <span>Double-Click / Z Zoom</span>
              <span>Space Select</span>
              <span>E Edit</span>
              <span>Esc Close</span>
            </div>
          </motion.footer>
        </div>
      )}
    </AnimatePresence>,
    document.body,
  );
}
