import { AnimatePresence, motion } from 'framer-motion';
import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { isRecipeModified } from '../../domain/develop';
import { isFaceOnlySlateCode, type SubjectCluster } from '../../domain/identity';
import { FAST_FADE, TACTILE_SPRING, TelemetryPill } from '../darkroom-kit';
import { GradedThumbnailImg } from '../develop/GradedThumbnailImg';

const HOVER_OPEN_DELAY_MS = 110;

export interface AnchorRectLike {
  readonly left: number;
  readonly right: number;
  readonly top: number;
  readonly bottom: number;
  readonly width: number;
  readonly height: number;
}

export interface PopoverPosition {
  readonly left: number;
  readonly top: number;
}

export interface SubjectHoverPreviewState {
  readonly cluster: SubjectCluster;
  readonly rect: AnchorRectLike;
}

/**
 * Shared hover-intent controller for subject folder lists:
 * - Short 110ms intent delay when entering from idle (prevents flashing on fast pointer sweeps).
 * - Instant glide when moving directly between adjacent subject rows.
 */
export function useSubjectHoverPreview() {
  const [preview, setPreview] = useState<SubjectHoverPreviewState | null>(null);
  const isOpenRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const clearPreview = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    isOpenRef.current = false;
    setPreview(null);
  }, []);

  const schedulePreview = useCallback((cluster: SubjectCluster, element: HTMLElement) => {
    const raw = element.getBoundingClientRect();
    const rect: AnchorRectLike = {
      left: raw.left,
      right: raw.right,
      top: raw.top,
      bottom: raw.bottom,
      width: raw.width,
      height: raw.height,
    };

    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    if (isOpenRef.current) {
      setPreview({ cluster, rect });
      return;
    }

    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      isOpenRef.current = true;
      setPreview({ cluster, rect });
    }, HOVER_OPEN_DELAY_MS);
  }, []);

  useEffect(() => {
    if (!preview) return;
    window.addEventListener('dragend', clearPreview);
    window.addEventListener('blur', clearPreview);
    return () => {
      window.removeEventListener('dragend', clearPreview);
      window.removeEventListener('blur', clearPreview);
    };
  }, [preview, clearPreview]);

  useEffect(() => clearPreview, [clearPreview]);

  return { preview, schedulePreview, clearPreview };
}

const POPOVER_WIDTH_PX = 272;
const POPOVER_GAP_PX = 12;
const TOP_SAFE_MARGIN_PX = 58;
const BOTTOM_SAFE_MARGIN_PX = 48;
const EDGE_SAFE_MARGIN_PX = 12;

/**
 * Places the subject hover preview cleanly to the right of the hovered sidebar card,
 * vertically centered on the row and clamped inside the darkroom workspace viewport.
 */
export function computeHoverPreviewPosition(
  anchorRect: AnchorRectLike,
  viewportWidth: number,
  viewportHeight: number,
  cardWidth: number = POPOVER_WIDTH_PX,
  cardHeight: number = 408,
): PopoverPosition {
  const rightCandidate = anchorRect.right + POPOVER_GAP_PX;
  const fitsRight = rightCandidate + cardWidth + EDGE_SAFE_MARGIN_PX <= viewportWidth;
  const left = fitsRight
    ? rightCandidate
    : Math.max(EDGE_SAFE_MARGIN_PX, anchorRect.left - cardWidth - POPOVER_GAP_PX);

  const rowCenterY = anchorRect.top + anchorRect.height / 2;
  const idealTop = rowCenterY - cardHeight / 2;
  const maxTop = Math.max(TOP_SAFE_MARGIN_PX, viewportHeight - cardHeight - BOTTOM_SAFE_MARGIN_PX);
  const top = Math.round(Math.max(TOP_SAFE_MARGIN_PX, Math.min(maxTop, idealTop)));

  return { left: Math.round(left), top };
}

interface SubjectHoverPreviewCardProps {
  readonly cluster: SubjectCluster | null;
  readonly anchorRect: AnchorRectLike | null;
}

/**
 * Floating obsidian darkroom hover preview for subject folders in the sidebar.
 * Rendered in a non-blocking portal (`pointer-events-none`) so scrolling, clicking,
 * and drag-and-drop onto subject cards remain completely unobstructed.
 */
export function SubjectHoverPreviewCard({
  cluster,
  anchorRect,
}: SubjectHoverPreviewCardProps) {
  if (typeof document === 'undefined') return null;

  const isVisible = Boolean(cluster && anchorRect);
  const hasMultiplePhotos = Boolean(cluster && cluster.photos.length > 1);
  const estimatedHeight = hasMultiplePhotos ? 408 : 338;

  const vw = typeof window !== 'undefined' ? window.innerWidth : 1280;
  const vh = typeof window !== 'undefined' ? window.innerHeight : 800;
  const position = anchorRect
    ? computeHoverPreviewPosition(anchorRect, vw, vh, POPOVER_WIDTH_PX, estimatedHeight)
    : { left: 290, top: 80 };

  const heroPhoto = cluster
    ? cluster.photos.find((p) => p.id === cluster.anchor.anchorPhotoId) ?? cluster.photos[0] ?? null
    : null;
  const previewStripPhotos = cluster ? cluster.photos.slice(0, 4) : [];
  const overflowCount = cluster ? Math.max(0, cluster.photos.length - 4) : 0;
  const faceOnly = cluster ? isFaceOnlySlateCode(cluster.anchor.slateCode) : true;
  const isSpecialBucket = cluster ? cluster.anchor.id.startsWith('_') : false;
  const heroEdited = heroPhoto ? isRecipeModified(heroPhoto.developRecipe) : false;

  return createPortal(
    <AnimatePresence>
      {isVisible && cluster && (
        <motion.aside
          key="subject-hover-preview"
          aria-hidden="true"
          initial={{ opacity: 0, x: -8, scale: 0.96, top: position.top, left: position.left }}
          animate={{ opacity: 1, x: 0, scale: 1, top: position.top, left: position.left }}
          exit={{ opacity: 0, x: -6, scale: 0.97, transition: FAST_FADE }}
          transition={TACTILE_SPRING}
          style={{ width: `${POPOVER_WIDTH_PX}px` }}
          className="fixed z-[60] pointer-events-none rounded-xl bg-surface-elevated/96 backdrop-blur-xl border border-hairline-strong shadow-[0_18px_50px_rgba(0,0,0,0.82)] overflow-hidden flex flex-col"
        >
          {/* Compact Subject Header */}
          <div className="px-3 py-2 bg-surface-low/95 border-b border-hairline flex items-center justify-between gap-2">
            <div className="min-w-0 flex items-center gap-1.5">
              <span className="font-display text-[13px] font-bold text-ink-primary truncate">
                {cluster.anchor.subjectName}
              </span>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              {!isSpecialBucket && !faceOnly && (
                <TelemetryPill tone="amber">{cluster.anchor.slateCode}</TelemetryPill>
              )}
              <TelemetryPill tone={cluster.anchor.ambiguousCount > 0 ? 'tangerine' : 'neutral'}>
                {cluster.photos.length} {cluster.photos.length === 1 ? 'photo' : 'photos'}
              </TelemetryPill>
            </div>
          </div>

          {/* Full Uncropped Hero Portrait Preview */}
          <div className="relative h-[268px] w-full bg-surface-lowest flex items-center justify-center overflow-hidden p-1.5">
            {heroPhoto ? (
              <GradedThumbnailImg
                src={heroPhoto.imageUrl}
                renderedSrc={heroPhoto.renderedThumbnailUrl}
                alt={cluster.anchor.subjectName}
                recipe={heroPhoto.developRecipe}
                loading="eager"
                decoding="async"
                className="w-full h-full object-contain rounded-lg"
              />
            ) : (
              <img
                src={cluster.anchor.thumbnailUrl}
                alt={cluster.anchor.subjectName}
                decoding="async"
                className="w-full h-full object-contain rounded-lg"
              />
            )}

            {heroEdited && (
              <span className="absolute bottom-2.5 right-2.5 px-1.5 py-0.5 rounded bg-surface-lowest/85 backdrop-blur border border-amber-core/50 tabular-nums-mono text-[9px] text-amber-soft">
                EDITED
              </span>
            )}
          </div>

          {/* Mini 4-Up Contact Sheet Strip when Subject Contains Multiple Photos */}
          {hasMultiplePhotos && (
            <div className="px-2.5 pt-2 pb-1.5 bg-surface border-t border-hairline grid grid-cols-4 gap-1.5">
              {previewStripPhotos.map((photo, idx) => {
                const isLastOverflowTile = idx === 3 && overflowCount > 0;
                return (
                  <div
                    key={photo.id}
                    className={`relative aspect-[3/4] rounded overflow-hidden bg-surface-lowest border ${
                      photo.id === heroPhoto?.id ? 'border-amber-core/70' : 'border-hairline'
                    }`}
                  >
                    <GradedThumbnailImg
                      src={photo.imageUrl}
                      renderedSrc={photo.renderedThumbnailUrl}
                      alt={photo.fileName}
                      recipe={photo.developRecipe}
                      decoding="async"
                      className="w-full h-full object-cover"
                    />
                    {isLastOverflowTile && (
                      <div className="absolute inset-0 bg-canvas/75 backdrop-blur-[1px] flex items-center justify-center tabular-nums-mono text-[11px] font-bold text-ink-primary">
                        +{overflowCount}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Subtle Anchor Filename & Capture Timestamp Footer */}
          <div className="px-3 py-1.5 bg-surface-low/95 border-t border-hairline flex items-center justify-between gap-2 tabular-nums-mono text-[10px] text-ink-telemetry">
            <span className="truncate text-ink-secondary">
              {heroPhoto?.fileName ?? 'Anchor preview'}
            </span>
            <span className="shrink-0">
              {heroPhoto?.capturedAtLabel ??
                (cluster.anchor.ambiguousCount > 0
                  ? `${cluster.anchor.ambiguousCount} review`
                  : faceOnly
                    ? 'Face-Only'
                    : 'Slate')}
            </span>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>,
    document.body,
  );
}
