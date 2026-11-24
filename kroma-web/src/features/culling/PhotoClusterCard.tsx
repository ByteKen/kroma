import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useMemo, useRef, useState, type CSSProperties, type SyntheticEvent } from 'react';
import { isRecipeModified } from '../../domain/develop';
import { formatSimilarityPercent, needsVisualMatchReview } from '../../domain/identity';
import type { PhotoAsset } from '../../domain/photo';
import { DarkroomTooltip, FAST_FADE, TACTILE_SPRING, TelemetryPill } from '../darkroom-kit';
import { GradedThumbnailImg } from '../develop/GradedThumbnailImg';
import { beginKromaDrag, clearActiveKromaDrag } from '../folders/kromaDragState';
import type { RectLike } from './lightboxGeometry';

const CARD_STAGE_ASPECT = 3 / 4;

/**
 * Computes the CSS rectangle matching an `object-cover` image inside the `aspect-[3/4]` stage
 * so normalized `[0..100%]` YuNet face boxes and 5-point landmarks sit pixel-exact on every photo aspect ratio.
 */
function computeCoverOverlayStyle(naturalAspect: number | null): CSSProperties {
  if (!naturalAspect || !Number.isFinite(naturalAspect) || naturalAspect <= 0) {
    return { inset: 0 };
  }
  if (naturalAspect < CARD_STAGE_ASPECT) {
    const heightPct = (CARD_STAGE_ASPECT / naturalAspect) * 100;
    const topPct = -((heightPct - 100) / 2);
    return { left: '0%', top: `${topPct}%`, width: '100%', height: `${heightPct}%` };
  }
  const widthPct = (naturalAspect / CARD_STAGE_ASPECT) * 100;
  const leftPct = -((widthPct - 100) / 2);
  return { left: `${leftPct}%`, top: '0%', width: `${widthPct}%`, height: '100%' };
}

interface PhotoClusterCardProps {
  readonly photo: PhotoAsset;
  readonly isSelected: boolean;
  readonly dragPhotoIds?: readonly string[];
  readonly virtualFolderLabel?: string | null;
  readonly isClustered: boolean;
  readonly reviewWarningThreshold: number;
  readonly isReviewApproved?: boolean;
  readonly showLandmarksOverlay: boolean;
  readonly showMetadata?: boolean;
  readonly onToggleSelect: (photoId: string, options?: { readonly shiftKey?: boolean }) => void;
  readonly onOpenLightbox: (photoId: string, originRect: RectLike | null) => void;
  readonly onOpenDevelop: (photoId: string) => void;
  readonly onManualMatch?: (photo: PhotoAsset) => void;
}

export const PhotoClusterCard = memo(function PhotoClusterCard({
  photo,
  isSelected,
  dragPhotoIds,
  virtualFolderLabel = null,
  isClustered,
  reviewWarningThreshold,
  isReviewApproved = false,
  showLandmarksOverlay,
  showMetadata = true,
  onToggleSelect,
  onOpenLightbox,
  onOpenDevelop,
  onManualMatch,
}: PhotoClusterCardProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [naturalAspect, setNaturalAspect] = useState<number | null>(null);
  const isAmbiguous =
    !isReviewApproved && needsVisualMatchReview(photo, reviewWarningThreshold);
  const edited = isRecipeModified(photo.developRecipe);
  const rotation = photo.developRecipe.cropTransform.rotationDeg;
  const coverOverlayStyle = computeCoverOverlayStyle(naturalAspect);

  const handleImgLoad = useCallback((e: SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
      const nextAspect = img.naturalWidth / img.naturalHeight;
      setNaturalAspect((prev) => (prev === nextAspect ? prev : nextAspect));
    }
  }, []);

  const imgStyle = useMemo<CSSProperties | undefined>(
    () =>
      !photo.renderedThumbnailUrl && rotation !== 0
        ? { transform: `rotate(${rotation}deg) scale(1.05)` }
        : undefined,
    [photo.renderedThumbnailUrl, rotation],
  );

  const cardBorderClass = photo.isSlateAnchor
    ? 'border-amber-core/60'
    : isSelected
      ? 'border-amber-core ring-1 ring-amber-core/40'
      : isAmbiguous
        ? 'border-tangerine-alert/60'
        : 'border-hairline hover:border-hairline-strong';

  const handleOpenLightbox = () => {
    const rect = stageRef.current?.getBoundingClientRect() ?? null;
    onOpenLightbox(
      photo.id,
      rect ? { left: rect.left, top: rect.top, width: rect.width, height: rect.height } : null,
    );
  };

  return (
    <article
      onClick={(e) => {
        onToggleSelect(photo.id, { shiftKey: e.shiftKey });
      }}
      onDoubleClick={handleOpenLightbox}
      className={`group relative flex flex-col rounded-lg overflow-hidden bg-surface border select-none cursor-pointer transition-colors ${cardBorderClass}`}
    >
      {/* Image Stage (always draggable onto any virtual folder or subject in the left sidebar) */}
      <div
        ref={stageRef}
        data-photo-stage-id={photo.id}
        draggable
        onDragStart={(e) => {
          const payload = dragPhotoIds && dragPhotoIds.length > 0 ? dragPhotoIds : [photo.id];
          beginKromaDrag(e.dataTransfer, { kind: 'photos', photoIds: payload });
        }}
        onDragEnd={() => {
          clearActiveKromaDrag();
        }}
        className="relative aspect-[3/4] bg-surface-lowest overflow-hidden active:cursor-grabbing"
      >
        <GradedThumbnailImg
          src={photo.imageUrl}
          renderedSrc={photo.renderedThumbnailUrl}
          alt={photo.fileName}
          recipe={photo.developRecipe}
          draggable={false}
          loading="lazy"
          decoding="async"
          onLoad={handleImgLoad}
          style={imgStyle}
          className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-[1.02]"
        />

        {/* Selection Checkbox (always available so any photo, including slates, can be multi-selected & dragged to folders) */}
        <div className="absolute top-2 left-2 z-10">
          <motion.button
            type="button"
            whileTap={{ scale: 0.86 }}
            transition={TACTILE_SPRING}
            onClick={(e) => {
              e.stopPropagation();
              onToggleSelect(photo.id, { shiftKey: e.shiftKey });
            }}
            onDoubleClick={(e) => e.stopPropagation()}
            aria-label={`Select ${photo.fileName}`}
            className={`w-4 h-4 rounded flex items-center justify-center border transition-colors cursor-pointer ${
              isSelected
                ? 'bg-amber-core border-amber-core text-amber-ink'
                : 'bg-surface-lowest/80 border-hairline-strong hover:border-amber-core'
            }`}
          >
            {isSelected && (
              <span className="material-symbols-outlined text-[13px] font-bold">check</span>
            )}
          </motion.button>
        </div>

        {photo.isSlateAnchor ? (
          <div className="absolute top-2 right-2 z-10">
            <span className="px-2 py-0.5 rounded bg-amber-core text-amber-ink font-display text-[10px] font-bold tracking-wider uppercase">
              SLATE
            </span>
          </div>
        ) : (
          <>
            {/* Optional YuNet 5-Point Face Box (aligned to exact object-cover image projection) */}
            <AnimatePresence>
              {showLandmarksOverlay && photo.faceOverlay && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={FAST_FADE}
                  style={coverOverlayStyle}
                  className="absolute pointer-events-none"
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
                      className={`absolute top-0 left-0 w-2.5 h-2.5 border-t-2 border-l-2 ${
                        isAmbiguous ? 'border-tangerine-alert' : 'border-emerald-focus'
                      }`}
                    />
                    <div
                      className={`absolute top-0 right-0 w-2.5 h-2.5 border-t-2 border-r-2 ${
                        isAmbiguous ? 'border-tangerine-alert' : 'border-emerald-focus'
                      }`}
                    />
                    <div
                      className={`absolute bottom-0 left-0 w-2.5 h-2.5 border-b-2 border-l-2 ${
                        isAmbiguous ? 'border-tangerine-alert' : 'border-emerald-focus'
                      }`}
                    />
                    <div
                      className={`absolute bottom-0 right-0 w-2.5 h-2.5 border-b-2 border-r-2 ${
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
                          idx === 2 ? 'w-1 h-1' : 'w-1.5 h-1.5'
                        } ${isAmbiguous ? 'bg-tangerine-alert' : 'bg-emerald-bright'}`}
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Similarity Match Pill (only shown after face clustering) */}
            {showMetadata && photo.faceOverlay && (
              <div className="absolute top-2 right-2 z-10">
                <TelemetryPill
                  tone={isAmbiguous ? 'tangerine' : 'emerald'}
                  dot
                  className="bg-surface-lowest/85 backdrop-blur"
                >
                  {formatSimilarityPercent(photo.sfaceCosineSimilarity)}
                </TelemetryPill>
              </div>
            )}
          </>
        )}

        {/* Assigned Virtual Department Folder Pill */}
        {showMetadata && virtualFolderLabel && (
          <div className="absolute bottom-2 left-2 z-10 max-w-[calc(100%-68px)] px-1.5 py-0.5 rounded bg-surface-lowest/88 backdrop-blur border border-hairline-strong flex items-center gap-1 tabular-nums-mono text-[9.5px] text-ink-primary">
            <span className="material-symbols-outlined text-[11px] text-amber-soft shrink-0">
              folder
            </span>
            <span className="truncate">{virtualFolderLabel}</span>
          </div>
        )}

        {/* Edited Indicator Badge */}
        {showMetadata && edited && (
          <div className="absolute bottom-2 right-2 z-10 px-1.5 py-0.5 rounded bg-surface-lowest/85 backdrop-blur border border-amber-core/50 tabular-nums-mono text-[9px] text-amber-soft">
            EDITED
          </div>
        )}

        {/* Quiet Hover Edit Action When Metadata Footer Is Hidden */}
        {!showMetadata && (
          <div
            className="absolute bottom-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={(e) => e.stopPropagation()}
            onDoubleClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => onOpenDevelop(photo.id)}
              className="px-2 py-1 rounded-md bg-surface-lowest/90 backdrop-blur border border-hairline-strong hover:border-amber-core/70 tabular-nums-mono text-[10px] text-ink-primary hover:text-amber-soft cursor-pointer transition-colors"
            >
              Edit
            </button>
          </div>
        )}
      </div>

      {/* Clean Minimal Footer (Capture Time + Manual Match + Edit) */}
      {showMetadata && (
        <div className="px-2.5 py-2 bg-surface flex items-center justify-between gap-1.5 border-t border-hairline">
          <div className="min-w-0 flex flex-col">
            <span className="tabular-nums-mono text-[11px] font-medium text-ink-primary truncate">
              {photo.fileName}
            </span>
            <span className="tabular-nums-mono text-[10px] text-ink-telemetry truncate">
              {photo.isSlateAnchor
                ? 'Slate Reference'
                : edited
                  ? 'Graded (.kroma.json)'
                  : photo.capturedAtLabel
                    ? `${photo.exif.shutterSpeed} • ${photo.capturedAtLabel}`
                    : photo.sharpnessPct < 65
                      ? `Soft Focus (${photo.sharpnessPct}%)`
                      : `${photo.exif.shutterSpeed}`}
            </span>
          </div>

          <div
            className="flex items-center gap-1 shrink-0"
            onClick={(e) => e.stopPropagation()}
            onDoubleClick={(e) => e.stopPropagation()}
          >
            {isClustered && onManualMatch && (
              <DarkroomTooltip content="Manual match" side="top">
                <button
                  type="button"
                  onClick={() => onManualMatch(photo)}
                  className="px-1.5 py-1 rounded-md bg-canvas border border-hairline hover:border-emerald-focus/60 tabular-nums-mono text-[10px] text-ink-secondary hover:text-emerald-bright cursor-pointer transition-colors"
                >
                  Match
                </button>
              </DarkroomTooltip>
            )}

            <DarkroomTooltip content="Edit in Develop" side="top">
              <button
                type="button"
                onClick={() => onOpenDevelop(photo.id)}
                className="px-2 py-1 rounded-md bg-canvas border border-hairline hover:border-amber-core/60 tabular-nums-mono text-[10px] text-ink-secondary hover:text-amber-soft cursor-pointer transition-colors"
              >
                Edit
              </button>
            </DarkroomTooltip>
          </div>
        </div>
      )}
    </article>
  );
});
