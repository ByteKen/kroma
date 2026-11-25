import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { isRecipeModified } from '../../domain/develop';
import type { PhotoAsset } from '../../domain/photo';
import {
  computeHoverPreviewPosition,
  type AnchorRectLike,
} from '../slates/SubjectHoverPreviewCard';
import { GradedThumbnailImg } from './GradedThumbnailImg';

const HOVER_ENTER_DELAY_MS = 130;
const HOVER_GLIDE_SETTLE_MS = 60;
const POPOVER_WIDTH_PX = 204;
const POPOVER_HEIGHT_PX = 252;

export interface PhotoHoverPreviewState {
  readonly photo: PhotoAsset;
  readonly rect: AnchorRectLike;
}

/**
 * Low-overhead hover-intent controller for filename rows inside expanded Folders:
 * - 130ms intent delay from idle + 60ms glide settle so sweeping rapidly down 30+ filenames
 *   mounts zero `<img>` nodes and triggers zero thumbnail decodes.
 * - Cancels immediately on mouse leave, sidebar scroll, window blur, or photo click.
 */
export function usePhotoNameHoverPreview() {
  const [preview, setPreview] = useState<PhotoHoverPreviewState | null>(null);
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

  const schedulePreview = useCallback((photo: PhotoAsset, element: HTMLElement) => {
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

    const delay = isOpenRef.current ? HOVER_GLIDE_SETTLE_MS : HOVER_ENTER_DELAY_MS;
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      isOpenRef.current = true;
      setPreview({ photo, rect });
    }, delay);
  }, []);

  useEffect(() => {
    if (!preview) return;
    window.addEventListener('blur', clearPreview);
    return () => {
      window.removeEventListener('blur', clearPreview);
    };
  }, [preview, clearPreview]);

  useEffect(() => clearPreview, [clearPreview]);

  return { preview, schedulePreview, clearPreview };
}

interface PhotoNameHoverPreviewProps {
  readonly photo: PhotoAsset | null;
  readonly anchorRect: AnchorRectLike | null;
}

/**
 * Ultra-light single-image floating preview for hovered photo filenames in Studio's Folders tree.
 * Reuses the already-cached 640px/1024px JPEG preview with async off-thread decode (`decoding="async"`),
 * avoids `backdrop-filter` recompositing over the WebGPU canvas, and triggers zero RAW IPC calls.
 */
export const PhotoNameHoverPreview = memo(function PhotoNameHoverPreview({
  photo,
  anchorRect,
}: PhotoNameHoverPreviewProps) {
  if (typeof document === 'undefined' || !photo || !anchorRect) {
    return null;
  }

  const vw = typeof window !== 'undefined' ? window.innerWidth : 1280;
  const vh = typeof window !== 'undefined' ? window.innerHeight : 800;
  const position = computeHoverPreviewPosition(
    anchorRect,
    vw,
    vh,
    POPOVER_WIDTH_PX,
    POPOVER_HEIGHT_PX,
  );
  const edited = isRecipeModified(photo.developRecipe);

  return createPortal(
    <aside
      aria-hidden="true"
      style={{
        width: `${POPOVER_WIDTH_PX}px`,
        left: `${position.left}px`,
        top: `${position.top}px`,
      }}
      className="fixed z-[60] pointer-events-none rounded-xl bg-surface-elevated border border-hairline-strong shadow-[0_14px_36px_rgba(0,0,0,0.72)] overflow-hidden flex flex-col"
    >
      <div className="relative h-[206px] w-full bg-surface-lowest flex items-center justify-center overflow-hidden p-1.5">
        <GradedThumbnailImg
          src={photo.imageUrl}
          renderedSrc={photo.renderedThumbnailUrl}
          alt={photo.fileName}
          recipe={photo.developRecipe}
          loading="eager"
          decoding="async"
          className="w-full h-full object-contain rounded-md"
        />
        {photo.isSlateAnchor && (
          <span className="absolute top-2 left-2 px-1.5 py-0.5 rounded bg-canvas/85 border border-hairline tabular-nums-mono text-[8.5px] text-ink-secondary">
            SLATE
          </span>
        )}
        {edited && (
          <span className="absolute bottom-2 right-2 px-1.5 py-0.5 rounded bg-canvas/85 border border-amber-core/50 tabular-nums-mono text-[8.5px] text-amber-soft">
            EDITED
          </span>
        )}
      </div>
      <div className="px-2.5 py-1.5 bg-surface-low border-t border-hairline flex items-center justify-between gap-2 tabular-nums-mono text-[10px]">
        <span className="truncate text-ink-primary">{photo.fileName}</span>
        {photo.capturedAtLabel && (
          <span className="shrink-0 text-ink-telemetry">{photo.capturedAtLabel}</span>
        )}
      </div>
    </aside>,
    document.body,
  );
});
