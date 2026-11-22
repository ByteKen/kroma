import { memo, useCallback, useRef, useState } from 'react';
import {
  FULL_FRAME_CROP_RECT,
  type CropAspectRatio,
  type CropRect,
  type CropTransformSettings,
} from '../../domain/develop';

type DragHandle = 'move' | 'nw' | 'ne' | 'sw' | 'se' | 'n' | 's' | 'e' | 'w';

interface CropOverlayProps {
  readonly crop: CropTransformSettings;
  readonly naturalAspect: number;
  readonly onChange: (next: CropTransformSettings) => void;
}

const ASPECT_RATIO_VALUES: Record<Exclude<CropAspectRatio, 'original' | 'free'>, number> = {
  '4:5': 4 / 5,
  '2:3': 2 / 3,
  '5:7': 5 / 7,
  '1:1': 1,
  '16:9': 16 / 9,
};

const MIN_CROP_SPAN = 0.1;

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

function targetNormalizedRatio(aspect: CropAspectRatio, naturalAspect: number): number | null {
  if (aspect === 'free') return null;
  if (aspect === 'original') return 1; // width_norm / height_norm = 1 preserves original aspect
  const desired = ASPECT_RATIO_VALUES[aspect];
  return desired / Math.max(0.1, naturalAspect);
}

export function fitCenteredCropForAspect(
  aspect: CropAspectRatio,
  naturalAspect: number,
): CropRect {
  if (aspect === 'original' || aspect === 'free') {
    return { ...FULL_FRAME_CROP_RECT };
  }
  const normRatio = targetNormalizedRatio(aspect, naturalAspect);
  if (!normRatio) return { ...FULL_FRAME_CROP_RECT };

  let width = 1;
  let height = 1;
  if (normRatio < 1) {
    width = normRatio;
  } else {
    height = 1 / normRatio;
  }
  return {
    left: Number(((1 - width) * 0.5).toFixed(4)),
    top: Number(((1 - height) * 0.5).toFixed(4)),
    width: Number(width.toFixed(4)),
    height: Number(height.toFixed(4)),
  };
}

/**
 * Interactive Lightroom-style Rule-of-Thirds crop overlay with 4 corner brackets,
 * 4 edge handles (N/S/E/W), interior pan, and optional aspect-ratio locking.
 */
export const CropOverlay = memo(function CropOverlay({
  crop,
  naturalAspect,
  onChange,
}: CropOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    handle: DragHandle;
    startX: number;
    startY: number;
    startRect: CropRect;
  } | null>(null);
  const [activeHandle, setActiveHandle] = useState<DragHandle | null>(null);

  const rect = crop.rect ?? FULL_FRAME_CROP_RECT;

  const beginDrag = (handle: DragHandle, e: React.PointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = {
      handle,
      startX: e.clientX,
      startY: e.clientY,
      startRect: { ...rect },
    };
    setActiveHandle(handle);
  };

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      const drag = dragRef.current;
      const box = containerRef.current?.getBoundingClientRect();
      if (!drag || !box || box.width <= 0 || box.height <= 0) return;

      const dx = (e.clientX - drag.startX) / box.width;
      const dy = (e.clientY - drag.startY) / box.height;
      const sr = drag.startRect;

      if (drag.handle === 'move') {
        const left = clamp(sr.left + dx, 0, 1 - sr.width);
        const top = clamp(sr.top + dy, 0, 1 - sr.height);
        onChange({
          ...crop,
          rect: {
            left: Number(left.toFixed(4)),
            top: Number(top.toFixed(4)),
            width: sr.width,
            height: sr.height,
          },
        });
        return;
      }

      let left = sr.left;
      let top = sr.top;
      let right = sr.left + sr.width;
      let bottom = sr.top + sr.height;

      const movesW = drag.handle === 'w' || drag.handle === 'nw' || drag.handle === 'sw';
      const movesE = drag.handle === 'e' || drag.handle === 'ne' || drag.handle === 'se';
      const movesN = drag.handle === 'n' || drag.handle === 'nw' || drag.handle === 'ne';
      const movesS = drag.handle === 's' || drag.handle === 'sw' || drag.handle === 'se';

      if (movesW) left = clamp(sr.left + dx, 0, right - MIN_CROP_SPAN);
      if (movesE) right = clamp(sr.left + sr.width + dx, left + MIN_CROP_SPAN, 1);
      if (movesN) top = clamp(sr.top + dy, 0, bottom - MIN_CROP_SPAN);
      if (movesS) bottom = clamp(sr.top + sr.height + dy, top + MIN_CROP_SPAN, 1);

      let width = right - left;
      let height = bottom - top;

      const lockedRatio = targetNormalizedRatio(crop.aspectRatio, naturalAspect);
      if (lockedRatio) {
        const isVerticalEdgeOnly = drag.handle === 'n' || drag.handle === 's';
        const isHorizontalEdgeOnly = drag.handle === 'e' || drag.handle === 'w';

        if (isVerticalEdgeOnly) {
          const centerX = sr.left + sr.width * 0.5;
          width = height * lockedRatio;
          if (width > 1) {
            width = 1;
            height = width / lockedRatio;
            if (movesN) top = bottom - height;
            else bottom = top + height;
          }
          left = clamp(centerX - width * 0.5, 0, 1 - width);
        } else if (isHorizontalEdgeOnly) {
          const centerY = sr.top + sr.height * 0.5;
          height = width / lockedRatio;
          if (height > 1) {
            height = 1;
            width = height * lockedRatio;
            if (movesW) left = right - width;
            else right = left + width;
          }
          top = clamp(centerY - height * 0.5, 0, 1 - height);
        } else {
          // Corner handle: derive height from width and keep opposite corner anchored
          height = width / lockedRatio;
          if (movesN) {
            top = bottom - height;
            if (top < 0) {
              top = 0;
              height = bottom;
              width = height * lockedRatio;
              if (movesW) left = right - width;
            }
          } else if (top + height > 1) {
            height = 1 - top;
            width = height * lockedRatio;
            if (movesW) left = right - width;
          }
        }

        left = clamp(left, 0, 1 - MIN_CROP_SPAN);
        top = clamp(top, 0, 1 - MIN_CROP_SPAN);
        width = clamp(width, MIN_CROP_SPAN, 1 - left);
        height = clamp(height, MIN_CROP_SPAN, 1 - top);
      }

      onChange({
        ...crop,
        rect: {
          left: Number(left.toFixed(4)),
          top: Number(top.toFixed(4)),
          width: Number(width.toFixed(4)),
          height: Number(height.toFixed(4)),
        },
      });
    },
    [crop, naturalAspect, onChange],
  );

  const endDrag = (e: React.PointerEvent) => {
    dragRef.current = null;
    setActiveHandle(null);
    if ((e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId)) {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    }
  };

  return (
    <div
      ref={containerRef}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      className="absolute inset-0 z-30 select-none touch-none"
    >
      {/* Dimmed Exterior Mask */}
      <div
        className="absolute inset-x-0 top-0 bg-black/62 pointer-events-none"
        style={{ height: `${rect.top * 100}%` }}
      />
      <div
        className="absolute inset-x-0 bottom-0 bg-black/62 pointer-events-none"
        style={{ height: `${Math.max(0, (1 - rect.top - rect.height) * 100)}%` }}
      />
      <div
        className="absolute left-0 bg-black/62 pointer-events-none"
        style={{
          top: `${rect.top * 100}%`,
          height: `${rect.height * 100}%`,
          width: `${rect.left * 100}%`,
        }}
      />
      <div
        className="absolute right-0 bg-black/62 pointer-events-none"
        style={{
          top: `${rect.top * 100}%`,
          height: `${rect.height * 100}%`,
          width: `${Math.max(0, (1 - rect.left - rect.width) * 100)}%`,
        }}
      />

      {/* Interactive Crop Frame + Rule-of-Thirds Grid */}
      <div
        onPointerDown={(e) => beginDrag('move', e)}
        style={{
          left: `${rect.left * 100}%`,
          top: `${rect.top * 100}%`,
          width: `${rect.width * 100}%`,
          height: `${rect.height * 100}%`,
        }}
        className={`absolute border border-amber-core/90 shadow-[0_0_0_1px_rgba(0,0,0,0.65)] ${
          activeHandle === 'move' ? 'cursor-grabbing' : 'cursor-move'
        }`}
      >
        {/* Rule of Thirds Guides */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute left-1/3 inset-y-0 w-px bg-white/30" />
          <div className="absolute left-2/3 inset-y-0 w-px bg-white/30" />
          <div className="absolute top-1/3 inset-x-0 h-px bg-white/30" />
          <div className="absolute top-2/3 inset-x-0 h-px bg-white/30" />
        </div>

        {/* 4 Lightroom Corner Bracket Handles */}
        <div
          onPointerDown={(e) => beginDrag('nw', e)}
          className="absolute -top-1 -left-1 w-4 h-4 cursor-nwse-resize"
        >
          <span className="block w-full h-full border-t-[3px] border-l-[3px] border-amber-core rounded-tl-[2px]" />
        </div>
        <div
          onPointerDown={(e) => beginDrag('ne', e)}
          className="absolute -top-1 -right-1 w-4 h-4 cursor-nesw-resize"
        >
          <span className="block w-full h-full border-t-[3px] border-r-[3px] border-amber-core rounded-tr-[2px]" />
        </div>
        <div
          onPointerDown={(e) => beginDrag('sw', e)}
          className="absolute -bottom-1 -left-1 w-4 h-4 cursor-nesw-resize"
        >
          <span className="block w-full h-full border-b-[3px] border-l-[3px] border-amber-core rounded-bl-[2px]" />
        </div>
        <div
          onPointerDown={(e) => beginDrag('se', e)}
          className="absolute -bottom-1 -right-1 w-4 h-4 cursor-nwse-resize"
        >
          <span className="block w-full h-full border-b-[3px] border-r-[3px] border-amber-core rounded-br-[2px]" />
        </div>

        {/* 4 Edge Handles (Top / Bottom / Left / Right) */}
        <div
          onPointerDown={(e) => beginDrag('n', e)}
          className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-8 h-3 flex items-center justify-center cursor-ns-resize"
        >
          <span className="w-6 h-[3px] rounded-full bg-amber-core shadow" />
        </div>
        <div
          onPointerDown={(e) => beginDrag('s', e)}
          className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-8 h-3 flex items-center justify-center cursor-ns-resize"
        >
          <span className="w-6 h-[3px] rounded-full bg-amber-core shadow" />
        </div>
        <div
          onPointerDown={(e) => beginDrag('w', e)}
          className="absolute top-1/2 -left-1.5 -translate-y-1/2 w-3 h-8 flex items-center justify-center cursor-ew-resize"
        >
          <span className="w-[3px] h-6 rounded-full bg-amber-core shadow" />
        </div>
        <div
          onPointerDown={(e) => beginDrag('e', e)}
          className="absolute top-1/2 -right-1.5 -translate-y-1/2 w-3 h-8 flex items-center justify-center cursor-ew-resize"
        >
          <span className="w-[3px] h-6 rounded-full bg-amber-core shadow" />
        </div>
      </div>
    </div>
  );
});
