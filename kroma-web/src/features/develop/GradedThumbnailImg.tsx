import { memo, useId, useMemo, type CSSProperties, type ReactEventHandler } from 'react';
import type { DevelopRecipe } from '../../domain/develop';
import { buildDevelopSvgFilterTables } from '../../domain/developShaderMath';

interface GradedThumbnailImgProps {
  readonly src: string;
  /** Optional 32-bit Rust/wgpu rendered JPEG thumbnail matching `recipe` 1:1 (skips SVG approximation). */
  readonly renderedSrc?: string | null;
  readonly alt: string;
  readonly recipe: DevelopRecipe;
  readonly className?: string;
  readonly style?: CSSProperties;
  readonly loading?: 'eager' | 'lazy';
  readonly decoding?: 'sync' | 'async' | 'auto';
  readonly draggable?: boolean;
  readonly showVignette?: boolean;
  readonly onLoad?: ReactEventHandler<HTMLImageElement>;
}

/**
 * Thumbnail renderer supporting two modes seamlessly:
 * - When a 32-bit Rust/`wgpu` rendered JPEG (`renderedSrc`) is available, renders it directly at
 *   60–120 FPS with exact Studio parity (32-bit scene-linear RAW, AI backdrops, masks, HSL, 3D LUTs)
 *   and zero SVG filter stacking.
 * - Otherwise falls back to a lightweight GPU-composited per-channel SVG `<feComponentTransfer>`
 *   approximation without consuming WebGL/WebGPU contexts across 50+ grid/filmstrip cards.
 */
export const GradedThumbnailImg = memo(function GradedThumbnailImg({
  src,
  renderedSrc,
  alt,
  recipe,
  className = '',
  style,
  loading = 'lazy',
  decoding = 'async',
  draggable = false,
  showVignette = true,
  onLoad,
}: GradedThumbnailImgProps) {
  const rawId = useId();
  const filterId = `kroma-thumb-${rawId.replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const useRendered = Boolean(renderedSrc);
  const tables = useMemo(
    () => (useRendered ? null : buildDevelopSvgFilterTables(recipe, 33)),
    [useRendered, recipe],
  );

  const mergedStyle: CSSProperties | undefined = tables
    ? { ...style, filter: `url(#${filterId})` }
    : style;

  return (
    <>
      {tables && (
        <svg width="0" height="0" className="absolute pointer-events-none" aria-hidden="true">
          <defs>
            <filter id={filterId} colorInterpolationFilters="sRGB">
              <feComponentTransfer>
                <feFuncR type="table" tableValues={tables.rTable} />
                <feFuncG type="table" tableValues={tables.gTable} />
                <feFuncB type="table" tableValues={tables.bTable} />
              </feComponentTransfer>
              {Math.abs(tables.saturate - 1) > 0.005 && (
                <feColorMatrix type="saturate" values={tables.saturate.toFixed(3)} />
              )}
            </filter>
          </defs>
        </svg>
      )}
      <img
        src={renderedSrc || src}
        alt={alt}
        loading={loading}
        decoding={decoding}
        draggable={draggable}
        onLoad={onLoad}
        style={mergedStyle}
        className={className}
      />
      {!useRendered && showVignette && tables?.vignetteOverlay && (
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ background: tables.vignetteOverlay }}
        />
      )}
    </>
  );
});
