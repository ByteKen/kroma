import { memo, useId, useMemo, type CSSProperties, type ReactEventHandler } from 'react';
import type { DevelopRecipe } from '../../domain/develop';
import { buildDevelopSvgFilterTables } from '../../domain/developShaderMath';

interface GradedThumbnailImgProps {
  readonly src: string;
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
 * Lightweight GPU-composited thumbnail renderer backed by per-channel SVG `<feComponentTransfer>`
 * + `<feColorMatrix>` filters matching Rust `color_math.rs` (exact cool/warm Kelvin WB, Tint,
 * linear 2^EV exposure, soft highlight roll-off, tonal regions, contrast, tone curve, saturation,
 * and vignette) without consuming WebGL contexts across 50+ grid/filmstrip cards.
 */
export const GradedThumbnailImg = memo(function GradedThumbnailImg({
  src,
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
  const tables = useMemo(() => buildDevelopSvgFilterTables(recipe, 33), [recipe]);

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
        src={src}
        alt={alt}
        loading={loading}
        decoding={decoding}
        draggable={draggable}
        onLoad={onLoad}
        style={mergedStyle}
        className={className}
      />
      {showVignette && tables?.vignetteOverlay && (
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ background: tables.vignetteOverlay }}
        />
      )}
    </>
  );
});
