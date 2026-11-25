import { useMemo } from 'react';
import type { LiveRgbHistogram } from '../../application/ports/studioGateway';
import { estimatePreviewHistogramBins, type DevelopRecipe } from '../../domain/develop';
import type { ExifTelemetry } from '../../domain/photo';

interface HistogramViewProps {
  readonly recipe: DevelopRecipe;
  readonly exif: ExifTelemetry;
  readonly liveHistogram?: LiveRgbHistogram | null;
  readonly showShadowClip?: boolean;
  readonly showHighlightClip?: boolean;
  readonly onToggleShadowClip?: () => void;
  readonly onToggleHighlightClip?: () => void;
}

function buildAreaPath(bins: readonly number[], width: number, height: number): string {
  const stepX = width / Math.max(1, bins.length - 1);
  const points = bins.map(
    (val, i) => `${(i * stepX).toFixed(1)},${(height - val * height).toFixed(1)}`,
  );
  return `M 0,${height} L ${points.join(' L ')} L ${width},${height} Z`;
}

export function HistogramView({
  recipe,
  exif,
  liveHistogram,
  showShadowClip = false,
  showHighlightClip = false,
  onToggleShadowClip,
  onToggleHighlightClip,
}: HistogramViewProps) {
  const { redPath, greenPath, bluePath, shadowsClipped, highlightsClipped } = useMemo(() => {
    const bins = liveHistogram ?? estimatePreviewHistogramBins(recipe);
    const n = bins.red.length;
    const shdClip =
      n > 0 && Math.max(bins.red[0] ?? 0, bins.green[0] ?? 0, bins.blue[0] ?? 0) > 0.32;
    const hiClip =
      n > 0 &&
      Math.max(bins.red[n - 1] ?? 0, bins.green[n - 1] ?? 0, bins.blue[n - 1] ?? 0) > 0.32;
    return {
      redPath: buildAreaPath(bins.red, 260, 76),
      greenPath: buildAreaPath(bins.green, 260, 76),
      bluePath: buildAreaPath(bins.blue, 260, 76),
      shadowsClipped: shdClip,
      highlightsClipped: hiClip,
    };
  }, [recipe, liveHistogram]);

  return (
    <div className="flex flex-col gap-2">
      {/* Frameless Darkroom RGB Waveform Canvas + Quiet Clipping Triangles (J) */}
      <div className="h-[74px] w-full rounded-xl bg-surface-lowest/95 border border-hairline/60 relative overflow-hidden">
        <div className="absolute inset-0 grid grid-cols-4 pointer-events-none">
          {[0, 1, 2, 3].map((idx) => (
            <div
              key={idx}
              className={idx < 3 ? 'border-r border-white/[0.04] h-full' : 'h-full'}
            />
          ))}
        </div>

        {/* Left Shadow Clipping Warning Toggle (Blue) */}
        {onToggleShadowClip && (
          <button
            type="button"
            onClick={onToggleShadowClip}
            title="Show Shadow Clipping (J) — blue overlay on crushed blacks"
            aria-label="Toggle shadow clipping warning"
            aria-pressed={showShadowClip}
            className={`absolute top-1.5 left-1.5 z-10 w-4 h-4 rounded flex items-center justify-center transition-all duration-150 cursor-pointer ${
              showShadowClip
                ? 'bg-blue-500/30 border border-blue-400/80 text-blue-300 opacity-100'
                : shadowsClipped
                  ? 'bg-black/50 text-blue-400 opacity-85 hover:opacity-100'
                  : 'text-ink-telemetry/50 opacity-45 hover:opacity-100 hover:text-ink-primary'
            }`}
          >
            <span className="text-[8px] leading-none">▲</span>
          </button>
        )}

        {/* Right Highlight Clipping Warning Toggle (Red) */}
        {onToggleHighlightClip && (
          <button
            type="button"
            onClick={onToggleHighlightClip}
            title="Show Highlight Clipping (J) — red overlay on blown whites"
            aria-label="Toggle highlight clipping warning"
            aria-pressed={showHighlightClip}
            className={`absolute top-1.5 right-1.5 z-10 w-4 h-4 rounded flex items-center justify-center transition-all duration-150 cursor-pointer ${
              showHighlightClip
                ? 'bg-red-500/30 border border-red-400/80 text-red-300 opacity-100'
                : highlightsClipped
                  ? 'bg-black/50 text-red-400 opacity-85 hover:opacity-100'
                  : 'text-ink-telemetry/50 opacity-45 hover:opacity-100 hover:text-ink-primary'
            }`}
          >
            <span className="text-[8px] leading-none">▲</span>
          </button>
        )}

        <svg
          viewBox="0 0 260 76"
          preserveAspectRatio="none"
          className="w-full h-full block"
          aria-label="Live RGB exposure histogram"
        >
          <path
            d={redPath}
            fill="rgba(239, 68, 68, 0.28)"
            stroke="rgba(239, 68, 68, 0.78)"
            strokeWidth="1"
          />
          <path
            d={greenPath}
            fill="rgba(16, 185, 129, 0.25)"
            stroke="rgba(78, 222, 163, 0.78)"
            strokeWidth="1"
          />
          <path
            d={bluePath}
            fill="rgba(59, 130, 246, 0.28)"
            stroke="rgba(96, 165, 250, 0.78)"
            strokeWidth="1"
          />
        </svg>
      </div>

      {/* Quiet Telemetry & EXIF Readout */}
      <div className="flex items-center justify-between gap-2 px-0.5 tabular-nums-mono text-[10px] text-ink-telemetry">
        <div className="flex items-center gap-1.5 truncate">
          <span>ISO {exif.iso}</span>
          <span className="opacity-40">·</span>
          <span>{exif.focalLengthMm}</span>
          <span className="opacity-40">·</span>
          <span>{exif.aperture}</span>
        </div>
        <span className="text-ink-secondary shrink-0">
          {recipe.whiteBalance.temperatureKelvin}K ·{' '}
          {recipe.basicTone.exposureEv >= 0 ? '+' : ''}
          {recipe.basicTone.exposureEv.toFixed(2)} EV
        </span>
      </div>
    </div>
  );
}
