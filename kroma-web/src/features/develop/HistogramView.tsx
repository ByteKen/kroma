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
    <div className="p-2.5 rounded-lg bg-surface-elevated border border-hairline flex flex-col gap-2">
      <div className="flex items-center justify-between tabular-nums-mono text-[10px]">
        <span className="uppercase tracking-wider text-ink-telemetry font-semibold">
          {liveHistogram ? 'RGB Histogram (Live)' : 'RGB Histogram'}
        </span>
        <span className="text-ink-secondary">
          {recipe.whiteBalance.temperatureKelvin}K •{' '}
          {recipe.basicTone.exposureEv >= 0 ? '+' : ''}
          {recipe.basicTone.exposureEv.toFixed(2)} EV
        </span>
      </div>

      {/* Crisp Darkroom RGB Waveform Canvas + Lightroom Clipping Triangles (J) */}
      <div className="h-[78px] w-full rounded bg-canvas border border-hairline relative overflow-hidden">
        <div className="absolute inset-0 grid grid-cols-5 pointer-events-none">
          {[0, 1, 2, 3, 4].map((idx) => (
            <div
              key={idx}
              className={idx < 4 ? 'border-r border-hairline/40 h-full' : 'h-full'}
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
            className={`absolute top-1 left-1 z-10 w-4 h-4 rounded flex items-center justify-center transition-colors cursor-pointer ${
              showShadowClip
                ? 'bg-blue-500/30 border border-blue-400 text-blue-300'
                : shadowsClipped
                  ? 'bg-surface-lowest/75 border border-blue-500/50 text-blue-400 hover:bg-blue-500/20'
                  : 'bg-surface-lowest/60 border border-hairline text-ink-telemetry/60 hover:text-ink-primary'
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
            className={`absolute top-1 right-1 z-10 w-4 h-4 rounded flex items-center justify-center transition-colors cursor-pointer ${
              showHighlightClip
                ? 'bg-red-500/30 border border-red-400 text-red-300'
                : highlightsClipped
                  ? 'bg-surface-lowest/75 border border-red-500/50 text-red-400 hover:bg-red-500/20'
                  : 'bg-surface-lowest/60 border border-hairline text-ink-telemetry/60 hover:text-ink-primary'
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
            fill="rgba(239, 68, 68, 0.34)"
            stroke="rgba(239, 68, 68, 0.85)"
            strokeWidth="1"
          />
          <path
            d={greenPath}
            fill="rgba(16, 185, 129, 0.30)"
            stroke="rgba(78, 222, 163, 0.85)"
            strokeWidth="1"
          />
          <path
            d={bluePath}
            fill="rgba(59, 130, 246, 0.32)"
            stroke="rgba(96, 165, 250, 0.85)"
            strokeWidth="1"
          />
        </svg>
      </div>

      {/* Compact EXIF Strip */}
      <div className="flex items-center justify-between tabular-nums-mono text-[10px] text-ink-telemetry">
        <span>ISO {exif.iso}</span>
        <span>{exif.focalLengthMm}</span>
        <span>{exif.aperture}</span>
        <span>{exif.shutterSpeed}</span>
      </div>
    </div>
  );
}
