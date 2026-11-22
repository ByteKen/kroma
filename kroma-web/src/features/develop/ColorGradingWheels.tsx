import { memo, useCallback, useRef, useState } from 'react';
import {
  createDefaultColorGrading,
  isColorGradingModified,
  type ColorGradingSettings,
  type ColorGradingWheel,
} from '../../domain/develop';
import { DarkroomScrubber } from '../darkroom-kit';

type WheelKey = 'shadows' | 'midtones' | 'highlights';
type ViewMode = 'all' | WheelKey;

interface ColorGradingWheelsProps {
  readonly value: ColorGradingSettings;
  readonly onChange: (next: ColorGradingSettings) => void;
}

const WHEELS: readonly { readonly key: WheelKey; readonly label: string }[] = [
  { key: 'shadows', label: 'Shadows' },
  { key: 'midtones', label: 'Midtones' },
  { key: 'highlights', label: 'Highlights' },
];

function formatSigned(v: number): string {
  const r = Math.round(v);
  return r > 0 ? `+${r}` : String(r);
}

interface SingleWheelDialProps {
  readonly label: string;
  readonly wheel: ColorGradingWheel;
  readonly sizePx: number;
  readonly onChange: (next: ColorGradingWheel) => void;
  readonly onSelectFocus?: () => void;
}

function SingleWheelDial({
  label,
  wheel,
  sizePx,
  onChange,
  onSelectFocus,
}: SingleWheelDialProps) {
  const dialRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);

  const updateFromPointer = useCallback(
    (clientX: number, clientY: number) => {
      const rect = dialRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0) return;
      const radius = rect.width * 0.5;
      const dx = (clientX - (rect.left + radius)) / Math.max(radius - 5, 1);
      const dy = (clientY - (rect.top + radius)) / Math.max(radius - 5, 1);
      const dist = Math.min(1, Math.hypot(dx, dy));
      // 0° = Red at top-right/right; align with CSS conic-gradient (0° at top = -90° math)
      const angleDeg = ((Math.atan2(dy, dx) * 180) / Math.PI + 90 + 360) % 360;
      const saturation = Math.round(dist * 100);
      onChange({
        ...wheel,
        hueDeg: saturation === 0 ? 0 : Math.round(angleDeg),
        saturation,
      });
    },
    [wheel, onChange],
  );

  const angleRad = ((wheel.hueDeg - 90) * Math.PI) / 180;
  const rPct = (wheel.saturation / 100) * 44;
  const puckX = 50 + Math.cos(angleRad) * rPct;
  const puckY = 50 + Math.sin(angleRad) * rPct;
  const active = wheel.saturation > 0 || wheel.luminance !== 0;

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="flex items-center justify-between w-full px-0.5">
        <button
          type="button"
          onClick={onSelectFocus}
          onDoubleClick={() => onChange({ hueDeg: 0, saturation: 0, luminance: 0 })}
          title="Click to focus • Double-click to reset wheel"
          className={`tabular-nums-mono text-[9.5px] uppercase tracking-wider transition-colors cursor-pointer ${
            active ? 'text-amber-soft font-semibold' : 'text-ink-telemetry hover:text-ink-primary'
          }`}
        >
          {label}
        </button>
        {active && (
          <button
            type="button"
            onClick={() => onChange({ hueDeg: 0, saturation: 0, luminance: 0 })}
            aria-label={`Reset ${label} wheel`}
            className="text-ink-telemetry hover:text-amber-soft cursor-pointer leading-none"
          >
            <span className="material-symbols-outlined text-[12px]">restart_alt</span>
          </button>
        )}
      </div>

      {/* Polar Conic Color Wheel */}
      <div
        ref={dialRef}
        role="slider"
        aria-label={`${label} color wheel`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={wheel.saturation}
        tabIndex={0}
        onDoubleClick={() => onChange({ hueDeg: 0, saturation: 0, luminance: 0 })}
        onPointerDown={(e) => {
          e.preventDefault();
          e.currentTarget.setPointerCapture(e.pointerId);
          setDragging(true);
          updateFromPointer(e.clientX, e.clientY);
        }}
        onPointerMove={(e) => {
          if (dragging) updateFromPointer(e.clientX, e.clientY);
        }}
        onPointerUp={(e) => {
          setDragging(false);
          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId);
          }
        }}
        onPointerCancel={() => setDragging(false)}
        style={{
          width: `${sizePx}px`,
          height: `${sizePx}px`,
          background:
            'radial-gradient(circle, rgba(24,26,31,0.92) 0%, rgba(24,26,31,0.28) 62%, transparent 100%), conic-gradient(#EF4444, #F59E0B, #EAB308, #22C55E, #06B6D4, #3B82F6, #A855F7, #EC4899, #EF4444)',
        }}
        className={`relative rounded-full border shadow-inner select-none touch-none ${
          dragging
            ? 'border-amber-core cursor-grabbing'
            : active
              ? 'border-amber-core/55 cursor-crosshair'
              : 'border-hairline-strong cursor-crosshair'
        }`}
      >
        {/* Center Crosshair */}
        <span className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full border border-white/40" />

        {/* Vector Tether Line */}
        {wheel.saturation > 0 && (
          <svg
            viewBox="0 0 100 100"
            className="pointer-events-none absolute inset-0 w-full h-full"
          >
            <line
              x1="50"
              y1="50"
              x2={puckX}
              y2={puckY}
              stroke="rgba(255,255,255,0.65)"
              strokeWidth="1.5"
            />
          </svg>
        )}

        {/* Draggable Color Puck */}
        <span
          style={{
            left: `${puckX}%`,
            top: `${puckY}%`,
            backgroundColor:
              wheel.saturation > 0
                ? `hsl(${wheel.hueDeg}, ${Math.max(45, wheel.saturation)}%, 55%)`
                : '#E5E7EB',
          }}
          className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 border-white shadow-[0_1px_4px_rgba(0,0,0,0.8)]"
        />
      </div>

      {/* Compact Luminance Slider Under Mini Wheel */}
      {sizePx < 100 && (
        <div className="w-full flex items-center gap-1">
          <input
            type="range"
            aria-label={`${label} luminance`}
            min={-100}
            max={100}
            step={1}
            value={wheel.luminance}
            onChange={(e) => onChange({ ...wheel, luminance: Number(e.target.value) })}
            onDoubleClick={() => onChange({ ...wheel, luminance: 0 })}
            title={`${label} Luminance (${formatSigned(wheel.luminance)}) • Double-click resets`}
            className="kroma-slider w-full"
          />
        </div>
      )}
    </div>
  );
}

/**
 * Lightroom-style 3-Way Color Grading panel (`Shadows`, `Midtones`, `Highlights` wheels +
 * per-wheel Hue/Saturation/Luminance + global `Blending` & `Balance`).
 */
export const ColorGradingWheels = memo(function ColorGradingWheels({
  value,
  onChange,
}: ColorGradingWheelsProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('all');
  const modified = isColorGradingModified(value);

  const patchWheel = (key: WheelKey, nextWheel: ColorGradingWheel) => {
    onChange({ ...value, [key]: nextWheel });
  };

  return (
    <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
          Color Grading (3-Way)
        </span>
        {modified && (
          <button
            type="button"
            onClick={() => onChange(createDefaultColorGrading())}
            className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
          >
            Reset
          </button>
        )}
      </div>

      {/* View Focus Selector (3-Way Grid vs Single Precision Wheel) */}
      <div className="grid grid-cols-4 gap-1 p-0.5 rounded-md bg-canvas border border-hairline">
        {(
          [
            { id: 'all', label: '3-Way' },
            { id: 'shadows', label: 'Shd' },
            { id: 'midtones', label: 'Mid' },
            { id: 'highlights', label: 'Hi' },
          ] as const
        ).map((tab) => {
          const isActive = viewMode === tab.id;
          const hasEdit =
            tab.id !== 'all' &&
            (value[tab.id].saturation > 0 || value[tab.id].luminance !== 0);
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setViewMode(tab.id)}
              className={`relative py-1 rounded tabular-nums-mono text-[10px] transition-colors cursor-pointer ${
                isActive
                  ? 'bg-surface-active text-amber-soft font-semibold border border-amber-core/45'
                  : 'text-ink-secondary hover:text-ink-primary'
              }`}
            >
              {tab.label}
              {hasEdit && (
                <span className="absolute top-0.5 right-1 w-1 h-1 rounded-full bg-amber-core" />
              )}
            </button>
          );
        })}
      </div>

      {viewMode === 'all' ? (
        <div className="grid grid-cols-3 gap-2 pt-0.5">
          {WHEELS.map((meta) => (
            <SingleWheelDial
              key={meta.key}
              label={meta.label}
              wheel={value[meta.key]}
              sizePx={74}
              onChange={(w) => patchWheel(meta.key, w)}
              onSelectFocus={() => setViewMode(meta.key)}
            />
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-2.5 items-center pt-0.5">
          <SingleWheelDial
            label={WHEELS.find((w) => w.key === viewMode)!.label}
            wheel={value[viewMode]}
            sizePx={136}
            onChange={(w) => patchWheel(viewMode, w)}
          />
          <div className="w-full flex flex-col gap-2">
            <DarkroomScrubber
              label="Hue"
              value={Math.round(value[viewMode].hueDeg)}
              min={0}
              max={360}
              step={1}
              defaultValue={0}
              formatValue={(v) => `${v}°`}
              trackGradient="linear-gradient(90deg, #EF4444, #F59E0B, #22C55E, #06B6D4, #3B82F6, #A855F7, #EF4444)"
              onChange={(hueDeg) => patchWheel(viewMode, { ...value[viewMode], hueDeg })}
            />
            <DarkroomScrubber
              label="Saturation"
              value={value[viewMode].saturation}
              min={0}
              max={100}
              step={1}
              defaultValue={0}
              formatValue={(v) => String(v)}
              onChange={(saturation) => patchWheel(viewMode, { ...value[viewMode], saturation })}
            />
            <DarkroomScrubber
              label="Luminance"
              value={value[viewMode].luminance}
              min={-100}
              max={100}
              step={1}
              defaultValue={0}
              formatValue={(v) => formatSigned(v)}
              onChange={(luminance) => patchWheel(viewMode, { ...value[viewMode], luminance })}
            />
          </div>
        </div>
      )}

      {/* Global Blending & Balance Controls */}
      <div className="pt-2 border-t border-hairline/70 flex flex-col gap-2">
        <DarkroomScrubber
          label="Blending"
          value={value.blending}
          min={0}
          max={100}
          step={1}
          defaultValue={50}
          formatValue={(v) => String(v)}
          onChange={(blending) => onChange({ ...value, blending })}
        />
        <DarkroomScrubber
          label="Balance"
          value={value.balance}
          min={-100}
          max={100}
          step={1}
          defaultValue={0}
          formatValue={(v) => formatSigned(v)}
          onChange={(balance) => onChange({ ...value, balance })}
        />
      </div>
    </section>
  );
});
