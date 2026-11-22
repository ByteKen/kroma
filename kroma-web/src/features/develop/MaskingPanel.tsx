import { memo, useState } from 'react';
import {
  STUDIO_BACKDROP_SWATCHES,
  STUDIO_GRADIENT_PRESETS,
  aiMaskKindLabel,
  backgroundRemovalPreviewCss,
  createBrushComponent,
  createColorRangeComponent,
  createDefaultLocalAdjustments,
  createLinearGradientComponent,
  createLuminanceRangeComponent,
  createMaskLayer,
  createRadialGradientComponent,
  hexToRgbTuple,
  isBackgroundRemovalActive,
  isLocalAdjustmentsModified,
  maskComponentLabel,
  rgbTupleToHex,
  type AiMaskKind,
  type BackgroundRemovalSettings,
  type LocalAdjustments,
  type MaskCombineOp,
  type MaskComponent,
  type MaskLayer,
} from '../../domain/masking';
import { DarkroomButton, DarkroomScrubber } from '../darkroom-kit';
import { AccordionSection } from './AccordionSection';
import type { BrushToolOptions } from './MaskCanvasOverlay';

interface MaskingPanelProps {
  readonly masks: readonly MaskLayer[];
  readonly backgroundRemoval: BackgroundRemovalSettings;
  readonly activeMaskId: string | null;
  readonly activeComponentId: string | null;
  readonly creatingNewMask: boolean;
  readonly brushOptions: BrushToolOptions;
  readonly colorPickActive: boolean;
  readonly busyAiKind: AiMaskKind | null;
  readonly busyBgRemoval: boolean;
  readonly onSelectMask: (layerId: string | null, componentId?: string | null) => void;
  readonly onRequestNewMask: () => void;
  readonly onCancelNewMask: () => void;
  readonly onChangeBrushOptions: (next: BrushToolOptions) => void;
  readonly onToggleColorPick: () => void;
  readonly onCreateAiMask: (
    kind: AiMaskKind,
    attachToLayerId?: string,
    op?: MaskCombineOp,
  ) => void;
  readonly onRemoveBackground: () => void;
  readonly onChangeBackgroundRemoval: (
    next: BackgroundRemovalSettings,
    label: string,
    coalesce?: boolean,
  ) => void;
  readonly onChangeMasks: (
    nextMasks: readonly MaskLayer[],
    label: string,
    coalesce?: boolean,
  ) => void;
}

const AI_MASK_TILES: readonly {
  readonly kind: AiMaskKind;
  readonly label: string;
  readonly icon: string;
}[] = [
    { kind: 'subject', label: 'Subject', icon: 'person' },
    { kind: 'background', label: 'Background', icon: 'wallpaper' },
    { kind: 'face_skin', label: 'Face & Skin', icon: 'face' },
    { kind: 'sky', label: 'Sky', icon: 'wb_sunny' },
  ];

const MANUAL_MASK_TILES: readonly {
  readonly type:
  | 'brush'
  | 'linear_gradient'
  | 'radial_gradient'
  | 'luminance_range'
  | 'color_range';
  readonly label: string;
  readonly icon: string;
}[] = [
    { type: 'brush', label: 'Brush', icon: 'brush' },
    { type: 'linear_gradient', label: 'Linear Gradient', icon: 'gradient' },
    { type: 'radial_gradient', label: 'Radial Gradient', icon: 'radio_button_checked' },
    { type: 'luminance_range', label: 'Luminance Range', icon: 'contrast' },
    { type: 'color_range', label: 'Color Range', icon: 'palette' },
  ];

function formatSigned(v: number): string {
  return `${v > 0 ? '+' : ''}${v}`;
}

export function createManualMaskComponent(
  type: 'brush' | 'linear_gradient' | 'radial_gradient' | 'luminance_range' | 'color_range',
  op: MaskCombineOp,
): MaskComponent {
  switch (type) {
    case 'brush':
      return createBrushComponent(op);
    case 'linear_gradient':
      return createLinearGradientComponent(op);
    case 'radial_gradient':
      return createRadialGradientComponent(op);
    case 'luminance_range':
      return createLuminanceRangeComponent(op);
    case 'color_range':
      return createColorRangeComponent(op);
  }
}

/**
 * Calm Lightroom-style Masking & 1-Click Background Removal inspector.
 * Layers live in the floating `Masks` panel on the photo viewport; this inspector stays focused on:
 * 1) Background Cutout & Studio Backdrop (1-click RMBG-1.4 transparent PNG or solid backdrop),
 * 2) Either the clean "Create New Mask" picker OR the selected mask's Tool Settings & collapsible Local Adjustments.
 */
export const MaskingPanel = memo(function MaskingPanel({
  masks,
  backgroundRemoval,
  activeMaskId,
  activeComponentId,
  creatingNewMask,
  brushOptions,
  colorPickActive,
  busyAiKind,
  busyBgRemoval,
  onSelectMask,
  onRequestNewMask,
  onCancelNewMask,
  onChangeBrushOptions,
  onToggleColorPick,
  onCreateAiMask,
  onRemoveBackground,
  onChangeBackgroundRemoval,
  onChangeMasks,
}: MaskingPanelProps) {
  const [openAdj, setOpenAdj] = useState({
    tone: true,
    color: false,
    presence: false,
  });

  const activeLayer = masks.find((m) => m.id === activeMaskId) ?? masks[0] ?? null;
  const activeComponent =
    activeLayer?.components.find((c) => c.id === activeComponentId) ??
    activeLayer?.components[activeLayer.components.length - 1] ??
    null;

  const bgActive = isBackgroundRemovalActive(backgroundRemoval);
  const showNewPicker = creatingNewMask || !activeLayer;

  const updateLayer = (
    layerId: string,
    updater: (layer: MaskLayer) => MaskLayer,
    label: string,
    coalesce = true,
  ) => {
    onChangeMasks(
      masks.map((m) => (m.id === layerId ? updater(m) : m)),
      label,
      coalesce,
    );
  };

  const updateActiveAdj = (patch: Partial<LocalAdjustments>, label: string, coalesce = true) => {
    if (!activeLayer) return;
    updateLayer(
      activeLayer.id,
      (layer) => ({
        ...layer,
        adjustments: { ...layer.adjustments, ...patch },
      }),
      `Mask ${label}`,
      coalesce,
    );
  };

  const updateActiveComponent = (nextComp: MaskComponent, label: string, coalesce = true) => {
    if (!activeLayer) return;
    updateLayer(
      activeLayer.id,
      (layer) => ({
        ...layer,
        components: layer.components.map((c) => (c.id === nextComp.id ? nextComp : c)),
      }),
      label,
      coalesce,
    );
  };

  const handleAddNewManualMask = (
    type: 'brush' | 'linear_gradient' | 'radial_gradient' | 'luminance_range' | 'color_range',
    label: string,
  ) => {
    const comp = createManualMaskComponent(type, 'add');
    const nextLayer = createMaskLayer(`${label} ${masks.length + 1}`, comp);
    onChangeMasks([...masks, nextLayer], `New ${label} Mask`, false);
    onSelectMask(nextLayer.id, comp.id);
    onCancelNewMask();
    if (type === 'brush') {
      onChangeBrushOptions({ ...brushOptions, erase: false });
    }
  };

  const adj = activeLayer?.adjustments;
  const toneModified = Boolean(
    adj &&
    (Math.abs(adj.exposureEv) > 1e-4 ||
      adj.contrast !== 0 ||
      adj.highlights !== 0 ||
      adj.shadows !== 0 ||
      adj.whites !== 0 ||
      adj.blacks !== 0),
  );
  const colorModified = Boolean(
    adj && (adj.temperatureShift !== 0 || adj.tintShift !== 0 || adj.saturation !== 0),
  );
  const presenceModified = Boolean(
    adj && (adj.texture !== 0 || adj.clarity !== 0 || adj.dehaze !== 0 || adj.sharpness !== 0),
  );

  const bgColorHex = rgbTupleToHex(backgroundRemoval.colorRgb);
  const bgColor2Hex = rgbTupleToHex(backgroundRemoval.color2Rgb);

  return (
    <div className="flex flex-col gap-3">
      {/* 1. 1-Click Background Removal & Studio Backdrop */}
      <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-2.5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 min-w-0">
            <span
              className={`w-1.5 h-1.5 rounded-full shrink-0 ${bgActive ? 'bg-emerald-bright' : 'bg-ink-telemetry/40'
                }`}
            />
            <span className="font-display text-[11.5px] font-semibold text-ink-primary truncate">
              Background Removal
            </span>
          </div>
          {bgActive && (
            <button
              type="button"
              onClick={() =>
                onChangeBackgroundRemoval(
                  { ...backgroundRemoval, enabled: false, mode: 'original' },
                  'Restore Original Background',
                  false,
                )
              }
              className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-amber-soft cursor-pointer shrink-0"
            >
              Restore
            </button>
          )}
        </div>

        {!bgActive ? (
          <>
            <DarkroomButton
              variant="secondary"
              size="sm"
              icon={busyBgRemoval ? 'progress_activity' : 'content_cut'}
              disabled={busyBgRemoval || busyAiKind !== null}
              onClick={onRemoveBackground}
              className="w-full justify-center"
            >
              {busyBgRemoval ? 'Removing Background…' : 'Remove Background'}
            </DarkroomButton>
            <p className="text-[10px] text-ink-telemetry">
              Transparent, solid, or gradient backdrop
            </p>
          </>
        ) : (
          <>
            {/* Transparent PNG | Solid Color | Studio Gradient */}
            <div className="grid grid-cols-3 gap-1 p-0.5 rounded-md bg-canvas border border-hairline">
              <button
                type="button"
                onClick={() =>
                  onChangeBackgroundRemoval(
                    { ...backgroundRemoval, enabled: true, mode: 'transparent' },
                    'Transparent PNG Cutout',
                    false,
                  )
                }
                className={`py-1 rounded text-[10.5px] font-display transition-colors cursor-pointer ${backgroundRemoval.mode === 'transparent'
                    ? 'bg-surface-active text-ink-primary font-semibold border border-amber-core/45'
                    : 'text-ink-secondary hover:text-ink-primary'
                  }`}
              >
                Transparent
              </button>
              <button
                type="button"
                onClick={() =>
                  onChangeBackgroundRemoval(
                    { ...backgroundRemoval, enabled: true, mode: 'solid_color' },
                    'Solid Studio Backdrop',
                    false,
                  )
                }
                className={`py-1 rounded text-[10.5px] font-display transition-colors cursor-pointer ${backgroundRemoval.mode === 'solid_color'
                    ? 'bg-surface-active text-ink-primary font-semibold border border-amber-core/45'
                    : 'text-ink-secondary hover:text-ink-primary'
                  }`}
              >
                Solid
              </button>
              <button
                type="button"
                onClick={() =>
                  onChangeBackgroundRemoval(
                    { ...backgroundRemoval, enabled: true, mode: 'gradient' },
                    'Gradient Studio Backdrop',
                    false,
                  )
                }
                className={`py-1 rounded text-[10.5px] font-display transition-colors cursor-pointer ${backgroundRemoval.mode === 'gradient'
                    ? 'bg-surface-active text-ink-primary font-semibold border border-amber-core/45'
                    : 'text-ink-secondary hover:text-ink-primary'
                  }`}
              >
                Gradient
              </button>
            </div>

            {backgroundRemoval.mode === 'solid_color' && (
              <div className="flex items-center justify-between gap-1.5 pt-0.5">
                <div className="flex items-center gap-1.5">
                  {STUDIO_BACKDROP_SWATCHES.map((swatch) => {
                    const isPicked =
                      backgroundRemoval.colorRgb[0] === swatch.rgb[0] &&
                      backgroundRemoval.colorRgb[1] === swatch.rgb[1] &&
                      backgroundRemoval.colorRgb[2] === swatch.rgb[2];
                    return (
                      <button
                        key={swatch.id}
                        type="button"
                        title={swatch.label}
                        aria-label={`Studio backdrop ${swatch.label}`}
                        onClick={() =>
                          onChangeBackgroundRemoval(
                            {
                              ...backgroundRemoval,
                              enabled: true,
                              mode: 'solid_color',
                              colorRgb: swatch.rgb,
                            },
                            `Backdrop ${swatch.label}`,
                            false,
                          )
                        }
                        className={`w-6 h-6 rounded-full border transition-transform cursor-pointer ${isPicked
                            ? 'border-amber-core ring-2 ring-amber-core/50 scale-105'
                            : 'border-white/25 hover:scale-105'
                          }`}
                        style={{ backgroundColor: swatch.hex }}
                      />
                    );
                  })}
                </div>

                {/* Custom Color Picker */}
                <label
                  title={`Custom color (${bgColorHex})`}
                  className="h-6 px-1.5 rounded-md bg-canvas border border-hairline hover:border-hairline-strong flex items-center gap-1 cursor-pointer shrink-0"
                >
                  <input
                    type="color"
                    aria-label="Custom backdrop color"
                    value={bgColorHex}
                    onChange={(e) =>
                      onChangeBackgroundRemoval(
                        {
                          ...backgroundRemoval,
                          enabled: true,
                          mode: 'solid_color',
                          colorRgb: hexToRgbTuple(e.target.value),
                        },
                        'Custom Backdrop Color',
                        true,
                      )
                    }
                    className="w-3.5 h-3.5 rounded border-0 bg-transparent p-0 cursor-pointer"
                  />
                  <span className="tabular-nums-mono text-[9px] text-ink-secondary">
                    {bgColorHex}
                  </span>
                </label>
              </div>
            )}

            {backgroundRemoval.mode === 'gradient' && (
              <div className="flex flex-col gap-2 pt-0.5">
                {/* Gradient Presets (6 Studio Swatches evenly spaced) */}
                <div className="flex items-center justify-between gap-1.5">
                  {STUDIO_GRADIENT_PRESETS.map((preset) => {
                    const isPicked =
                      backgroundRemoval.gradientStyle === preset.style &&
                      backgroundRemoval.colorRgb[0] === preset.color1Rgb[0] &&
                      backgroundRemoval.colorRgb[1] === preset.color1Rgb[1] &&
                      backgroundRemoval.colorRgb[2] === preset.color1Rgb[2] &&
                      backgroundRemoval.color2Rgb[0] === preset.color2Rgb[0] &&
                      backgroundRemoval.color2Rgb[1] === preset.color2Rgb[1] &&
                      backgroundRemoval.color2Rgb[2] === preset.color2Rgb[2];
                    const previewBg = backgroundRemovalPreviewCss({
                      colorRgb: preset.color1Rgb,
                      color2Rgb: preset.color2Rgb,
                      gradientStyle: preset.style,
                      gradientAngleDeg: preset.angleDeg,
                    });
                    return (
                      <button
                        key={preset.id}
                        type="button"
                        title={preset.label}
                        aria-label={`Studio gradient ${preset.label}`}
                        onClick={() =>
                          onChangeBackgroundRemoval(
                            {
                              ...backgroundRemoval,
                              enabled: true,
                              mode: 'gradient',
                              gradientStyle: preset.style,
                              gradientAngleDeg: preset.angleDeg,
                              colorRgb: preset.color1Rgb,
                              color2Rgb: preset.color2Rgb,
                            },
                            `Backdrop ${preset.label}`,
                            false,
                          )
                        }
                        className={`w-7 h-7 rounded-full border transition-transform cursor-pointer shrink-0 ${isPicked
                            ? 'border-amber-core ring-2 ring-amber-core/50 scale-105'
                            : 'border-white/25 hover:scale-105'
                          }`}
                        style={{ background: previewBg }}
                      />
                    );
                  })}
                </div>

                {/* Style Toggle (Radial Spot / Linear) + Swap Colors */}
                <div className="flex items-center justify-between gap-1.5">
                  <div className="flex-1 grid grid-cols-2 gap-0.5 p-0.5 rounded-md bg-canvas border border-hairline">
                    <button
                      type="button"
                      title="Radial Studio Spotlight"
                      onClick={() =>
                        onChangeBackgroundRemoval(
                          { ...backgroundRemoval, gradientStyle: 'radial' },
                          'Radial Studio Gradient',
                          false,
                        )
                      }
                      className={`py-0.5 rounded text-[10px] font-display cursor-pointer transition-colors ${backgroundRemoval.gradientStyle === 'radial'
                          ? 'bg-surface-active text-amber-soft font-semibold'
                          : 'text-ink-telemetry hover:text-ink-primary'
                        }`}
                    >
                      Radial Spot
                    </button>
                    <button
                      type="button"
                      title="Linear Directional Gradient"
                      onClick={() =>
                        onChangeBackgroundRemoval(
                          { ...backgroundRemoval, gradientStyle: 'linear' },
                          'Linear Studio Gradient',
                          false,
                        )
                      }
                      className={`py-0.5 rounded text-[10px] font-display cursor-pointer transition-colors ${backgroundRemoval.gradientStyle === 'linear'
                          ? 'bg-surface-active text-amber-soft font-semibold'
                          : 'text-ink-telemetry hover:text-ink-primary'
                        }`}
                    >
                      Linear
                    </button>
                  </div>

                  <button
                    type="button"
                    title="Swap gradient colors"
                    aria-label="Swap gradient colors"
                    onClick={() =>
                      onChangeBackgroundRemoval(
                        {
                          ...backgroundRemoval,
                          colorRgb: backgroundRemoval.color2Rgb,
                          color2Rgb: backgroundRemoval.colorRgb,
                        },
                        'Swap Gradient Colors',
                        false,
                      )
                    }
                    className="h-6 px-2 rounded-md bg-canvas border border-hairline hover:border-hairline-strong text-ink-telemetry hover:text-amber-soft flex items-center gap-1 text-[10px] font-display shrink-0 cursor-pointer transition-colors"
                  >
                    <span className="material-symbols-outlined text-[13px]">swap_horiz</span>
                    <span>Swap</span>
                  </button>
                </div>

                {/* Custom Center/Start & Edge/End Color Pickers (clean, zero truncation) */}
                <div className="grid grid-cols-2 gap-1.5">
                  <div className="flex flex-col gap-1 min-w-0">
                    <span className="tabular-nums-mono text-[9px] uppercase tracking-wider text-ink-telemetry">
                      {backgroundRemoval.gradientStyle === 'radial' ? 'Center' : 'Start'}
                    </span>
                    <label className="h-7 px-2 rounded-md bg-canvas border border-hairline hover:border-hairline-strong flex items-center justify-between gap-1.5 cursor-pointer">
                      <input
                        type="color"
                        aria-label={
                          backgroundRemoval.gradientStyle === 'radial'
                            ? 'Gradient center color'
                            : 'Gradient start color'
                        }
                        value={bgColorHex}
                        onChange={(e) =>
                          onChangeBackgroundRemoval(
                            {
                              ...backgroundRemoval,
                              enabled: true,
                              mode: 'gradient',
                              colorRgb: hexToRgbTuple(e.target.value),
                            },
                            'Gradient Color 1',
                            true,
                          )
                        }
                        className="w-4 h-4 rounded border-0 bg-transparent p-0 cursor-pointer shrink-0"
                      />
                      <span className="tabular-nums-mono text-[10px] text-ink-secondary">
                        {bgColorHex}
                      </span>
                    </label>
                  </div>

                  <div className="flex flex-col gap-1 min-w-0">
                    <span className="tabular-nums-mono text-[9px] uppercase tracking-wider text-ink-telemetry">
                      {backgroundRemoval.gradientStyle === 'radial' ? 'Edge' : 'End'}
                    </span>
                    <label className="h-7 px-2 rounded-md bg-canvas border border-hairline hover:border-hairline-strong flex items-center justify-between gap-1.5 cursor-pointer">
                      <input
                        type="color"
                        aria-label={
                          backgroundRemoval.gradientStyle === 'radial'
                            ? 'Gradient edge color'
                            : 'Gradient end color'
                        }
                        value={bgColor2Hex}
                        onChange={(e) =>
                          onChangeBackgroundRemoval(
                            {
                              ...backgroundRemoval,
                              enabled: true,
                              mode: 'gradient',
                              color2Rgb: hexToRgbTuple(e.target.value),
                            },
                            'Gradient Color 2',
                            true,
                          )
                        }
                        className="w-4 h-4 rounded border-0 bg-transparent p-0 cursor-pointer shrink-0"
                      />
                      <span className="tabular-nums-mono text-[10px] text-ink-secondary">
                        {bgColor2Hex}
                      </span>
                    </label>
                  </div>
                </div>

                {backgroundRemoval.gradientStyle === 'linear' && (
                  <DarkroomScrubber
                    label="Angle"
                    value={Math.round(backgroundRemoval.gradientAngleDeg)}
                    min={0}
                    max={360}
                    step={5}
                    defaultValue={180}
                    formatValue={(v) => `${v}°`}
                    onChange={(gradientAngleDeg) =>
                      onChangeBackgroundRemoval(
                        { ...backgroundRemoval, gradientAngleDeg },
                        'Gradient Angle',
                        true,
                      )
                    }
                  />
                )}
              </div>
            )}

            <DarkroomScrubber
              label="Edge Softness"
              value={Math.round(backgroundRemoval.feather)}
              min={0}
              max={100}
              defaultValue={25}
              formatValue={(v) => String(v)}
              onChange={(feather) =>
                onChangeBackgroundRemoval(
                  { ...backgroundRemoval, feather },
                  'Cutout Edge Softness',
                  true,
                )
              }
            />
          </>
        )}
      </section>

      {/* 2A. Create New Mask Picker (Lightroom Mask Picker) */}
      {showNewPicker ? (
        <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2">
            <span className="font-display text-[12px] font-semibold text-ink-primary">
              Create New Mask
            </span>
            {masks.length > 0 && (
              <button
                type="button"
                onClick={onCancelNewMask}
                className="text-[10.5px] text-ink-telemetry hover:text-ink-primary cursor-pointer"
              >
                Cancel
              </button>
            )}
          </div>

          {/* AI Selection */}
          <div className="flex flex-col gap-1.5">
            <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
              AI Selection
            </span>
            <div className="grid grid-cols-2 gap-1.5">
              {AI_MASK_TILES.map((tile) => {
                const busy = busyAiKind === tile.kind;
                return (
                  <button
                    key={tile.kind}
                    type="button"
                    disabled={busyAiKind !== null || busyBgRemoval}
                    onClick={() => onCreateAiMask(tile.kind)}
                    className="h-9 px-2.5 rounded-lg bg-canvas hover:bg-surface-active border border-hairline hover:border-amber-core/55 text-[11.5px] font-display text-ink-primary flex items-center gap-2 transition-colors cursor-pointer disabled:opacity-50"
                  >
                    <span
                      className={`material-symbols-outlined text-[16px] ${busy ? 'animate-spin text-amber-core' : 'text-amber-soft'
                        }`}
                    >
                      {busy ? 'progress_activity' : tile.icon}
                    </span>
                    <span className="truncate">{busy ? 'Detecting…' : tile.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Manual & Range Tools */}
          <div className="flex flex-col gap-1.5">
            <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
              Manual &amp; Range
            </span>
            <div className="flex flex-col gap-1">
              {MANUAL_MASK_TILES.map((tile) => (
                <button
                  key={tile.type}
                  type="button"
                  onClick={() => handleAddNewManualMask(tile.type, tile.label)}
                  className="h-8 px-2.5 rounded-lg bg-canvas hover:bg-surface-active border border-hairline hover:border-hairline-strong text-[11.5px] font-display text-ink-secondary hover:text-ink-primary flex items-center justify-between gap-2 transition-colors cursor-pointer"
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <span className="material-symbols-outlined text-[15px] text-ink-telemetry">
                      {tile.icon}
                    </span>
                    <span className="truncate">{tile.label}</span>
                  </span>
                  <span className="material-symbols-outlined text-[14px] text-ink-telemetry/70">
                    add
                  </span>
                </button>
              ))}
            </div>
          </div>
        </section>
      ) : (
        /* 2B. Selected Mask Editor (Tool Settings + Collapsible Local Adjustments) */
        <>
          {/* Active Layer Summary & Opacity */}
          <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-2.5">
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="font-display text-[12px] font-semibold text-ink-primary truncate">
                  {activeLayer.name}
                </div>
                {activeComponent && (
                  <div className="tabular-nums-mono text-[9.5px] text-ink-telemetry truncate">
                    Active tool: {maskComponentLabel(activeComponent)}
                  </div>
                )}
              </div>
              <DarkroomButton
                variant="subtle"
                size="xs"
                icon="add"
                onClick={onRequestNewMask}
                title="Create another mask layer"
              >
                New
              </DarkroomButton>
            </div>

            <DarkroomScrubber
              label="Mask Opacity"
              value={activeLayer.opacity}
              min={0}
              max={100}
              defaultValue={100}
              formatValue={(v) => `${v}%`}
              onChange={(opacity) =>
                updateLayer(activeLayer.id, (l) => ({ ...l, opacity }), 'Mask Opacity', true)
              }
            />

            {/* Contextual Tool Settings for Selected Sub-Mask Component */}
            {activeComponent && (
              <div className="pt-2 border-t border-hairline/70 flex flex-col gap-2">
                {activeComponent.type === 'brush' && (
                  <>
                    <div className="grid grid-cols-2 gap-1.5">
                      <DarkroomButton
                        variant={!brushOptions.erase ? 'primary' : 'subtle'}
                        size="xs"
                        icon="brush"
                        onClick={() => onChangeBrushOptions({ ...brushOptions, erase: false })}
                      >
                        Paint (+)
                      </DarkroomButton>
                      <DarkroomButton
                        variant={brushOptions.erase ? 'primary' : 'subtle'}
                        size="xs"
                        icon="ink_eraser"
                        onClick={() => onChangeBrushOptions({ ...brushOptions, erase: true })}
                      >
                        Erase (−)
                      </DarkroomButton>
                    </div>
                    <DarkroomScrubber
                      label="Size ([ / ])"
                      value={Math.round(brushOptions.size * 1000)}
                      min={8}
                      max={250}
                      defaultValue={65}
                      formatValue={(v) => `${(v / 10).toFixed(1)}%`}
                      onChange={(v) => onChangeBrushOptions({ ...brushOptions, size: v / 1000 })}
                    />
                    <DarkroomScrubber
                      label="Feather"
                      value={Math.round(brushOptions.feather * 100)}
                      min={0}
                      max={100}
                      defaultValue={55}
                      formatValue={(v) => `${v}%`}
                      onChange={(v) => onChangeBrushOptions({ ...brushOptions, feather: v / 100 })}
                    />
                    <DarkroomScrubber
                      label="Flow"
                      value={Math.round(brushOptions.flow * 100)}
                      min={5}
                      max={100}
                      defaultValue={100}
                      formatValue={(v) => `${v}%`}
                      onChange={(v) => onChangeBrushOptions({ ...brushOptions, flow: v / 100 })}
                    />
                    {activeComponent.strokes.length > 0 && (
                      <div className="flex justify-end">
                        <button
                          type="button"
                          onClick={() =>
                            updateActiveComponent(
                              { ...activeComponent, strokes: activeComponent.strokes.slice(0, -1) },
                              'Undo Brush Stroke',
                              false,
                            )
                          }
                          className="text-[10px] text-ink-secondary hover:text-amber-soft cursor-pointer"
                        >
                          Undo last stroke
                        </button>
                      </div>
                    )}
                  </>
                )}

                {activeComponent.type === 'linear_gradient' && (
                  <DarkroomScrubber
                    label="Gradient Feather"
                    value={Math.round(activeComponent.feather)}
                    min={0}
                    max={100}
                    defaultValue={55}
                    formatValue={(v) => `${v}%`}
                    onChange={(feather) =>
                      updateActiveComponent({ ...activeComponent, feather }, 'Linear Feather')
                    }
                  />
                )}

                {activeComponent.type === 'radial_gradient' && (
                  <>
                    <DarkroomScrubber
                      label="Radial Feather"
                      value={Math.round(activeComponent.feather)}
                      min={0}
                      max={100}
                      defaultValue={50}
                      formatValue={(v) => `${v}%`}
                      onChange={(feather) =>
                        updateActiveComponent({ ...activeComponent, feather }, 'Radial Feather')
                      }
                    />
                    <DarkroomScrubber
                      label="Rotation"
                      value={Math.round(activeComponent.rotationDeg)}
                      min={-180}
                      max={180}
                      defaultValue={0}
                      formatValue={(v) => `${v}°`}
                      onChange={(rotationDeg) =>
                        updateActiveComponent({ ...activeComponent, rotationDeg }, 'Radial Rotation')
                      }
                    />
                  </>
                )}

                {activeComponent.type === 'luminance_range' && (
                  <>
                    <div
                      className="h-3 rounded border border-hairline relative overflow-hidden"
                      style={{
                        background:
                          'linear-gradient(to right, #09090b 0%, #71717a 50%, #fafafa 100%)',
                      }}
                    >
                      <div
                        className="absolute top-0 bottom-0 bg-rose-500/45 border-x border-rose-300"
                        style={{
                          left: `${(activeComponent.minLuma * 100).toFixed(1)}%`,
                          width: `${Math.max(2, (activeComponent.maxLuma - activeComponent.minLuma) * 100).toFixed(1)}%`,
                        }}
                      />
                    </div>
                    <DarkroomScrubber
                      label="Min Luminance"
                      value={Math.round(activeComponent.minLuma * 100)}
                      min={0}
                      max={Math.round(activeComponent.maxLuma * 100)}
                      defaultValue={45}
                      formatValue={(v) => `${v}%`}
                      onChange={(v) =>
                        updateActiveComponent(
                          {
                            ...activeComponent,
                            minLuma: Math.min(activeComponent.maxLuma, v / 100),
                          },
                          'Luma Min',
                        )
                      }
                    />
                    <DarkroomScrubber
                      label="Max Luminance"
                      value={Math.round(activeComponent.maxLuma * 100)}
                      min={Math.round(activeComponent.minLuma * 100)}
                      max={100}
                      defaultValue={100}
                      formatValue={(v) => `${v}%`}
                      onChange={(v) =>
                        updateActiveComponent(
                          {
                            ...activeComponent,
                            maxLuma: Math.max(activeComponent.minLuma, v / 100),
                          },
                          'Luma Max',
                        )
                      }
                    />
                    <DarkroomScrubber
                      label="Smoothness"
                      value={Math.round(activeComponent.feather * 200)}
                      min={2}
                      max={100}
                      defaultValue={28}
                      formatValue={(v) => `${v}%`}
                      onChange={(v) =>
                        updateActiveComponent(
                          { ...activeComponent, feather: v / 200 },
                          'Luma Feather',
                        )
                      }
                    />
                  </>
                )}

                {activeComponent.type === 'color_range' && (
                  <>
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className="w-5 h-5 rounded border border-white/30 shadow-inner"
                          style={{
                            backgroundColor: `hsl(${Math.round(activeComponent.targetHueDeg)}, ${Math.round(activeComponent.targetSat * 100)}%, ${Math.round(activeComponent.targetLuma * 100)}%)`,
                          }}
                        />
                        <span className="tabular-nums-mono text-[10px] text-ink-secondary">
                          H {Math.round(activeComponent.targetHueDeg)}° · S{' '}
                          {Math.round(activeComponent.targetSat * 100)}%
                        </span>
                      </div>
                      <DarkroomButton
                        variant={colorPickActive ? 'primary' : 'secondary'}
                        size="xs"
                        icon="colorize"
                        onClick={onToggleColorPick}
                      >
                        {colorPickActive ? 'Click Photo…' : 'Pick Color'}
                      </DarkroomButton>
                    </div>
                    <DarkroomScrubber
                      label="Target Hue"
                      value={Math.round(activeComponent.targetHueDeg)}
                      min={0}
                      max={360}
                      defaultValue={28}
                      formatValue={(v) => `${v}°`}
                      onChange={(targetHueDeg) =>
                        updateActiveComponent({ ...activeComponent, targetHueDeg }, 'Color Hue')
                      }
                    />
                    <DarkroomScrubber
                      label="Refine"
                      value={Math.round(activeComponent.tolerance * 100)}
                      min={4}
                      max={100}
                      defaultValue={26}
                      formatValue={(v) => `${v}%`}
                      onChange={(v) =>
                        updateActiveComponent(
                          { ...activeComponent, tolerance: v / 100 },
                          'Color Range',
                        )
                      }
                    />
                    <DarkroomScrubber
                      label="Feather"
                      value={Math.round(activeComponent.feather * 100)}
                      min={0}
                      max={100}
                      defaultValue={35}
                      formatValue={(v) => `${v}%`}
                      onChange={(v) =>
                        updateActiveComponent(
                          { ...activeComponent, feather: v / 100 },
                          'Color Feather',
                        )
                      }
                    />
                  </>
                )}

                {activeComponent.type === 'ai_matte' && (
                  <>
                    <div className="flex items-center justify-between text-[10px] text-ink-secondary">
                      <span>AI {aiMaskKindLabel(activeComponent.kind)} Matte</span>
                      <span className="tabular-nums-mono text-ink-telemetry">
                        {activeComponent.matteWidth}×{activeComponent.matteHeight}
                      </span>
                    </div>
                    <DarkroomScrubber
                      label="Edge Softness"
                      value={Math.round(activeComponent.feather)}
                      min={0}
                      max={100}
                      defaultValue={25}
                      formatValue={(v) => String(v)}
                      onChange={(feather) =>
                        updateActiveComponent({ ...activeComponent, feather }, 'AI Matte Feather')
                      }
                    />
                  </>
                )}
              </div>
            )}
          </section>

          {/* Collapsible Lightroom-Style Local Adjustments */}
          <div className="rounded-lg bg-surface border border-hairline overflow-hidden">
            <div className="px-3 py-2 bg-surface-low/60 border-b border-hairline flex items-center justify-between">
              <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry font-semibold">
                Local Adjustments
              </span>
              {isLocalAdjustmentsModified(activeLayer.adjustments) && (
                <button
                  type="button"
                  onClick={() =>
                    updateLayer(
                      activeLayer.id,
                      (l) => ({ ...l, adjustments: createDefaultLocalAdjustments() }),
                      'Reset Local Adjustments',
                      false,
                    )
                  }
                  className="tabular-nums-mono text-[9.5px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                >
                  Reset All
                </button>
              )}
            </div>

            <AccordionSection
              title="Tone"
              open={openAdj.tone}
              onToggle={() => setOpenAdj((s) => ({ ...s, tone: !s.tone }))}
              modified={toneModified}
              onReset={() =>
                updateActiveAdj(
                  {
                    exposureEv: 0,
                    contrast: 0,
                    highlights: 0,
                    shadows: 0,
                    whites: 0,
                    blacks: 0,
                  },
                  'Reset Tone',
                  false,
                )
              }
            >
              <DarkroomScrubber
                label="Exposure"
                value={activeLayer.adjustments.exposureEv}
                min={-4}
                max={4}
                step={0.05}
                defaultValue={0}
                formatValue={(v) => `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`}
                onChange={(exposureEv) => updateActiveAdj({ exposureEv }, 'Exposure')}
              />
              <DarkroomScrubber
                label="Contrast"
                value={activeLayer.adjustments.contrast}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(contrast) => updateActiveAdj({ contrast }, 'Contrast')}
              />
              <DarkroomScrubber
                label="Highlights"
                value={activeLayer.adjustments.highlights}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(highlights) => updateActiveAdj({ highlights }, 'Highlights')}
              />
              <DarkroomScrubber
                label="Shadows"
                value={activeLayer.adjustments.shadows}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(shadows) => updateActiveAdj({ shadows }, 'Shadows')}
              />
              <DarkroomScrubber
                label="Whites"
                value={activeLayer.adjustments.whites}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(whites) => updateActiveAdj({ whites }, 'Whites')}
              />
              <DarkroomScrubber
                label="Blacks"
                value={activeLayer.adjustments.blacks}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(blacks) => updateActiveAdj({ blacks }, 'Blacks')}
              />
            </AccordionSection>

            <AccordionSection
              title="Color"
              open={openAdj.color}
              onToggle={() => setOpenAdj((s) => ({ ...s, color: !s.color }))}
              modified={colorModified}
              onReset={() =>
                updateActiveAdj(
                  { temperatureShift: 0, tintShift: 0, saturation: 0 },
                  'Reset Color',
                  false,
                )
              }
            >
              <DarkroomScrubber
                label="Temp"
                value={activeLayer.adjustments.temperatureShift}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                trackGradient="linear-gradient(to right, #3b82f6, #71717a 50%, #f59e0b)"
                onChange={(temperatureShift) => updateActiveAdj({ temperatureShift }, 'Temp')}
              />
              <DarkroomScrubber
                label="Tint"
                value={activeLayer.adjustments.tintShift}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                trackGradient="linear-gradient(to right, #10b981, #71717a 50%, #ec4899)"
                onChange={(tintShift) => updateActiveAdj({ tintShift }, 'Tint')}
              />
              <DarkroomScrubber
                label="Saturation"
                value={activeLayer.adjustments.saturation}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(saturation) => updateActiveAdj({ saturation }, 'Saturation')}
              />
            </AccordionSection>

            <AccordionSection
              title="Presence & Detail"
              open={openAdj.presence}
              onToggle={() => setOpenAdj((s) => ({ ...s, presence: !s.presence }))}
              modified={presenceModified}
              onReset={() =>
                updateActiveAdj(
                  { texture: 0, clarity: 0, dehaze: 0, sharpness: 0 },
                  'Reset Presence',
                  false,
                )
              }
            >
              <DarkroomScrubber
                label="Texture"
                value={activeLayer.adjustments.texture}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(texture) => updateActiveAdj({ texture }, 'Texture')}
              />
              <DarkroomScrubber
                label="Clarity"
                value={activeLayer.adjustments.clarity}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(clarity) => updateActiveAdj({ clarity }, 'Clarity')}
              />
              <DarkroomScrubber
                label="Dehaze"
                value={activeLayer.adjustments.dehaze}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(dehaze) => updateActiveAdj({ dehaze }, 'Dehaze')}
              />
              <DarkroomScrubber
                label="Sharpness / Soften"
                value={activeLayer.adjustments.sharpness}
                min={-100}
                max={100}
                defaultValue={0}
                formatValue={formatSigned}
                onChange={(sharpness) => updateActiveAdj({ sharpness }, 'Sharpness')}
              />
            </AccordionSection>
          </div>
        </>
      )}
    </div>
  );
});
