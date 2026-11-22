import { invoke } from '@tauri-apps/api/core';
import { open as openNativeDialog } from '@tauri-apps/plugin-dialog';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useMemo, useRef, useState } from 'react';
import { isTauriDesktopEnvironment } from '../../adapters/gateways/tauriStudioGateway';
import {
  loadCustomLutsFromStorage,
  saveCustomLutsToStorage,
  type CustomCubeLutEntry,
} from '../../adapters/storage/customLutsStorage';
import {
  loadCustomPresetsFromStorage,
  saveCustomPresetsToStorage,
} from '../../adapters/storage/customPresetsStorage';
import type { LiveRgbHistogram } from '../../application/ports/studioGateway';
import { cacheParsedCubeLut3D, parseCubeLut3D } from '../../domain/cubeLut3d';
import {
  computeAutoToneFromHistogram,
  createDefaultCameraCalibration,
  createDefaultColorGrading,
  createDefaultCropTransform,
  createDefaultDevelopRecipe,
  estimatePreviewHistogramBins,
  getModifiedDevelopModules,
  getStudioDevelopPresets,
  isRecipeModified,
  WHITE_BALANCE_PRESETS,
  type BasicToneSettings,
  type CameraCalibrationSettings,
  type CropAspectRatio,
  type DevelopPreset,
  type DevelopRecipe,
  type HslChannelKey,
  type Lut3DSettings,
  type ParametricToneCurve,
  type RelativeDevelopDelta,
} from '../../domain/develop';
import {
  createDefaultBackgroundRemoval,
  isBackgroundRemovalActive,
  type AiMaskKind,
  type BackgroundRemovalSettings,
  type MaskCombineOp,
  type MaskLayer,
} from '../../domain/masking';
import type { PhotoAsset } from '../../domain/photo';
import {
  DarkroomButton,
  DarkroomScrubber,
  DarkroomToggle,
  FAST_FADE,
  SegmentedControl,
  TACTILE_SPRING,
  TelemetryPill,
  type SegmentedOption,
} from '../darkroom-kit';
import { AccordionSection } from './AccordionSection';
import { ColorGradingWheels } from './ColorGradingWheels';
import { fitCenteredCropForAspect } from './CropOverlay';
import { HistogramView } from './HistogramView';
import { InteractiveToneCurveEditor } from './InteractiveToneCurveEditor';
import type { BrushToolOptions } from './MaskCanvasOverlay';
import { MaskingPanel } from './MaskingPanel';

export type DevelopToolMode = 'edit' | 'crop' | 'mask' | 'presets';

type EditAccordionKey =
  | 'basic'
  | 'toneCurve'
  | 'colorMixer'
  | 'colorGrading'
  | 'detail'
  | 'effects'
  | 'calibration';

const TOOL_STRIP_ITEMS: readonly {
  readonly id: DevelopToolMode;
  readonly icon: string;
  readonly label: string;
  readonly shortcut: string;
  readonly title: string;
}[] = [
  { id: 'edit', icon: 'tune', label: 'Edit', shortcut: 'E', title: 'Develop Panels (E)' },
  { id: 'crop', icon: 'crop', label: 'Crop', shortcut: 'R', title: 'Crop & Straighten (R)' },
  {
    id: 'mask',
    icon: 'layers',
    label: 'Mask',
    shortcut: 'M',
    title: 'Masks & Background Removal (M)',
  },
  {
    id: 'presets',
    icon: 'auto_fix_high',
    label: 'Presets',
    shortcut: 'P',
    title: 'Quick Develop & Presets (P)',
  },
];

const TREATMENT_OPTIONS: readonly SegmentedOption<'color' | 'bw'>[] = [
  { id: 'color', label: 'Color', title: 'Full color grading' },
  { id: 'bw', label: 'B&W', title: 'Monochrome B&W treatment with 8-channel B&W Mix' },
];

const QUICK_NUDGES: readonly {
  readonly id: string;
  readonly label: string;
  readonly shortLabel: string;
  readonly delta: RelativeDevelopDelta;
}[] = [
  { id: 'exp-down', label: 'Exposure -0.33 EV', shortLabel: '-1/3 EV', delta: { exposureDeltaEv: -0.33 } },
  { id: 'exp-up', label: 'Exposure +0.33 EV', shortLabel: '+1/3 EV', delta: { exposureDeltaEv: 0.33 } },
  { id: 'wb-cool', label: 'Cooler -250K', shortLabel: '-250K', delta: { temperatureDeltaKelvin: -250 } },
  { id: 'wb-warm', label: 'Warmer +250K', shortLabel: '+250K', delta: { temperatureDeltaKelvin: 250 } },
  { id: 'con-up', label: 'Contrast +10', shortLabel: '+10 Con', delta: { contrastDelta: 10 } },
  { id: 'shd-up', label: 'Lift Shadows +15', shortLabel: '+15 Shd', delta: { shadowsDelta: 15 } },
  { id: 'hi-down', label: 'Recover Highlights -15', shortLabel: '-15 Hi', delta: { highlightsDelta: -15 } },
  { id: 'vib-up', label: 'Vibrance +10', shortLabel: '+10 Vib', delta: { vibranceDelta: 10 } },
];

const HSL_CHANNELS: readonly { key: HslChannelKey; label: string; swatch: string }[] = [
  { key: 'red', label: 'Red', swatch: '#EF4444' },
  { key: 'orange', label: 'Orange', swatch: '#F97316' },
  { key: 'yellow', label: 'Yellow', swatch: '#EAB308' },
  { key: 'green', label: 'Green', swatch: '#22C55E' },
  { key: 'aqua', label: 'Aqua', swatch: '#06B6D4' },
  { key: 'blue', label: 'Blue', swatch: '#3B82F6' },
  { key: 'purple', label: 'Purple', swatch: '#A855F7' },
  { key: 'magenta', label: 'Magenta', swatch: '#EC4899' },
];

const STUDIO_LUTS: readonly { label: string; lut: Lut3DSettings | null }[] = [
  { label: 'None (Native Camera Profile)', lut: null },
  {
    label: 'Kodak_Portra_400.cube',
    lut: { name: 'Kodak_Portra_400.cube', cubePath: 'luts/Kodak_Portra_400.cube', intensity: 65 },
  },
  {
    label: 'Fuji_Pro_400H.cube',
    lut: { name: 'Fuji_Pro_400H.cube', cubePath: 'luts/Fuji_Pro_400H.cube', intensity: 60 },
  },
  {
    label: 'Kroma_Studio_Neutral.cube',
    lut: {
      name: 'Kroma_Studio_Neutral.cube',
      cubePath: 'luts/Kroma_Studio_Neutral.cube',
      intensity: 50,
    },
  },
  {
    label: 'Ilford_HP5_Mono.cube',
    lut: { name: 'Ilford_HP5_Mono.cube', cubePath: 'luts/Ilford_HP5_Mono.cube', intensity: 85 },
  },
];

const ASPECT_OPTIONS: readonly { readonly id: CropAspectRatio; readonly label: string }[] = [
  { id: 'free', label: 'Free (Custom)' },
  { id: 'original', label: 'Original' },
  { id: '1:1', label: '1:1 (Square)' },
  { id: '4:5', label: '4:5 (8×10)' },
  { id: '5:7', label: '5:7' },
  { id: '2:3', label: '2:3 (4×6)' },
  { id: '16:9', label: '16:9 (Wide)' },
];

interface DevelopInspectorPanelProps {
  readonly photo: PhotoAsset;
  readonly liveHistogram: LiveRgbHistogram | null;
  readonly hasSlateAnchor: boolean;
  readonly hasCopiedRecipe: boolean;
  readonly selectedCount: number;
  readonly widthPx?: number;
  readonly showShadowClip: boolean;
  readonly showHighlightClip: boolean;
  readonly eyedropperActive: boolean;
  readonly naturalAspect?: number;
  readonly toolMode: DevelopToolMode;
  readonly onChangeToolMode: (mode: DevelopToolMode) => void;
  readonly activeMaskId: string | null;
  readonly activeMaskComponentId: string | null;
  readonly creatingNewMask: boolean;
  readonly brushOptions: BrushToolOptions;
  readonly colorPickActive: boolean;
  readonly busyAiMaskKind: AiMaskKind | null;
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
  readonly onToggleShadowClip: () => void;
  readonly onToggleHighlightClip: () => void;
  readonly onToggleEyedropper: () => void;
  readonly onChangeRecipe: (next: DevelopRecipe, label?: string, coalesce?: boolean) => void;
  readonly onApplyPreset: (presetName: string, recipe: DevelopRecipe) => void;
  readonly onApplyRelativeNudge: (delta: RelativeDevelopDelta, label: string) => void;
  readonly onSyncSlateWb: () => void;
  readonly onOpenSyncModal: () => void;
  readonly onCopyRecipe: () => void;
  readonly onPasteRecipe: () => void;
}

function formatSigned(v: number, decimals = 0): string {
  const formatted = decimals > 0 ? v.toFixed(decimals) : String(Math.round(v));
  return v > 0 ? `+${formatted}` : formatted;
}

export const DevelopInspectorPanel = memo(function DevelopInspectorPanel({
  photo,
  liveHistogram,
  hasSlateAnchor,
  hasCopiedRecipe,
  selectedCount,
  widthPx = 336,
  showShadowClip,
  showHighlightClip,
  eyedropperActive,
  naturalAspect = 2 / 3,
  toolMode,
  onChangeToolMode,
  activeMaskId,
  activeMaskComponentId,
  creatingNewMask,
  brushOptions,
  colorPickActive,
  busyAiMaskKind,
  busyBgRemoval,
  onSelectMask,
  onRequestNewMask,
  onCancelNewMask,
  onChangeBrushOptions,
  onToggleColorPick,
  onCreateAiMask,
  onRemoveBackground,
  onChangeBackgroundRemoval,
  onToggleShadowClip,
  onToggleHighlightClip,
  onToggleEyedropper,
  onChangeRecipe,
  onApplyPreset,
  onApplyRelativeNudge,
  onSyncSlateWb,
  onOpenSyncModal,
  onCopyRecipe,
  onPasteRecipe,
}: DevelopInspectorPanelProps) {
  // Classic Lightroom Classic accordion state: ONLY Basic is expanded by default.
  const [openSections, setOpenSections] = useState<Record<EditAccordionKey, boolean>>({
    basic: true,
    toneCurve: false,
    colorMixer: false,
    colorGrading: false,
    detail: false,
    effects: false,
    calibration: false,
  });
  const [activeHslChannel, setActiveHslChannel] = useState<HslChannelKey>('orange');
  const [lastLockedAspect, setLastLockedAspect] = useState<Exclude<CropAspectRatio, 'free'>>('original');
  const [customPresets, setCustomPresets] = useState<readonly DevelopPreset[]>(() =>
    loadCustomPresetsFromStorage(),
  );
  const [customLuts, setCustomLuts] = useState<readonly CustomCubeLutEntry[]>(() =>
    loadCustomLutsFromStorage(),
  );
  const [lutImportError, setLutImportError] = useState<string | null>(null);
  const lutFileInputRef = useRef<HTMLInputElement>(null);
  const [newPresetName, setNewPresetName] = useState('');
  const [showNewPresetForm, setShowNewPresetForm] = useState(false);

  const builtInPresets = useMemo(() => getStudioDevelopPresets(), []);
  const recipe = photo.developRecipe;
  const modified = isRecipeModified(recipe);
  const modifiedModules = useMemo(() => getModifiedDevelopModules(recipe), [recipe]);
  const bgRemoval = recipe.backgroundRemoval ?? createDefaultBackgroundRemoval();
  const bgActive = isBackgroundRemovalActive(bgRemoval);
  const maskBadgeCount = (recipe.masks?.length ?? 0) + (bgActive ? 1 : 0);

  const toggleSection = (key: EditAccordionKey) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));

  const patchBasic = <K extends keyof BasicToneSettings>(key: K, val: BasicToneSettings[K]) => {
    onChangeRecipe(
      {
        ...recipe,
        basicTone: { ...recipe.basicTone, [key]: val },
      },
      'Basic Tone',
    );
  };

  const patchCurve = <K extends keyof ParametricToneCurve>(key: K, val: number) => {
    onChangeRecipe(
      {
        ...recipe,
        toneCurve: { ...recipe.toneCurve, [key]: val },
      },
      'Tone Curve',
    );
  };

  const patchHsl = (field: 'hue' | 'saturation' | 'luminance', val: number) => {
    onChangeRecipe(
      {
        ...recipe,
        hslMixer: {
          ...recipe.hslMixer,
          [activeHslChannel]: {
            ...recipe.hslMixer[activeHslChannel],
            [field]: val,
          },
        },
      },
      `HSL ${activeHslChannel}`,
    );
  };

  const patchCalibration = <K extends keyof CameraCalibrationSettings>(
    key: K,
    val: CameraCalibrationSettings[K],
  ) => {
    onChangeRecipe(
      {
        ...recipe,
        cameraCalibration: { ...recipe.cameraCalibration, [key]: val },
      },
      'Camera Calibration',
    );
  };

  const handleAutoTone = () => {
    const hist = liveHistogram ?? estimatePreviewHistogramBins(recipe);
    const nextBasic = computeAutoToneFromHistogram(hist, recipe.basicTone);
    onChangeRecipe({ ...recipe, basicTone: nextBasic }, 'Auto Tone', false);
  };

  const handleSaveCustomPreset = () => {
    const trimmed = newPresetName.trim();
    if (!trimmed) return;
    const created: DevelopPreset = {
      id: `custom-${Date.now()}`,
      name: trimmed,
      category: 'Custom',
      recipe,
      isCustom: true,
    };
    const next = [created, ...customPresets];
    setCustomPresets(next);
    saveCustomPresetsToStorage(next);
    setNewPresetName('');
    setShowNewPresetForm(false);
  };

  const handleDeleteCustomPreset = (presetId: string) => {
    const next = customPresets.filter((p) => p.id !== presetId);
    setCustomPresets(next);
    saveCustomPresetsToStorage(next);
  };

  const handleImportCubeLut = async () => {
    setLutImportError(null);
    if (isTauriDesktopEnvironment()) {
      try {
        const selected = await openNativeDialog({
          directory: false,
          multiple: false,
          title: 'Import 3D .cube LUT Profile',
          filters: [{ name: '3D Cube LUT (.cube)', extensions: ['cube'] }],
        });
        if (!selected || Array.isArray(selected)) return;
        const imported = await invoke<{
          name: string;
          cubePath: string;
          gridSize: number;
          content: string;
        }>('import_cube_lut_file', { sourcePath: selected });
        const parsed = parseCubeLut3D(imported.content);
        cacheParsedCubeLut3D(imported.cubePath, parsed);

        const entry: CustomCubeLutEntry = {
          id: `lut-${Date.now()}`,
          name: imported.name,
          cubePath: imported.cubePath,
          gridSize: imported.gridSize,
        };
        const deduped = customLuts.filter(
          (l) => l.cubePath !== entry.cubePath && l.name !== entry.name,
        );
        const next = [entry, ...deduped];
        setCustomLuts(next);
        saveCustomLutsToStorage(next);
        onChangeRecipe(
          {
            ...recipe,
            lut3d: { name: entry.name, cubePath: entry.cubePath, intensity: 75 },
          },
          `LUT ${entry.name}`,
          false,
        );
      } catch (err) {
        setLutImportError(err instanceof Error ? err.message : String(err));
      }
      return;
    }

    lutFileInputRef.current?.click();
  };

  const handleBrowserLutFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setLutImportError(null);
    try {
      const content = await file.text();
      const parsed = parseCubeLut3D(content);
      const cubePath = `custom-luts/${file.name}`;
      cacheParsedCubeLut3D(cubePath, parsed);

      const entry: CustomCubeLutEntry = {
        id: `lut-${Date.now()}`,
        name: file.name,
        cubePath,
        gridSize: parsed.size,
        inlineContent: content,
      };
      const deduped = customLuts.filter(
        (l) => l.cubePath !== entry.cubePath && l.name !== entry.name,
      );
      const next = [entry, ...deduped];
      setCustomLuts(next);
      saveCustomLutsToStorage(next);
      onChangeRecipe(
        {
          ...recipe,
          lut3d: { name: entry.name, cubePath: entry.cubePath, intensity: 75 },
        },
        `LUT ${entry.name}`,
        false,
      );
    } catch (err) {
      setLutImportError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteCustomLut = (lutId: string, cubePath: string) => {
    const next = customLuts.filter((l) => l.id !== lutId);
    setCustomLuts(next);
    saveCustomLutsToStorage(next);
    if (recipe.lut3d?.cubePath === cubePath) {
      onChangeRecipe({ ...recipe, lut3d: null }, 'Clear LUT', false);
    }
  };

  const matchedWbPresetId =
    WHITE_BALANCE_PRESETS.find(
      (p) =>
        p.temperatureKelvin === recipe.whiteBalance.temperatureKelvin &&
        p.tint === recipe.whiteBalance.tint,
    )?.id ?? 'custom';

  const currentHsl = recipe.hslMixer[activeHslChannel];
  const activeChannelMeta = HSL_CHANNELS.find((c) => c.key === activeHslChannel)!;
  const detailModified =
    recipe.detailOptics.sharpeningAmount !== 0 ||
    recipe.detailOptics.luminanceNoiseReduction !== 0 ||
    recipe.detailOptics.colorNoiseReduction !== 0;
  const effectsModified =
    recipe.detailOptics.vignetteAmount !== 0 ||
    recipe.detailOptics.grainAmount !== 0;

  return (
    <aside
      aria-label="Studio Develop Inspector"
      style={{ width: `${widthPx}px` }}
      className="shrink-0 bg-surface-low flex flex-col overflow-hidden"
    >
      {/* Top RGB Histogram + Authentic Lightroom 4-Icon Tool Strip */}
      <div className="p-3 pb-2 border-b border-hairline flex flex-col gap-2.5 shrink-0 bg-surface">
        <HistogramView
          recipe={recipe}
          exif={photo.exif}
          liveHistogram={liveHistogram}
          showShadowClip={showShadowClip}
          showHighlightClip={showHighlightClip}
          onToggleShadowClip={onToggleShadowClip}
          onToggleHighlightClip={onToggleHighlightClip}
        />

        {/* Lightroom Classic / CC Tool Strip directly beneath Histogram */}
        <nav aria-label="Develop Tools" className="grid grid-cols-4 gap-1">
          {TOOL_STRIP_ITEMS.map((item) => {
            const active = toolMode === item.id;
            const showCropDot = item.id === 'crop' && modifiedModules.cropTransform;
            const showMaskCount = item.id === 'mask' && maskBadgeCount > 0;
            return (
              <button
                key={item.id}
                type="button"
                title={item.title}
                aria-pressed={active}
                onClick={() => onChangeToolMode(item.id)}
                className={`relative py-1.5 px-1 rounded-lg flex flex-col items-center justify-center gap-0.5 transition-colors cursor-pointer ${
                  active
                    ? 'text-amber-soft'
                    : 'text-ink-secondary hover:text-ink-primary hover:bg-surface-low/80'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="lightroom-tool-strip-indicator"
                    transition={TACTILE_SPRING}
                    className="absolute inset-0 rounded-lg bg-surface-active border border-amber-core/50"
                  />
                )}
                <span className="relative z-10 flex items-center justify-center">
                  <span className="material-symbols-outlined text-[18px] leading-none">
                    {item.icon}
                  </span>
                  {showCropDot && (
                    <span className="absolute -top-0.5 -right-1.5 w-1.5 h-1.5 rounded-full bg-amber-core" />
                  )}
                  {showMaskCount && (
                    <span className="absolute -top-1.5 -right-3 min-w-[14px] h-[14px] px-1 rounded-full bg-amber-core text-amber-ink tabular-nums-mono text-[8.5px] font-bold flex items-center justify-center leading-none">
                      {maskBadgeCount}
                    </span>
                  )}
                </span>
                <span className="relative z-10 font-display text-[10px] font-medium leading-none">
                  {item.label}
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Scrollable Contextual Tool Drawer */}
      <div className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait" initial={false}>
          {/* 1. EDIT MODE: Classic Lightroom Collapsible Accordion Stack */}
          {toolMode === 'edit' && (
            <motion.div
              key="mode-edit"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={FAST_FADE}
              className="flex flex-col"
            >
              {/* Basic (Open by default) */}
              <AccordionSection
                title="Basic"
                open={openSections.basic}
                onToggle={() => toggleSection('basic')}
                modified={modifiedModules.whiteBalance || modifiedModules.basicTone}
                onReset={() => {
                  const def = createDefaultDevelopRecipe();
                  onChangeRecipe(
                    { ...recipe, whiteBalance: def.whiteBalance, basicTone: def.basicTone },
                    'Reset Basic',
                    false,
                  );
                }}
              >
                {/* Treatment + Auto Tone */}
                <div className="flex items-center justify-between gap-2">
                  <SegmentedControl
                    layoutId="develop-treatment-mode"
                    options={TREATMENT_OPTIONS}
                    value={recipe.basicTone.blackAndWhite ? 'bw' : 'color'}
                    onChange={(mode) =>
                      onChangeRecipe(
                        {
                          ...recipe,
                          basicTone: { ...recipe.basicTone, blackAndWhite: mode === 'bw' },
                        },
                        mode === 'bw' ? 'B&W Treatment' : 'Color Treatment',
                        false,
                      )
                    }
                  />
                  <DarkroomButton
                    variant="subtle"
                    size="xs"
                    icon="auto_awesome"
                    onClick={handleAutoTone}
                    title="Auto-balance Exposure, Contrast, Highlights, Shadows, Whites & Blacks"
                  >
                    Auto
                  </DarkroomButton>
                </div>

                {/* White Balance */}
                <div className="flex items-center gap-1.5 pt-1">
                  <DarkroomButton
                    variant={eyedropperActive ? 'emerald' : 'subtle'}
                    size="xs"
                    icon="colorize"
                    onClick={onToggleEyedropper}
                    title="WB Eyedropper (W) — click a neutral white/gray spot"
                    aria-label="White Balance Eyedropper"
                  />
                  <select
                    aria-label="White Balance preset"
                    value={matchedWbPresetId}
                    onChange={(e) => {
                      const chosen = WHITE_BALANCE_PRESETS.find((p) => p.id === e.target.value);
                      if (!chosen) return;
                      onChangeRecipe(
                        {
                          ...recipe,
                          whiteBalance: {
                            temperatureKelvin: chosen.temperatureKelvin,
                            tint: chosen.tint,
                          },
                        },
                        `WB ${chosen.label}`,
                        false,
                      );
                    }}
                    className="flex-1 min-w-0 h-7 px-2 rounded-md bg-canvas border border-hairline hover:border-hairline-strong text-[11px] text-ink-primary focus:outline-none focus:border-amber-core"
                  >
                    {matchedWbPresetId === 'custom' && <option value="custom">Custom</option>}
                    {WHITE_BALANCE_PRESETS.map((preset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.label} ({preset.temperatureKelvin}K)
                      </option>
                    ))}
                  </select>
                  {hasSlateAnchor && (
                    <DarkroomButton
                      variant="emerald"
                      size="xs"
                      onClick={onSyncSlateWb}
                      title="Auto WB from subject slate"
                    >
                      Slate
                    </DarkroomButton>
                  )}
                </div>

                <DarkroomScrubber
                  label="Temp"
                  value={recipe.whiteBalance.temperatureKelvin}
                  min={2000}
                  max={10500}
                  step={50}
                  defaultValue={5500}
                  formatValue={(v) => `${v}K`}
                  trackGradient="linear-gradient(90deg, #3B82F6 0%, #64748B 45%, #F59E0B 100%)"
                  onChange={(temperatureKelvin) =>
                    onChangeRecipe({
                      ...recipe,
                      whiteBalance: { ...recipe.whiteBalance, temperatureKelvin },
                    })
                  }
                />
                <DarkroomScrubber
                  label="Tint"
                  value={recipe.whiteBalance.tint}
                  min={-150}
                  max={150}
                  step={1}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  trackGradient="linear-gradient(90deg, #10B981 0%, #64748B 50%, #EC4899 100%)"
                  onChange={(tint) =>
                    onChangeRecipe({
                      ...recipe,
                      whiteBalance: { ...recipe.whiteBalance, tint },
                    })
                  }
                />

                <div className="h-px bg-hairline/70 my-0.5" />

                {/* Tone */}
                <DarkroomScrubber
                  label="Exposure"
                  value={recipe.basicTone.exposureEv}
                  min={-5}
                  max={5}
                  step={0.05}
                  defaultValue={0}
                  formatValue={(v) => `${formatSigned(v, 2)} EV`}
                  onChange={(val) => patchBasic('exposureEv', Number(val.toFixed(2)))}
                />
                <DarkroomScrubber
                  label="Contrast"
                  value={recipe.basicTone.contrast}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(val) => patchBasic('contrast', val)}
                />
                <DarkroomScrubber
                  label="Highlights"
                  value={recipe.basicTone.highlights}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(val) => patchBasic('highlights', val)}
                />
                <DarkroomScrubber
                  label="Shadows"
                  value={recipe.basicTone.shadows}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(val) => patchBasic('shadows', val)}
                />
                <DarkroomScrubber
                  label="Whites"
                  value={recipe.basicTone.whites}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(val) => patchBasic('whites', val)}
                />
                <DarkroomScrubber
                  label="Blacks"
                  value={recipe.basicTone.blacks}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(val) => patchBasic('blacks', val)}
                />

                <div className="h-px bg-hairline/70 my-0.5" />

                {/* Presence */}
                <DarkroomScrubber
                  label="Texture"
                  value={recipe.basicTone.texture}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(val) => patchBasic('texture', val)}
                />
                <DarkroomScrubber
                  label="Clarity"
                  value={recipe.basicTone.clarity}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(val) => patchBasic('clarity', val)}
                />
                <DarkroomScrubber
                  label="Dehaze"
                  value={recipe.basicTone.dehaze}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(val) => patchBasic('dehaze', val)}
                />
                {!recipe.basicTone.blackAndWhite ? (
                  <>
                    <DarkroomScrubber
                      label="Vibrance"
                      value={recipe.basicTone.vibrance}
                      min={-100}
                      max={100}
                      defaultValue={0}
                      formatValue={(v) => formatSigned(v)}
                      onChange={(val) => patchBasic('vibrance', val)}
                    />
                    <DarkroomScrubber
                      label="Saturation"
                      value={recipe.basicTone.saturation}
                      min={-100}
                      max={100}
                      defaultValue={0}
                      formatValue={(v) => formatSigned(v)}
                      onChange={(val) => patchBasic('saturation', val)}
                    />
                  </>
                ) : (
                  <p className="text-[10.5px] text-ink-telemetry bg-canvas px-2.5 py-1.5 rounded border border-hairline">
                    B&amp;W mode active — expand <strong className="text-ink-primary">B&amp;W Mix</strong> below to adjust tones.
                  </p>
                )}
              </AccordionSection>

              {/* Tone Curve */}
              <AccordionSection
                title="Tone Curve"
                open={openSections.toneCurve}
                onToggle={() => toggleSection('toneCurve')}
                modified={modifiedModules.toneCurve}
                onReset={() =>
                  onChangeRecipe(
                    { ...recipe, toneCurve: createDefaultDevelopRecipe().toneCurve },
                    'Reset Tone Curve',
                    false,
                  )
                }
              >
                <InteractiveToneCurveEditor
                  curve={recipe.toneCurve}
                  recipe={recipe}
                  liveHistogram={liveHistogram}
                  onChange={(nextCurve) =>
                    onChangeRecipe({ ...recipe, toneCurve: nextCurve }, 'Tone Curve')
                  }
                />
                <div className="h-px bg-hairline/70 my-0.5" />
                <DarkroomScrubber
                  label="Highlights"
                  value={recipe.toneCurve.highlights}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCurve('highlights', v)}
                />
                <DarkroomScrubber
                  label="Lights"
                  value={recipe.toneCurve.lights}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCurve('lights', v)}
                />
                <DarkroomScrubber
                  label="Darks"
                  value={recipe.toneCurve.darks}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCurve('darks', v)}
                />
                <DarkroomScrubber
                  label="Shadows"
                  value={recipe.toneCurve.shadows}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCurve('shadows', v)}
                />
              </AccordionSection>

              {/* Color Mixer / B&W Mix */}
              <AccordionSection
                title={recipe.basicTone.blackAndWhite ? 'B&W Mix' : 'Color Mixer'}
                open={openSections.colorMixer}
                onToggle={() => toggleSection('colorMixer')}
                modified={modifiedModules.hslMixer}
                onReset={() =>
                  onChangeRecipe(
                    { ...recipe, hslMixer: createDefaultDevelopRecipe().hslMixer },
                    'Reset Color Mixer',
                    false,
                  )
                }
              >
                {recipe.basicTone.blackAndWhite ? (
                  <div className="flex flex-col gap-2">
                    {HSL_CHANNELS.map((channel) => (
                      <DarkroomScrubber
                        key={channel.key}
                        label={channel.label}
                        value={recipe.hslMixer[channel.key].luminance}
                        min={-100}
                        max={100}
                        defaultValue={0}
                        formatValue={(v) => formatSigned(v)}
                        onChange={(luminance) =>
                          onChangeRecipe(
                            {
                              ...recipe,
                              hslMixer: {
                                ...recipe.hslMixer,
                                [channel.key]: { ...recipe.hslMixer[channel.key], luminance },
                              },
                            },
                            `B&W ${channel.label}`,
                          )
                        }
                      />
                    ))}
                  </div>
                ) : (
                  <>
                    <div className="flex items-center justify-between bg-canvas p-1.5 rounded-lg border border-hairline">
                      {HSL_CHANNELS.map((channel) => {
                        const isActive = channel.key === activeHslChannel;
                        const shift = recipe.hslMixer[channel.key];
                        const hasEdit =
                          shift.hue !== 0 || shift.saturation !== 0 || shift.luminance !== 0;

                        return (
                          <button
                            key={channel.key}
                            type="button"
                            onClick={() => setActiveHslChannel(channel.key)}
                            title={`${channel.label} channel`}
                            className="relative w-7 h-7 rounded-full flex items-center justify-center cursor-pointer"
                          >
                            {isActive && (
                              <motion.span
                                layoutId="hsl-active-dot"
                                transition={TACTILE_SPRING}
                                className="absolute inset-0 rounded-full border-2 border-amber-core bg-surface-active"
                              />
                            )}
                            <span
                              className="relative z-10 w-3.5 h-3.5 rounded-full block shadow-sm"
                              style={{ backgroundColor: channel.swatch }}
                            />
                            {hasEdit && (
                              <span className="absolute -bottom-0.5 w-1 h-1 rounded-full bg-amber-soft" />
                            )}
                          </button>
                        );
                      })}
                    </div>

                    <DarkroomScrubber
                      label={`${activeChannelMeta.label} Hue`}
                      value={currentHsl.hue}
                      min={-100}
                      max={100}
                      defaultValue={0}
                      formatValue={(v) => formatSigned(v)}
                      onChange={(v) => patchHsl('hue', v)}
                    />
                    <DarkroomScrubber
                      label={`${activeChannelMeta.label} Saturation`}
                      value={currentHsl.saturation}
                      min={-100}
                      max={100}
                      defaultValue={0}
                      formatValue={(v) => formatSigned(v)}
                      onChange={(v) => patchHsl('saturation', v)}
                    />
                    <DarkroomScrubber
                      label={`${activeChannelMeta.label} Luminance`}
                      value={currentHsl.luminance}
                      min={-100}
                      max={100}
                      defaultValue={0}
                      formatValue={(v) => formatSigned(v)}
                      onChange={(v) => patchHsl('luminance', v)}
                    />
                  </>
                )}
              </AccordionSection>

              {/* Color Grading (3-Way Wheels) */}
              <AccordionSection
                title="Color Grading"
                open={openSections.colorGrading}
                onToggle={() => toggleSection('colorGrading')}
                modified={modifiedModules.colorGrading}
                onReset={() =>
                  onChangeRecipe(
                    { ...recipe, colorGrading: createDefaultColorGrading() },
                    'Reset Color Grading',
                    false,
                  )
                }
              >
                <ColorGradingWheels
                  value={recipe.colorGrading}
                  onChange={(colorGrading) =>
                    onChangeRecipe({ ...recipe, colorGrading }, 'Color Grading')
                  }
                />
              </AccordionSection>

              {/* Detail (Sharpening & Noise Reduction) */}
              <AccordionSection
                title="Detail"
                open={openSections.detail}
                onToggle={() => toggleSection('detail')}
                modified={detailModified}
                onReset={() =>
                  onChangeRecipe(
                    {
                      ...recipe,
                      detailOptics: {
                        ...recipe.detailOptics,
                        sharpeningAmount: 0,
                        luminanceNoiseReduction: 0,
                        colorNoiseReduction: 0,
                      },
                    },
                    'Reset Detail',
                    false,
                  )
                }
              >
                <DarkroomScrubber
                  label="Sharpening"
                  value={recipe.detailOptics.sharpeningAmount}
                  min={0}
                  max={150}
                  defaultValue={0}
                  formatValue={(v) => String(v)}
                  onChange={(sharpeningAmount) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        detailOptics: { ...recipe.detailOptics, sharpeningAmount },
                      },
                      'Sharpening',
                    )
                  }
                />
                <DarkroomScrubber
                  label="Luminance NR"
                  value={recipe.detailOptics.luminanceNoiseReduction}
                  min={0}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => String(v)}
                  onChange={(luminanceNoiseReduction) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        detailOptics: { ...recipe.detailOptics, luminanceNoiseReduction },
                      },
                      'Luminance NR',
                    )
                  }
                />
                <DarkroomScrubber
                  label="Color NR"
                  value={recipe.detailOptics.colorNoiseReduction}
                  min={0}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => String(v)}
                  onChange={(colorNoiseReduction) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        detailOptics: { ...recipe.detailOptics, colorNoiseReduction },
                      },
                      'Color NR',
                    )
                  }
                />
              </AccordionSection>

              {/* Effects (Vignette & Film Grain) */}
              <AccordionSection
                title="Effects"
                open={openSections.effects}
                onToggle={() => toggleSection('effects')}
                modified={effectsModified}
                onReset={() =>
                  onChangeRecipe(
                    {
                      ...recipe,
                      detailOptics: {
                        ...recipe.detailOptics,
                        vignetteAmount: 0,
                        vignetteMidpoint: 50,
                        vignetteFeather: 50,
                        grainAmount: 0,
                        grainSize: 25,
                      },
                    },
                    'Reset Effects',
                    false,
                  )
                }
              >
                <DarkroomScrubber
                  label="Vignette"
                  value={recipe.detailOptics.vignetteAmount}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(vignetteAmount) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        detailOptics: { ...recipe.detailOptics, vignetteAmount },
                      },
                      'Vignette',
                    )
                  }
                />
                <DarkroomScrubber
                  label="Midpoint"
                  value={recipe.detailOptics.vignetteMidpoint}
                  min={0}
                  max={100}
                  defaultValue={50}
                  formatValue={(v) => String(v)}
                  onChange={(vignetteMidpoint) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        detailOptics: { ...recipe.detailOptics, vignetteMidpoint },
                      },
                      'Vignette Midpoint',
                    )
                  }
                />
                <DarkroomScrubber
                  label="Feather"
                  value={recipe.detailOptics.vignetteFeather}
                  min={0}
                  max={100}
                  defaultValue={50}
                  formatValue={(v) => String(v)}
                  onChange={(vignetteFeather) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        detailOptics: { ...recipe.detailOptics, vignetteFeather },
                      },
                      'Vignette Feather',
                    )
                  }
                />

                <div className="h-px bg-hairline/70 my-0.5" />

                <DarkroomScrubber
                  label="Grain Amount"
                  value={recipe.detailOptics.grainAmount}
                  min={0}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => String(v)}
                  onChange={(grainAmount) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        detailOptics: { ...recipe.detailOptics, grainAmount },
                      },
                      'Film Grain',
                    )
                  }
                />
                <DarkroomScrubber
                  label="Grain Size"
                  value={recipe.detailOptics.grainSize}
                  min={0}
                  max={100}
                  defaultValue={25}
                  formatValue={(v) => String(v)}
                  onChange={(grainSize) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        detailOptics: { ...recipe.detailOptics, grainSize },
                      },
                      'Film Grain Size',
                    )
                  }
                />
              </AccordionSection>

              {/* Camera Calibration */}
              <AccordionSection
                title="Calibration"
                open={openSections.calibration}
                onToggle={() => toggleSection('calibration')}
                modified={modifiedModules.cameraCalibration}
                onReset={() =>
                  onChangeRecipe(
                    { ...recipe, cameraCalibration: createDefaultCameraCalibration() },
                    'Reset Calibration',
                    false,
                  )
                }
              >
                <DarkroomScrubber
                  label="Shadows Tint"
                  value={recipe.cameraCalibration.shadowsTint}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  trackGradient="linear-gradient(90deg, #10B981 0%, #64748B 50%, #EC4899 100%)"
                  onChange={(v) => patchCalibration('shadowsTint', v)}
                />
                <div className="h-px bg-hairline/60 my-0.5" />
                <span className="tabular-nums-mono text-[9.5px] text-red-400 font-semibold uppercase tracking-wider">
                  Red Primary
                </span>
                <DarkroomScrubber
                  label="Hue"
                  value={recipe.cameraCalibration.redHue}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCalibration('redHue', v)}
                />
                <DarkroomScrubber
                  label="Saturation"
                  value={recipe.cameraCalibration.redSaturation}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCalibration('redSaturation', v)}
                />
                <div className="h-px bg-hairline/60 my-0.5" />
                <span className="tabular-nums-mono text-[9.5px] text-emerald-400 font-semibold uppercase tracking-wider">
                  Green Primary
                </span>
                <DarkroomScrubber
                  label="Hue"
                  value={recipe.cameraCalibration.greenHue}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCalibration('greenHue', v)}
                />
                <DarkroomScrubber
                  label="Saturation"
                  value={recipe.cameraCalibration.greenSaturation}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCalibration('greenSaturation', v)}
                />
                <div className="h-px bg-hairline/60 my-0.5" />
                <span className="tabular-nums-mono text-[9.5px] text-blue-400 font-semibold uppercase tracking-wider">
                  Blue Primary
                </span>
                <DarkroomScrubber
                  label="Hue"
                  value={recipe.cameraCalibration.blueHue}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCalibration('blueHue', v)}
                />
                <DarkroomScrubber
                  label="Saturation"
                  value={recipe.cameraCalibration.blueSaturation}
                  min={-100}
                  max={100}
                  defaultValue={0}
                  formatValue={(v) => formatSigned(v)}
                  onChange={(v) => patchCalibration('blueSaturation', v)}
                />
              </AccordionSection>
            </motion.div>
          )}

          {/* 2. CROP MODE: Dedicated Lightroom Crop & Straighten Drawer */}
          {toolMode === 'crop' && (
            <motion.div
              key="mode-crop"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={FAST_FADE}
              className="p-3 flex flex-col gap-3"
            >
              <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-display text-[12px] font-semibold text-ink-primary">
                    Crop &amp; Straighten
                  </span>
                  {modifiedModules.cropTransform && (
                    <button
                      type="button"
                      onClick={() =>
                        onChangeRecipe(
                          { ...recipe, cropTransform: createDefaultCropTransform() },
                          'Reset Crop',
                          false,
                        )
                      }
                      className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                    >
                      Reset
                    </button>
                  )}
                </div>

                {/* Lightroom Aspect Ratio Dropdown + Lock Toggle */}
                <div className="flex flex-col gap-1.5">
                  <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                    Aspect
                  </span>
                  <div className="flex items-center gap-1.5">
                    <select
                      aria-label="Crop Aspect Ratio"
                      value={recipe.cropTransform.aspectRatio}
                      onChange={(e) => {
                        const nextAspect = e.target.value as CropAspectRatio;
                        if (nextAspect !== 'free') {
                          setLastLockedAspect(nextAspect);
                        }
                        const label =
                          ASPECT_OPTIONS.find((o) => o.id === nextAspect)?.label ?? nextAspect;
                        onChangeRecipe(
                          {
                            ...recipe,
                            cropTransform: {
                              ...recipe.cropTransform,
                              aspectRatio: nextAspect,
                              rect:
                                nextAspect === 'free'
                                  ? recipe.cropTransform.rect
                                  : fitCenteredCropForAspect(nextAspect, naturalAspect),
                            },
                          },
                          `Aspect ${label}`,
                          false,
                        );
                      }}
                      className="flex-1 min-w-0 h-7 px-2 rounded-md bg-canvas border border-hairline hover:border-hairline-strong text-[11px] text-ink-primary focus:outline-none focus:border-amber-core"
                    >
                      {ASPECT_OPTIONS.map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </select>

                    <button
                      type="button"
                      title={
                        recipe.cropTransform.aspectRatio === 'free'
                          ? 'Lock aspect ratio'
                          : 'Unlock for freeform crop'
                      }
                      aria-label={
                        recipe.cropTransform.aspectRatio === 'free'
                          ? 'Lock aspect ratio'
                          : 'Unlock aspect ratio'
                      }
                      aria-pressed={recipe.cropTransform.aspectRatio !== 'free'}
                      onClick={() => {
                        const isCurrentlyFree = recipe.cropTransform.aspectRatio === 'free';
                        if (isCurrentlyFree) {
                          onChangeRecipe(
                            {
                              ...recipe,
                              cropTransform: {
                                ...recipe.cropTransform,
                                aspectRatio: lastLockedAspect,
                                rect: fitCenteredCropForAspect(lastLockedAspect, naturalAspect),
                              },
                            },
                            'Lock Aspect Ratio',
                            false,
                          );
                        } else {
                          setLastLockedAspect(recipe.cropTransform.aspectRatio as Exclude<CropAspectRatio, 'free'>);
                          onChangeRecipe(
                            {
                              ...recipe,
                              cropTransform: {
                                ...recipe.cropTransform,
                                aspectRatio: 'free',
                              },
                            },
                            'Unlock Aspect Ratio',
                            false,
                          );
                        }
                      }}
                      className={`h-7 w-7 rounded-md border flex items-center justify-center shrink-0 transition-colors cursor-pointer ${
                        recipe.cropTransform.aspectRatio !== 'free'
                          ? 'bg-surface-active border-amber-core/55 text-amber-soft'
                          : 'bg-canvas border-hairline text-ink-telemetry hover:text-ink-primary'
                      }`}
                    >
                      <span className="material-symbols-outlined text-[15px]">
                        {recipe.cropTransform.aspectRatio !== 'free' ? 'lock' : 'lock_open'}
                      </span>
                    </button>
                  </div>
                </div>

                {/* Straighten */}
                <DarkroomScrubber
                  label="Angle"
                  value={recipe.cropTransform.rotationDeg}
                  min={-45}
                  max={45}
                  step={0.1}
                  defaultValue={0}
                  formatValue={(v) => `${formatSigned(v, 1)}°`}
                  onChange={(rotationDeg) =>
                    onChangeRecipe(
                      {
                        ...recipe,
                        cropTransform: {
                          ...recipe.cropTransform,
                          rotationDeg: Number(rotationDeg.toFixed(1)),
                        },
                      },
                      'Straighten',
                    )
                  }
                />

                {/* Rotate & Flip */}
                <div className="flex flex-col gap-1.5">
                  <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                    Rotate &amp; Flip
                  </span>
                  <div className="grid grid-cols-3 gap-1.5">
                    <DarkroomButton
                      variant={recipe.cropTransform.flipHorizontal ? 'secondary' : 'subtle'}
                      size="xs"
                      icon="flip"
                      onClick={() =>
                        onChangeRecipe(
                          {
                            ...recipe,
                            cropTransform: {
                              ...recipe.cropTransform,
                              flipHorizontal: !recipe.cropTransform.flipHorizontal,
                            },
                          },
                          'Flip Horizontal',
                          false,
                        )
                      }
                    >
                      Flip H
                    </DarkroomButton>
                    <DarkroomButton
                      variant={recipe.cropTransform.flipVertical ? 'secondary' : 'subtle'}
                      size="xs"
                      icon="flip"
                      onClick={() =>
                        onChangeRecipe(
                          {
                            ...recipe,
                            cropTransform: {
                              ...recipe.cropTransform,
                              flipVertical: !recipe.cropTransform.flipVertical,
                            },
                          },
                          'Flip Vertical',
                          false,
                        )
                      }
                    >
                      Flip V
                    </DarkroomButton>
                    <DarkroomButton
                      variant="subtle"
                      size="xs"
                      icon="rotate_90_degrees_cw"
                      onClick={() =>
                        onChangeRecipe(
                          {
                            ...recipe,
                            cropTransform: {
                              ...recipe.cropTransform,
                              quarterTurns: ((recipe.cropTransform.quarterTurns ?? 0) + 1) % 4,
                            },
                          },
                          'Rotate 90°',
                          false,
                        )
                      }
                    >
                      90° CW
                    </DarkroomButton>
                  </div>
                </div>

                <DarkroomToggle
                  checked={recipe.cropTransform.eyeLevelAutoLocked}
                  onChange={(eyeLevelAutoLocked) =>
                    onChangeRecipe({
                      ...recipe,
                      cropTransform: {
                        ...recipe.cropTransform,
                        eyeLevelAutoLocked,
                        rotationDeg: eyeLevelAutoLocked ? 0.6 : 0,
                      },
                    })
                  }
                  label="Auto-Level Eye Horizon"
                />

                <DarkroomButton
                  variant="primary"
                  size="sm"
                  icon="check"
                  onClick={() => onChangeToolMode('edit')}
                  className="w-full justify-center mt-1"
                >
                  Done (Enter)
                </DarkroomButton>
              </section>
            </motion.div>
          )}

          {/* 3. MASK MODE: Background Cutout & Local Masking */}
          {toolMode === 'mask' && (
            <motion.div
              key="mode-mask"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={FAST_FADE}
              className="p-3 flex flex-col gap-3"
            >
              <MaskingPanel
                masks={recipe.masks ?? []}
                backgroundRemoval={bgRemoval}
                activeMaskId={activeMaskId}
                activeComponentId={activeMaskComponentId}
                creatingNewMask={creatingNewMask}
                brushOptions={brushOptions}
                colorPickActive={colorPickActive}
                busyAiKind={busyAiMaskKind}
                busyBgRemoval={busyBgRemoval}
                onSelectMask={onSelectMask}
                onRequestNewMask={onRequestNewMask}
                onCancelNewMask={onCancelNewMask}
                onChangeBrushOptions={onChangeBrushOptions}
                onToggleColorPick={onToggleColorPick}
                onCreateAiMask={onCreateAiMask}
                onRemoveBackground={onRemoveBackground}
                onChangeBackgroundRemoval={onChangeBackgroundRemoval}
                onChangeMasks={(nextMasks: readonly MaskLayer[], label: string, coalesce = true) =>
                  onChangeRecipe({ ...recipe, masks: nextMasks }, label, coalesce)
                }
              />
            </motion.div>
          )}

          {/* 4. PRESETS MODE: Quick Develop + Custom & Built-in Presets */}
          {toolMode === 'presets' && (
            <motion.div
              key="mode-presets"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={FAST_FADE}
              className="p-3 flex flex-col gap-3"
            >
              {/* Quick Develop Relative Batch Nudges */}
              <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                    Quick Develop
                  </span>
                  <TelemetryPill tone="amber">
                    {selectedCount > 0 ? `${selectedCount} sel` : '1 active'}
                  </TelemetryPill>
                </div>
                <div className="grid grid-cols-2 gap-1.5">
                  {QUICK_NUDGES.map((nudge) => (
                    <DarkroomButton
                      key={nudge.id}
                      variant="subtle"
                      size="xs"
                      onClick={() => onApplyRelativeNudge(nudge.delta, nudge.label)}
                      title={nudge.label}
                      className="tabular-nums-mono justify-center"
                    >
                      {nudge.shortLabel}
                    </DarkroomButton>
                  ))}
                </div>
              </section>

              {/* User Custom Saved Presets */}
              <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                    My Presets ({customPresets.length})
                  </span>
                  <DarkroomButton
                    variant="subtle"
                    size="xs"
                    icon={showNewPresetForm ? 'close' : 'add'}
                    onClick={() => setShowNewPresetForm((v) => !v)}
                  >
                    {showNewPresetForm ? 'Cancel' : 'Save Look'}
                  </DarkroomButton>
                </div>

                {showNewPresetForm && (
                  <div className="flex items-center gap-1.5">
                    <input
                      type="text"
                      value={newPresetName}
                      onChange={(e) => setNewPresetName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveCustomPreset();
                        if (e.key === 'Escape') setShowNewPresetForm(false);
                      }}
                      placeholder="Preset name (e.g. Clean Studio WB)"
                      className="flex-1 min-w-0 h-7 px-2 rounded-md bg-canvas border border-amber-core/60 text-[11px] text-ink-primary placeholder:text-ink-telemetry focus:outline-none"
                    />
                    <DarkroomButton
                      variant="primary"
                      size="xs"
                      disabled={!newPresetName.trim()}
                      onClick={handleSaveCustomPreset}
                    >
                      Save
                    </DarkroomButton>
                  </div>
                )}

                {customPresets.length === 0 ? (
                  <p className="text-[10.5px] text-ink-telemetry">
                    Save your current sliders, curves &amp; color grading as a reusable preset.
                  </p>
                ) : (
                  <div className="flex flex-col gap-1.5">
                    {customPresets.map((preset) => (
                      <div
                        key={preset.id}
                        className="p-2 rounded-lg bg-surface-low border border-hairline hover:bg-surface-active hover:border-amber-core/45 transition-colors flex items-center justify-between gap-2 group"
                      >
                        <button
                          type="button"
                          onClick={() => onApplyPreset(preset.name, preset.recipe)}
                          className="flex-1 min-w-0 text-left cursor-pointer"
                        >
                          <div className="text-[12px] text-ink-primary font-medium truncate group-hover:text-amber-soft transition-colors">
                            {preset.name}
                          </div>
                          <div className="tabular-nums-mono text-[9.5px] text-ink-telemetry">
                            Custom • {preset.recipe.whiteBalance.temperatureKelvin}K
                          </div>
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteCustomPreset(preset.id)}
                          title={`Delete ${preset.name}`}
                          aria-label={`Delete preset ${preset.name}`}
                          className="p-1 rounded text-ink-telemetry hover:text-crimson-clip cursor-pointer shrink-0"
                        >
                          <span className="material-symbols-outlined text-[14px]">delete</span>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              {/* Built-In Studio Develop Presets */}
              <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-2">
                <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                  Studio Presets
                </span>
                <div className="flex flex-col gap-1.5">
                  {builtInPresets.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => onApplyPreset(preset.name, preset.recipe)}
                      className="p-2 rounded-lg bg-surface-low border border-hairline hover:bg-surface-active hover:border-amber-core/45 transition-colors text-left flex items-center justify-between gap-2 cursor-pointer group"
                    >
                      <div className="min-w-0">
                        <div className="text-[12px] text-ink-primary font-medium truncate group-hover:text-amber-soft transition-colors">
                          {preset.name}
                        </div>
                        <div className="tabular-nums-mono text-[9.5px] text-ink-telemetry">
                          {preset.category} • {preset.recipe.whiteBalance.temperatureKelvin}K
                        </div>
                      </div>
                      <span className="material-symbols-outlined text-[15px] text-ink-telemetry group-hover:text-amber-core shrink-0">
                        auto_fix_high
                      </span>
                    </button>
                  ))}
                </div>
              </section>

              {/* 3D .cube LUT Profiles (Custom Uploads + Built-In Studio LUTs) */}
              <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                    3D .cube LUT Profile
                  </span>
                  <DarkroomButton
                    variant="subtle"
                    size="xs"
                    icon="upload_file"
                    onClick={() => void handleImportCubeLut()}
                    title="Import a custom 3D .cube LUT file from disk"
                  >
                    Import .cube…
                  </DarkroomButton>
                </div>

                <input
                  ref={lutFileInputRef}
                  type="file"
                  accept=".cube"
                  onChange={(e) => void handleBrowserLutFileSelected(e)}
                  className="hidden"
                />

                {lutImportError && (
                  <div className="px-2.5 py-1.5 rounded-md bg-crimson-clip/10 border border-crimson-clip/40 text-[10.5px] text-crimson-clip">
                    {lutImportError}
                  </div>
                )}

                {recipe.lut3d && (
                  <div className="pb-1">
                    <DarkroomScrubber
                      label="LUT Intensity"
                      value={recipe.lut3d.intensity}
                      min={0}
                      max={100}
                      defaultValue={65}
                      formatValue={(v) => `${v}%`}
                      onChange={(intensity) =>
                        onChangeRecipe(
                          {
                            ...recipe,
                            lut3d: { ...recipe.lut3d!, intensity },
                          },
                          'LUT Intensity',
                        )
                      }
                    />
                  </div>
                )}

                <div className="flex flex-col gap-1">
                  {STUDIO_LUTS.map((entry) => {
                    const isSelected =
                      (entry.lut === null && recipe.lut3d === null) ||
                      (entry.lut !== null && recipe.lut3d?.cubePath === entry.lut.cubePath);

                    return (
                      <button
                        key={entry.label}
                        type="button"
                        onClick={() =>
                          onChangeRecipe(
                            {
                              ...recipe,
                              lut3d: entry.lut ? { ...entry.lut } : null,
                            },
                            entry.lut ? `LUT ${entry.lut.name}` : 'Clear LUT',
                            false,
                          )
                        }
                        className={`px-2.5 py-1.5 rounded-md border text-left tabular-nums-mono text-[10.5px] flex items-center justify-between transition-colors cursor-pointer ${
                          isSelected
                            ? 'bg-surface-active border-amber-core/60 text-amber-soft font-medium'
                            : 'bg-canvas border-hairline text-ink-secondary hover:text-ink-primary'
                        }`}
                      >
                        <span className="truncate">{entry.label}</span>
                        {isSelected && (
                          <span className="material-symbols-outlined text-[13px] text-amber-core shrink-0">
                            check
                          </span>
                        )}
                      </button>
                    );
                  })}

                  {customLuts.map((custom) => {
                    const isSelected = recipe.lut3d?.cubePath === custom.cubePath;
                    return (
                      <div
                        key={custom.id}
                        className={`px-2.5 py-1.5 rounded-md border tabular-nums-mono text-[10.5px] flex items-center justify-between gap-2 transition-colors ${
                          isSelected
                            ? 'bg-surface-active border-amber-core/60 text-amber-soft font-medium'
                            : 'bg-canvas border-hairline text-ink-secondary hover:text-ink-primary'
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() =>
                            onChangeRecipe(
                              {
                                ...recipe,
                                lut3d: {
                                  name: custom.name,
                                  cubePath: custom.cubePath,
                                  intensity: recipe.lut3d?.intensity ?? 75,
                                },
                              },
                              `LUT ${custom.name}`,
                              false,
                            )
                          }
                          className="flex-1 min-w-0 text-left flex items-center justify-between gap-2 cursor-pointer"
                        >
                          <span className="truncate">{custom.name}</span>
                          <span className="text-[9px] text-ink-telemetry shrink-0">
                            {custom.gridSize}³
                          </span>
                        </button>
                        {isSelected && (
                          <span className="material-symbols-outlined text-[13px] text-amber-core shrink-0">
                            check
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={() => handleDeleteCustomLut(custom.id, custom.cubePath)}
                          title={`Remove ${custom.name}`}
                          aria-label={`Remove LUT ${custom.name}`}
                          className="text-ink-telemetry hover:text-crimson-clip cursor-pointer shrink-0 flex items-center"
                        >
                          <span className="material-symbols-outlined text-[13px]">delete</span>
                        </button>
                      </div>
                    );
                  })}
                </div>
              </section>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Pinned Lightroom Batch Sync Command Footer */}
      <div className="p-3 border-t border-hairline bg-surface flex flex-col gap-2 shrink-0">
        <div className="grid grid-cols-3 gap-1.5">
          <DarkroomButton
            variant="subtle"
            size="xs"
            icon="restart_alt"
            disabled={!modified}
            onClick={() => onChangeRecipe(createDefaultDevelopRecipe(), 'Reset edits', false)}
            title="Reset all edits"
          >
            Reset
          </DarkroomButton>
          <DarkroomButton
            variant="secondary"
            size="xs"
            icon="content_copy"
            onClick={onCopyRecipe}
            title="Copy recipe (Ctrl+Shift+C)"
          >
            Copy
          </DarkroomButton>
          <DarkroomButton
            variant="secondary"
            size="xs"
            icon="content_paste"
            disabled={!hasCopiedRecipe}
            onClick={onPasteRecipe}
            title="Paste recipe (Ctrl+Shift+V)"
          >
            Paste
          </DarkroomButton>
        </div>

        <DarkroomButton
          variant="primary"
          size="md"
          icon="tune"
          onClick={onOpenSyncModal}
          title="Edits on this photo auto-save automatically. Click to copy settings to other photos (Ctrl+Shift+S)"
          className="w-full justify-between"
        >
          <span>Sync to Other Photos</span>
          <kbd className="tabular-nums-mono text-[9.5px] opacity-75">Ctrl+Shift+S</kbd>
        </DarkroomButton>
      </div>
    </aside>
  );
});
