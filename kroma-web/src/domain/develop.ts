import {
  createDefaultBackgroundRemoval,
  hydrateBackgroundRemoval,
  hydrateMaskLayers,
  isBackgroundRemovalActive,
  isMaskLayerEffective,
  type BackgroundRemovalSettings,
  type MaskLayer,
} from './masking';
import { isToneCurveModified, type ToneCurvePoint } from './toneCurve';

export type { ToneCurveChannel, ToneCurvePoint } from './toneCurve';

export interface WhiteBalanceSettings {
  readonly temperatureKelvin: number; // 2000..50000
  readonly tint: number; // -150..+150
}

export interface WhiteBalancePreset {
  readonly id: string;
  readonly label: string;
  readonly temperatureKelvin: number;
  readonly tint: number;
}

export const WHITE_BALANCE_PRESETS: readonly WhiteBalancePreset[] = [
  { id: 'as-shot', label: 'As Shot', temperatureKelvin: 5500, tint: 0 },
  { id: 'daylight', label: 'Daylight', temperatureKelvin: 5500, tint: 10 },
  { id: 'cloudy', label: 'Cloudy', temperatureKelvin: 6500, tint: 10 },
  { id: 'shade', label: 'Shade', temperatureKelvin: 7500, tint: 10 },
  { id: 'tungsten', label: 'Tungsten', temperatureKelvin: 2850, tint: 0 },
  { id: 'fluorescent', label: 'Fluorescent', temperatureKelvin: 3800, tint: 21 },
  { id: 'flash', label: 'Flash', temperatureKelvin: 5500, tint: 0 },
];

export interface BasicToneSettings {
  readonly blackAndWhite: boolean;
  readonly exposureEv: number; // -5.0..+5.0
  readonly contrast: number; // -100..+100
  readonly highlights: number; // -100..+100
  readonly shadows: number; // -100..+100
  readonly whites: number; // -100..+100
  readonly blacks: number; // -100..+100
  readonly texture: number; // -100..+100
  readonly clarity: number; // -100..+100
  readonly dehaze: number; // -100..+100
  readonly vibrance: number; // -100..+100
  readonly saturation: number; // -100..+100
}

export interface ParametricToneCurve {
  readonly highlights: number;
  readonly lights: number;
  readonly darks: number;
  readonly shadows: number;
  readonly points?: readonly ToneCurvePoint[];
  readonly redPoints?: readonly ToneCurvePoint[];
  readonly greenPoints?: readonly ToneCurvePoint[];
  readonly bluePoints?: readonly ToneCurvePoint[];
}

export interface HslChannelShift {
  readonly hue: number;
  readonly saturation: number;
  readonly luminance: number;
}

export type HslChannelKey =
  | 'red'
  | 'orange'
  | 'yellow'
  | 'green'
  | 'aqua'
  | 'blue'
  | 'purple'
  | 'magenta';

export type HslColorMixer = Readonly<Record<HslChannelKey, HslChannelShift>>;

export interface ColorGradingWheel {
  readonly hueDeg: number; // 0..360
  readonly saturation: number; // 0..100
  readonly luminance: number; // -100..+100
}

export interface ColorGradingSettings {
  readonly shadows: ColorGradingWheel;
  readonly midtones: ColorGradingWheel;
  readonly highlights: ColorGradingWheel;
  readonly blending: number; // 0..100 (default 50)
  readonly balance: number; // -100..+100 (default 0)
}

export interface CameraCalibrationSettings {
  readonly shadowsTint: number; // -100..+100
  readonly redHue: number; // -100..+100
  readonly redSaturation: number; // -100..+100
  readonly greenHue: number; // -100..+100
  readonly greenSaturation: number; // -100..+100
  readonly blueHue: number; // -100..+100
  readonly blueSaturation: number; // -100..+100
}

export interface Lut3DSettings {
  readonly name: string;
  readonly cubePath: string;
  readonly intensity: number; // 0..100
}

export interface DetailOpticsSettings {
  readonly sharpeningAmount: number; // 0..150
  readonly luminanceNoiseReduction: number; // 0..100
  readonly colorNoiseReduction: number; // 0..100
  readonly vignetteAmount: number; // -100..+100
  readonly vignetteMidpoint: number; // 0..100 (default 50)
  readonly vignetteFeather: number; // 0..100 (default 50)
  readonly grainAmount: number; // 0..100 (default 0)
  readonly grainSize: number; // 0..100 (default 25)
}

export interface CropRect {
  readonly left: number; // 0..1
  readonly top: number; // 0..1
  readonly width: number; // 0..1
  readonly height: number; // 0..1
}

export type CropAspectRatio = 'original' | 'free' | '4:5' | '2:3' | '5:7' | '1:1' | '16:9';

export interface CropTransformSettings {
  readonly rect: CropRect;
  readonly rotationDeg: number; // -45..+45
  readonly aspectRatio: CropAspectRatio;
  readonly flipHorizontal: boolean;
  readonly flipVertical: boolean;
  readonly quarterTurns: number; // 0..3 (90° CW steps)
  readonly eyeLevelAutoLocked: boolean;
}

export interface DevelopRecipe {
  readonly whiteBalance: WhiteBalanceSettings;
  readonly basicTone: BasicToneSettings;
  readonly toneCurve: ParametricToneCurve;
  readonly hslMixer: HslColorMixer;
  readonly colorGrading: ColorGradingSettings;
  readonly cameraCalibration: CameraCalibrationSettings;
  readonly lut3d: Lut3DSettings | null;
  readonly detailOptics: DetailOpticsSettings;
  readonly cropTransform: CropTransformSettings;
  readonly masks: readonly MaskLayer[];
  readonly backgroundRemoval?: BackgroundRemovalSettings;
}

export interface DevelopSyncMask {
  readonly whiteBalance: boolean;
  readonly basicTone: boolean;
  readonly toneCurve: boolean;
  readonly hslMixer: boolean;
  readonly colorGrading: boolean;
  readonly cameraCalibration: boolean;
  readonly lut3d: boolean;
  readonly detailOptics: boolean;
  readonly cropTransform: boolean;
  readonly masks: boolean;
}

export interface RelativeDevelopDelta {
  readonly exposureDeltaEv?: number;
  readonly temperatureDeltaKelvin?: number;
  readonly tintDelta?: number;
  readonly contrastDelta?: number;
  readonly highlightsDelta?: number;
  readonly shadowsDelta?: number;
  readonly vibranceDelta?: number;
  readonly saturationDelta?: number;
}

export interface DevelopPreset {
  readonly id: string;
  readonly name: string;
  readonly category: 'Portrait' | 'Editorial' | 'Monochrome' | 'Utility' | 'Custom';
  readonly recipe: DevelopRecipe;
  readonly isCustom?: boolean;
}

export const FULL_FRAME_CROP_RECT: CropRect = { left: 0, top: 0, width: 1, height: 1 };
const ZERO_HSL: HslChannelShift = { hue: 0, saturation: 0, luminance: 0 };
const ZERO_WHEEL: ColorGradingWheel = { hueDeg: 0, saturation: 0, luminance: 0 };

export function createDefaultColorGrading(): ColorGradingSettings {
  return {
    shadows: { ...ZERO_WHEEL },
    midtones: { ...ZERO_WHEEL },
    highlights: { ...ZERO_WHEEL },
    blending: 50,
    balance: 0,
  };
}

export function createDefaultCameraCalibration(): CameraCalibrationSettings {
  return {
    shadowsTint: 0,
    redHue: 0,
    redSaturation: 0,
    greenHue: 0,
    greenSaturation: 0,
    blueHue: 0,
    blueSaturation: 0,
  };
}

export function createDefaultCropTransform(): CropTransformSettings {
  return {
    rect: { ...FULL_FRAME_CROP_RECT },
    rotationDeg: 0,
    aspectRatio: 'original',
    flipHorizontal: false,
    flipVertical: false,
    quarterTurns: 0,
    eyeLevelAutoLocked: false,
  };
}

export function createDefaultDevelopRecipe(): DevelopRecipe {
  return {
    whiteBalance: { temperatureKelvin: 5500, tint: 0 },
    basicTone: {
      blackAndWhite: false,
      exposureEv: 0,
      contrast: 0,
      highlights: 0,
      shadows: 0,
      whites: 0,
      blacks: 0,
      texture: 0,
      clarity: 0,
      dehaze: 0,
      vibrance: 0,
      saturation: 0,
    },
    toneCurve: { highlights: 0, lights: 0, darks: 0, shadows: 0 },
    hslMixer: {
      red: { ...ZERO_HSL },
      orange: { ...ZERO_HSL },
      yellow: { ...ZERO_HSL },
      green: { ...ZERO_HSL },
      aqua: { ...ZERO_HSL },
      blue: { ...ZERO_HSL },
      purple: { ...ZERO_HSL },
      magenta: { ...ZERO_HSL },
    },
    colorGrading: createDefaultColorGrading(),
    cameraCalibration: createDefaultCameraCalibration(),
    lut3d: null,
    detailOptics: {
      sharpeningAmount: 0,
      luminanceNoiseReduction: 0,
      colorNoiseReduction: 0,
      vignetteAmount: 0,
      vignetteMidpoint: 50,
      vignetteFeather: 50,
      grainAmount: 0,
      grainSize: 25,
    },
    cropTransform: createDefaultCropTransform(),
    masks: [],
    backgroundRemoval: createDefaultBackgroundRemoval(),
  };
}

/**
 * Safely hydrates a partial or legacy saved DevelopRecipe JSON object with defaults for any
 * newly introduced Phase 1+ fields (e.g. Color Grading, Camera Calibration, Grain, Crop Rect).
 */
export function hydrateDevelopRecipe(raw: Partial<DevelopRecipe> | null | undefined): DevelopRecipe {
  const def = createDefaultDevelopRecipe();
  if (!raw) return def;
  const cg = raw.colorGrading;
  const cal = raw.cameraCalibration;
  const det = raw.detailOptics;
  const crop = raw.cropTransform;
  const rect = crop?.rect;

  return {
    whiteBalance: { ...def.whiteBalance, ...raw.whiteBalance },
    basicTone: { ...def.basicTone, ...raw.basicTone },
    toneCurve: { ...def.toneCurve, ...raw.toneCurve },
    hslMixer: {
      red: { ...ZERO_HSL, ...raw.hslMixer?.red },
      orange: { ...ZERO_HSL, ...raw.hslMixer?.orange },
      yellow: { ...ZERO_HSL, ...raw.hslMixer?.yellow },
      green: { ...ZERO_HSL, ...raw.hslMixer?.green },
      aqua: { ...ZERO_HSL, ...raw.hslMixer?.aqua },
      blue: { ...ZERO_HSL, ...raw.hslMixer?.blue },
      purple: { ...ZERO_HSL, ...raw.hslMixer?.purple },
      magenta: { ...ZERO_HSL, ...raw.hslMixer?.magenta },
    },
    colorGrading: {
      shadows: { ...ZERO_WHEEL, ...cg?.shadows },
      midtones: { ...ZERO_WHEEL, ...cg?.midtones },
      highlights: { ...ZERO_WHEEL, ...cg?.highlights },
      blending: cg?.blending ?? def.colorGrading.blending,
      balance: cg?.balance ?? def.colorGrading.balance,
    },
    cameraCalibration: { ...def.cameraCalibration, ...cal },
    lut3d: raw.lut3d ? { ...raw.lut3d } : null,
    detailOptics: { ...def.detailOptics, ...det },
    cropTransform: {
      ...def.cropTransform,
      ...crop,
      rect: rect
        ? {
            left: clamp(rect.left, 0, 0.95),
            top: clamp(rect.top, 0, 0.95),
            width: clamp(rect.width, 0.05, 1),
            height: clamp(rect.height, 0.05, 1),
          }
        : { ...FULL_FRAME_CROP_RECT },
    },
    masks: hydrateMaskLayers(raw.masks),
    backgroundRemoval: hydrateBackgroundRemoval(raw.backgroundRemoval),
  };
}

export function createColorAndToneSyncMask(): DevelopSyncMask {
  return {
    whiteBalance: true,
    basicTone: true,
    toneCurve: true,
    hslMixer: true,
    colorGrading: true,
    cameraCalibration: true,
    lut3d: true,
    detailOptics: true,
    cropTransform: false,
    masks: false,
  };
}

export function isColorGradingModified(cg: ColorGradingSettings): boolean {
  return (
    cg.shadows.saturation !== 0 ||
    cg.shadows.luminance !== 0 ||
    cg.midtones.saturation !== 0 ||
    cg.midtones.luminance !== 0 ||
    cg.highlights.saturation !== 0 ||
    cg.highlights.luminance !== 0
  );
}

export function isCameraCalibrationModified(cal: CameraCalibrationSettings): boolean {
  return (
    cal.shadowsTint !== 0 ||
    cal.redHue !== 0 ||
    cal.redSaturation !== 0 ||
    cal.greenHue !== 0 ||
    cal.greenSaturation !== 0 ||
    cal.blueHue !== 0 ||
    cal.blueSaturation !== 0
  );
}

export function isCropRectModified(rect: CropRect | undefined): boolean {
  if (!rect) return false;
  return (
    rect.left > 1e-4 ||
    rect.top > 1e-4 ||
    Math.abs(rect.width - 1) > 1e-4 ||
    Math.abs(rect.height - 1) > 1e-4
  );
}

export function getModifiedDevelopModules(recipe: DevelopRecipe): DevelopSyncMask {
  const def = createDefaultDevelopRecipe();
  const bt = recipe.basicTone;
  const d = recipe.detailOptics;
  const c = recipe.cropTransform;

  return {
    whiteBalance:
      recipe.whiteBalance.temperatureKelvin !== def.whiteBalance.temperatureKelvin ||
      recipe.whiteBalance.tint !== def.whiteBalance.tint,
    basicTone:
      bt.blackAndWhite ||
      Math.abs(bt.exposureEv) > 1e-4 ||
      bt.contrast !== 0 ||
      bt.highlights !== 0 ||
      bt.shadows !== 0 ||
      bt.whites !== 0 ||
      bt.blacks !== 0 ||
      bt.texture !== 0 ||
      bt.clarity !== 0 ||
      bt.dehaze !== 0 ||
      bt.vibrance !== 0 ||
      bt.saturation !== 0,
    toneCurve: isToneCurveModified(recipe.toneCurve),
    hslMixer: Object.values(recipe.hslMixer).some(
      (ch) => ch.hue !== 0 || ch.saturation !== 0 || ch.luminance !== 0,
    ),
    colorGrading: isColorGradingModified(recipe.colorGrading),
    cameraCalibration: isCameraCalibrationModified(recipe.cameraCalibration),
    lut3d: recipe.lut3d !== null,
    detailOptics:
      d.sharpeningAmount !== 0 ||
      d.luminanceNoiseReduction !== 0 ||
      d.colorNoiseReduction !== 0 ||
      d.vignetteAmount !== 0 ||
      d.grainAmount !== 0,
    cropTransform:
      isCropRectModified(c.rect) ||
      Math.abs(c.rotationDeg) > 1e-4 ||
      c.aspectRatio !== 'original' ||
      c.flipHorizontal ||
      c.flipVertical ||
      (c.quarterTurns % 4) !== 0 ||
      c.eyeLevelAutoLocked,
    masks:
      (recipe.masks ?? []).some(isMaskLayerEffective) ||
      isBackgroundRemovalActive(recipe.backgroundRemoval),
  };
}

export function isRecipeModified(recipe: DevelopRecipe): boolean {
  const modules = getModifiedDevelopModules(recipe);
  return Object.values(modules).some(Boolean);
}

/**
 * Solves neutral Kelvin + Tint white balance from a sampled neutral gray/white RGB pixel
 * (`0..1` normalized sRGB) clicked with the WB Eyedropper tool (`W`).
 * Matches `WhiteBalance::calibrated_from_neutral_rgb` in Rust `internal/domain/src/develop.rs`.
 */
export function calibrateWhiteBalanceFromSampledRgb(
  r: number,
  g: number,
  b: number,
): WhiteBalanceSettings {
  const safeR = Math.max(1e-3, r);
  const safeG = Math.max(1e-3, g);
  const safeB = Math.max(1e-3, b);

  const rbRatio = clamp(safeB / safeR, 0.5, 2.0);
  const temperatureKelvin = Math.round(clamp(5500 * rbRatio, 2000, 50000) / 50) * 50;
  const rbAvg = 0.5 * (safeR + safeB);
  const tint = Math.round(clamp((safeG / rbAvg - 1) * 180, -150, 150));

  return { temperatureKelvin, tint };
}

/**
 * Computes a Lightroom-style non-destructive Auto Tone balance from an RGB histogram.
 * Centers tonal median cleanly while recovering clipped highlights and lifting crushed shadows.
 */
export function computeAutoToneFromHistogram(
  histogram: {
    readonly red: readonly number[];
    readonly green: readonly number[];
    readonly blue: readonly number[];
  },
  current: BasicToneSettings,
): BasicToneSettings {
  const n = Math.min(histogram.red.length, histogram.green.length, histogram.blue.length);
  if (n < 4) return current;

  const lumaBins: number[] = new Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.max(0, 0.2126 * histogram.red[i] + 0.7152 * histogram.green[i] + 0.0722 * histogram.blue[i]);
    lumaBins[i] = v;
    total += v;
  }
  if (total <= 1e-6) return current;

  const percentile = (p: number): number => {
    const target = total * p;
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc += lumaBins[i];
      if (acc >= target) return i / (n - 1);
    }
    return 1;
  };

  const p02 = percentile(0.02);
  const p50 = percentile(0.5);
  const p98 = percentile(0.98);

  const exposureEv = Number(clamp((0.46 - p50) * 2.1, -1.5, 1.5).toFixed(2));
  const highlights = Math.round(clamp((0.86 - p98) * 160, -45, 25));
  const shadows = Math.round(clamp((0.18 - p02) * 180, -20, 45));
  const spread = p98 - p02;
  const contrast = Math.round(clamp((0.72 - spread) * 75, -15, 25));
  const whites = Math.round(clamp((0.94 - p98) * 110, -25, 20));
  const blacks = Math.round(clamp((0.05 - p02) * 120, -25, 15));

  return {
    ...current,
    exposureEv,
    contrast,
    highlights,
    shadows,
    whites,
    blacks,
  };
}

export function getStudioDevelopPresets(): readonly DevelopPreset[] {
  const base = createDefaultDevelopRecipe();
  return [
    {
      id: 'preset-studio-clean',
      name: 'Studio Portrait Clean',
      category: 'Portrait',
      recipe: {
        ...base,
        whiteBalance: { temperatureKelvin: 5650, tint: 4 },
        basicTone: {
          ...base.basicTone,
          exposureEv: 0.25,
          contrast: 12,
          highlights: -24,
          shadows: 22,
          whites: 8,
          blacks: -6,
          clarity: 6,
          vibrance: 14,
        },
        detailOptics: { ...base.detailOptics, sharpeningAmount: 55, vignetteAmount: -14 },
      },
    },
    {
      id: 'preset-warm-editorial',
      name: 'Warm Golden Editorial',
      category: 'Editorial',
      recipe: {
        ...base,
        whiteBalance: { temperatureKelvin: 6350, tint: 8 },
        basicTone: {
          ...base.basicTone,
          exposureEv: 0.15,
          contrast: 18,
          highlights: -32,
          shadows: 18,
          blacks: -12,
          vibrance: 18,
          saturation: 6,
        },
        toneCurve: { highlights: -8, lights: 12, darks: -10, shadows: 8 },
        colorGrading: {
          ...base.colorGrading,
          shadows: { hueDeg: 215, saturation: 14, luminance: -4 },
          highlights: { hueDeg: 38, saturation: 18, luminance: 4 },
        },
        detailOptics: { ...base.detailOptics, sharpeningAmount: 48, vignetteAmount: -22 },
      },
    },
    {
      id: 'preset-corporate-crisp',
      name: 'Corporate Headshot Crisp',
      category: 'Portrait',
      recipe: {
        ...base,
        whiteBalance: { temperatureKelvin: 5450, tint: 2 },
        basicTone: {
          ...base.basicTone,
          exposureEv: 0.35,
          contrast: 14,
          highlights: -18,
          shadows: 28,
          whites: 12,
          texture: 8,
          clarity: 10,
          vibrance: 10,
        },
        detailOptics: { ...base.detailOptics, sharpeningAmount: 65, vignetteAmount: -10 },
      },
    },
    {
      id: 'preset-film-portra',
      name: 'Portra 400 Studio Film',
      category: 'Editorial',
      recipe: {
        ...base,
        whiteBalance: { temperatureKelvin: 5900, tint: 6 },
        basicTone: {
          ...base.basicTone,
          exposureEv: 0.2,
          contrast: 8,
          highlights: -20,
          shadows: 26,
          vibrance: 12,
        },
        cameraCalibration: {
          ...base.cameraCalibration,
          blueHue: -18,
          blueSaturation: 14,
          redHue: 8,
        },
        detailOptics: {
          ...base.detailOptics,
          vignetteAmount: -16,
          grainAmount: 18,
          grainSize: 28,
        },
      },
    },
    {
      id: 'preset-bw-classic',
      name: 'Silver Darkroom B&W',
      category: 'Monochrome',
      recipe: {
        ...base,
        basicTone: {
          ...base.basicTone,
          blackAndWhite: true,
          exposureEv: 0.15,
          contrast: 34,
          highlights: -15,
          shadows: 14,
          whites: 18,
          blacks: -22,
          clarity: 22,
        },
        hslMixer: {
          ...base.hslMixer,
          orange: { hue: 0, saturation: 0, luminance: 18 },
          blue: { hue: 0, saturation: 0, luminance: -20 },
        },
        detailOptics: {
          ...base.detailOptics,
          sharpeningAmount: 60,
          vignetteAmount: -28,
          grainAmount: 24,
          grainSize: 32,
        },
      },
    },
    {
      id: 'preset-reset',
      name: 'Reset to Original RAW',
      category: 'Utility',
      recipe: base,
    },
  ];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function syncDevelopRecipe(
  target: DevelopRecipe,
  source: DevelopRecipe,
  mask: DevelopSyncMask,
): DevelopRecipe {
  return {
    whiteBalance: mask.whiteBalance ? { ...source.whiteBalance } : target.whiteBalance,
    basicTone: mask.basicTone ? { ...source.basicTone } : target.basicTone,
    toneCurve: mask.toneCurve
      ? {
          ...source.toneCurve,
          points: source.toneCurve.points?.map((p) => ({ ...p })),
          redPoints: source.toneCurve.redPoints?.map((p) => ({ ...p })),
          greenPoints: source.toneCurve.greenPoints?.map((p) => ({ ...p })),
          bluePoints: source.toneCurve.bluePoints?.map((p) => ({ ...p })),
        }
      : target.toneCurve,
    hslMixer: mask.hslMixer ? { ...source.hslMixer } : target.hslMixer,
    colorGrading: mask.colorGrading
      ? {
          ...source.colorGrading,
          shadows: { ...source.colorGrading.shadows },
          midtones: { ...source.colorGrading.midtones },
          highlights: { ...source.colorGrading.highlights },
        }
      : target.colorGrading,
    cameraCalibration: mask.cameraCalibration
      ? { ...source.cameraCalibration }
      : target.cameraCalibration,
    lut3d: mask.lut3d ? (source.lut3d ? { ...source.lut3d } : null) : target.lut3d,
    detailOptics: mask.detailOptics ? { ...source.detailOptics } : target.detailOptics,
    cropTransform: mask.cropTransform
      ? { ...source.cropTransform, rect: { ...source.cropTransform.rect } }
      : target.cropTransform,
    masks: mask.masks
      ? (source.masks ?? []).map((m) => ({
          ...m,
          components: [...m.components],
          adjustments: { ...m.adjustments },
        }))
      : target.masks ?? [],
    backgroundRemoval: mask.masks
      ? hydrateBackgroundRemoval(source.backgroundRemoval)
      : hydrateBackgroundRemoval(target.backgroundRemoval),
  };
}

export function applyRelativeDevelopDelta(
  recipe: DevelopRecipe,
  delta: RelativeDevelopDelta,
): DevelopRecipe {
  const nextTemp = clamp(
    recipe.whiteBalance.temperatureKelvin + (delta.temperatureDeltaKelvin ?? 0),
    2000,
    50000,
  );
  const nextTint = clamp(recipe.whiteBalance.tint + (delta.tintDelta ?? 0), -150, 150);

  return {
    ...recipe,
    whiteBalance: {
      temperatureKelvin: Math.round(nextTemp),
      tint: Math.round(nextTint),
    },
    basicTone: {
      ...recipe.basicTone,
      exposureEv: Number(
        clamp(recipe.basicTone.exposureEv + (delta.exposureDeltaEv ?? 0), -5, 5).toFixed(2),
      ),
      contrast: Math.round(clamp(recipe.basicTone.contrast + (delta.contrastDelta ?? 0), -100, 100)),
      highlights: Math.round(
        clamp(recipe.basicTone.highlights + (delta.highlightsDelta ?? 0), -100, 100),
      ),
      shadows: Math.round(
        clamp(recipe.basicTone.shadows + (delta.shadowsDelta ?? 0), -100, 100),
      ),
      vibrance: Math.round(
        clamp(recipe.basicTone.vibrance + (delta.vibranceDelta ?? 0), -100, 100),
      ),
      saturation: Math.round(
        clamp(recipe.basicTone.saturation + (delta.saturationDelta ?? 0), -100, 100),
      ),
    },
  };
}

/**
 * Computes a real-time CSS filter approximation for non-WebGL fallback rendering.
 */
export function recipeToCssFilter(recipe: DevelopRecipe): string {
  const exposureScale = Math.pow(2, clamp(recipe.basicTone.exposureEv, -3, 3) / 2.2);
  const shadowLift = recipe.basicTone.shadows * 0.0014;
  const highlightPull = recipe.basicTone.highlights * 0.001;
  const curveMid = (recipe.toneCurve.lights + recipe.toneCurve.darks) * 0.001;
  const brightness = clamp(exposureScale + shadowLift + highlightPull + curveMid, 0.2, 2.8);
  const contrast = clamp(
    1 +
      (recipe.basicTone.contrast * 0.85 +
        recipe.basicTone.clarity * 0.25 +
        recipe.basicTone.dehaze * 0.25 +
        (recipe.basicTone.whites - recipe.basicTone.blacks) * 0.22) /
        100,
    0.35,
    2.2,
  );
  const lutBoost = recipe.lut3d ? (recipe.lut3d.intensity / 100) * 0.12 : 0;
  const saturate = recipe.basicTone.blackAndWhite
    ? 0
    : clamp(
        1 +
          (recipe.basicTone.saturation +
            recipe.basicTone.vibrance * 0.55 +
            recipe.hslMixer.orange.saturation * 0.2) /
            100 +
          lutBoost,
        0,
        2.2,
      );
  const warmthShift = (recipe.whiteBalance.temperatureKelvin - 5500) / 10000;
  const sepia = warmthShift > 0 ? clamp(warmthShift * 0.38 + lutBoost * 0.6, 0, 0.42) : lutBoost * 0.4;
  const hueRotateDeg = clamp(
    (warmthShift < 0 ? warmthShift * 28 : 0) +
      recipe.whiteBalance.tint * 0.1 +
      recipe.hslMixer.orange.hue * 0.12,
    -30,
    30,
  );

  return `brightness(${brightness.toFixed(3)}) contrast(${contrast.toFixed(3)}) saturate(${saturate.toFixed(3)}) sepia(${sepia.toFixed(3)}) hue-rotate(${hueRotateDeg.toFixed(1)}deg)`;
}

/**
 * Generates a responsive 48-bin RGB + Luminance preview histogram shaped by the active recipe.
 * Replaced by real pixel histograms from Rust when Tauri IPC is connected.
 */
export function estimatePreviewHistogramBins(recipe: DevelopRecipe): {
  readonly red: readonly number[];
  readonly green: readonly number[];
  readonly blue: readonly number[];
} {
  const bins = 48;
  const expShift = recipe.basicTone.exposureEv * 6.5 + recipe.basicTone.shadows * 0.04;
  const contrastSpread = clamp(1 + recipe.basicTone.contrast / 140, 0.55, 1.75);
  const tempShift = (recipe.whiteBalance.temperatureKelvin - 5500) / 650;
  const tintShift = recipe.whiteBalance.tint / 35;

  const sampleCurve = (channelShift: number): number[] => {
    const out: number[] = [];
    for (let i = 0; i < bins; i++) {
      const normalized = (i - bins * 0.5) / contrastSpread + bins * 0.5 - expShift - channelShift;
      const backdropPeak = Math.exp(-0.5 * ((normalized - 11) / 5.5) ** 2) * 0.78;
      const skinPeak = Math.exp(-0.5 * ((normalized - 29) / 7.2) ** 2) * 0.95;
      const highlightTail = Math.exp(-0.5 * ((normalized - 39) / 4.5) ** 2) * 0.35;
      out.push(clamp(backdropPeak + skinPeak + highlightTail, 0.02, 1));
    }
    return out;
  };

  if (recipe.basicTone.blackAndWhite) {
    const mono = sampleCurve(0);
    return { red: mono, green: mono, blue: mono };
  }

  return {
    red: sampleCurve(tempShift * 1.15),
    green: sampleCurve(-tintShift * 0.9),
    blue: sampleCurve(-tempShift * 1.15),
  };
}
