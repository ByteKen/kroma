import {
  isCameraCalibrationModified,
  isColorGradingModified,
  isRecipeModified,
  type BasicToneSettings,
  type CameraCalibrationSettings,
  type ColorGradingWheel,
  type CropRect,
  type DevelopRecipe,
  type HslColorMixer,
  type ParametricToneCurve,
  type WhiteBalanceSettings,
} from './develop';
import {
  evaluateComposedChannelToneCurve,
  isToneCurveModified,
} from './toneCurve';

export const NEUTRAL_KELVIN = 5500;
export const DEFAULT_VIGNETTE_MIDPOINT = 50;
export const DEFAULT_VIGNETTE_FEATHER = 50;

export interface CompiledShaderUniforms {
  readonly linearStageActive: boolean;
  readonly exposureGain: number;
  readonly wbGains: readonly [number, number, number];
  readonly calActive: boolean;
  readonly calShadowsTint: number;
  /** Column-major 9 floats ready for WebGL `uniformMatrix3fv(loc, false, ...)`. */
  readonly calMatrixColMajor: readonly number[];
  readonly blackAndWhite: boolean;
  readonly shadows: number;
  readonly highlights: number;
  readonly whites: number;
  readonly blacks: number;
  readonly dehaze: number;
  readonly contrast: number;
  readonly vibrance: number;
  readonly saturation: number;
  readonly toneCurveActive: boolean;
  readonly hslActive: boolean;
  /** Packed 8-band HSL shifts (`red..magenta`, 3 floats each: `[hue, saturation, luminance]`). */
  readonly hslShifts: readonly number[];
  readonly colorGradingActive: boolean;
  readonly cgShdTint: readonly [number, number, number];
  readonly cgShdLum: number;
  readonly cgMidTint: readonly [number, number, number];
  readonly cgMidLum: number;
  readonly cgHiTint: readonly [number, number, number];
  readonly cgHiLum: number;
  readonly cgPivot: number;
  readonly cgSpread: number;
  readonly vignetteAmount: number;
  readonly vignetteMidpoint: number;
  readonly vignetteFeather: number;
  readonly spatialStageActive: boolean;
  readonly texture: number;
  readonly clarity: number;
  readonly luminanceNoiseReduction: number;
  readonly colorNoiseReduction: number;
  readonly sharpeningAmount: number;
  readonly grainAmount: number;
  readonly grainSize: number;
}

export interface ObjectFitUvTransform {
  readonly scaleX: number;
  readonly scaleY: number;
  readonly offsetX: number;
  readonly offsetY: number;
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

function clamp01(v: number): number {
  return clamp(v, 0, 1);
}

export function srgbToLinear(c: number): number {
  const x = clamp01(c);
  if (x <= 0.04045) return x / 12.92;
  return Math.pow((x + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(c: number): number {
  const x = clamp01(c);
  if (x <= 0.0031308) return x * 12.92;
  return 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

export function softClip(x: number): number {
  if (x <= 0.92) return Math.max(0, x);
  const excess = x - 0.92;
  return 0.92 + 0.08 * (1 - Math.exp(-excess / 0.08));
}

/** Exact ITU-R BT.709 (Linear sRGB) → ITU-R BT.2020 (Linear Rec.2020, D65) 3×3 matrix. */
export function rec709ToRec2020(rgb: readonly [number, number, number]): [number, number, number] {
  return [
    0.6274039 * rgb[0] + 0.329283 * rgb[1] + 0.0433131 * rgb[2],
    0.0690973 * rgb[0] + 0.9195404 * rgb[1] + 0.0113623 * rgb[2],
    0.0163914 * rgb[0] + 0.0880133 * rgb[1] + 0.8955953 * rgb[2],
  ];
}

/** Exact ITU-R BT.2020 (Linear Rec.2020) → ITU-R BT.709 (Linear sRGB, D65) 3×3 matrix. */
export function rec2020ToRec709(rgb: readonly [number, number, number]): [number, number, number] {
  return [
    1.660491 * rgb[0] - 0.5876411 * rgb[1] - 0.0728499 * rgb[2],
    -0.1245505 * rgb[0] + 1.1328999 * rgb[1] - 0.0083494 * rgb[2],
    -0.0181508 * rgb[0] - 0.1005789 * rgb[1] + 1.1187297 * rgb[2],
  ];
}

/**
 * Hue- and luminance-preserving smooth gamut compression from Linear Rec.2020 to Linear sRGB.
 * Leaves all neutral grays, skin tones, and colors within 85% linear sRGB saturation untouched,
 * while rolling out-of-sRGB-gamut camera colors smoothly onto the displayable boundary.
 */
export function compressRec2020ToLinearSrgb(
  lin2020: readonly [number, number, number],
): [number, number, number] {
  const srgb = rec2020ToRec709(lin2020);
  const y = Math.max(0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2], 0);
  if (y <= 1e-6) {
    return [0, 0, 0];
  }
  const minC = Math.min(srgb[0], srgb[1], srgb[2]);
  const d = (y - minC) / y;
  const threshold = 0.85;
  if (d <= threshold) {
    return [Math.max(0, srgb[0]), Math.max(0, srgb[1]), Math.max(0, srgb[2])];
  }
  const excess = d - threshold;
  const limit = 1 - threshold;
  const dCompressed = threshold + limit * (excess / (excess + limit));
  const scale = dCompressed / d;
  return [
    Math.max(0, y + (srgb[0] - y) * scale),
    Math.max(0, y + (srgb[1] - y) * scale),
    Math.max(0, y + (srgb[2] - y) * scale),
  ];
}

/**
 * Scene-linear HDR highlight recovery matching `recover_hdr_highlights` in Rust `color_math.rs`.
 */
export function recoverHdrHighlights(
  lin: readonly [number, number, number],
  highlights: number,
  whites: number,
): [number, number, number] {
  const peak = Math.max(lin[0], lin[1], lin[2]);
  if (peak <= 0.65) {
    return [lin[0], lin[1], lin[2]];
  }
  const hlPull = highlights < 0 ? -highlights / 100 : 0;
  const whPull = whites < 0 ? -whites / 100 : 0;
  const pull = Math.min(1, hlPull * 0.65 + whPull * 0.35);
  if (pull <= 1e-5) {
    return [lin[0], lin[1], lin[2]];
  }
  const knee = 0.65;
  const excess = peak - knee;
  const weight = smoothstep(0.65, 1.35, peak);
  const targetPeak = knee + excess / (1 + pull * 2.4 * weight);
  const scale = targetPeak / Math.max(peak, 1e-5);
  return [lin[0] * scale, lin[1] * scale, lin[2] * scale];
}

/**
 * Computes normalized RGB linear multipliers for Kelvin temperature & green/magenta tint.
 * Matches `compute_white_balance_gains` in `internal/infrastructure/src/develop/color_math.rs`.
 */
export function computeWhiteBalanceGains(
  wb: WhiteBalanceSettings,
): readonly [number, number, number] {
  const kelvinShift = Math.log(Math.max(1000, wb.temperatureKelvin) / NEUTRAL_KELVIN);
  const warmStrength = clamp(kelvinShift, -1.0, 1.2);
  const rGain = clamp(1.0 + warmStrength * 0.42, 0.45, 2.2);
  const bGain = clamp(1.0 - warmStrength * 0.42, 0.45, 2.2);
  const gGain = clamp(1.0 - wb.tint / 350.0, 0.55, 1.65);

  const luma = 0.2126 * rGain + 0.7152 * gGain + 0.0722 * bGain;
  const inv = 1.0 / Math.max(luma, 1e-4);
  return [rGain * inv, gGain * inv, bGain * inv];
}

/**
 * Computes row-sum-1 3×3 Camera Calibration primary matrix matching Rust `compute_calibration_primary_matrix`.
 * Returns row-major `[[r00,r01,r02], [r10,r11,r12], [r20,r21,r22]]`.
 */
export function computeCalibrationPrimaryMatrix(
  cal: CameraCalibrationSettings,
): readonly [
  readonly [number, number, number],
  readonly [number, number, number],
  readonly [number, number, number],
] {
  const hr = (cal.redHue / 100) * 0.32;
  const hg = (cal.greenHue / 100) * 0.32;
  const hb = (cal.blueHue / 100) * 0.32;

  const sr = (cal.redSaturation / 100) * 0.42;
  const sg = (cal.greenSaturation / 100) * 0.42;
  const sb = (cal.blueSaturation / 100) * 0.42;

  const rows: [number, number, number][] = [
    [1 + sr * 0.44, -hg * 0.4 - sg * 0.22, hb * 0.4 - sb * 0.22],
    [hr * 0.45 - sr * 0.22, 1 + sg * 0.44, -hb * 0.45 - sb * 0.22],
    [-hr * 0.35 - sr * 0.22, hg * 0.4 - sg * 0.22, 1 + sb * 0.44],
  ];

  for (const row of rows) {
    const sum = row[0] + row[1] + row[2];
    const inv = Math.abs(sum) > 1e-4 ? 1 / sum : 1;
    row[0] *= inv;
    row[1] *= inv;
    row[2] *= inv;
  }

  return [rows[0], rows[1], rows[2]];
}

function compileColorWheel(wheel: ColorGradingWheel): {
  readonly tint: readonly [number, number, number];
  readonly lum: number;
} {
  const sat = clamp01(wheel.saturation / 100);
  const pure = hslToRgb(wheel.hueDeg, 1, 0.5);
  const strength = sat * 0.34;
  return {
    tint: [
      (pure[0] - 0.5) * strength,
      (pure[1] - 0.5) * strength,
      (pure[2] - 0.5) * strength,
    ],
    lum: (wheel.luminance / 100) * 0.18,
  };
}

/**
 * Builds a 256×1 RGBA8 LUT texture packing composed Master + per-channel Red/Green/Blue tone
 * curves into R, G, B bytes (`build_rgb_tone_curve_luts` in Rust `color_math.rs`).
 */
export function buildToneCurveLutRgba(curve: ParametricToneCurve): Uint8Array {
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    const r = clamp01(evaluateComposedChannelToneCurve(curve, x, 'red'));
    const g = clamp01(evaluateComposedChannelToneCurve(curve, x, 'green'));
    const b = clamp01(evaluateComposedChannelToneCurve(curve, x, 'blue'));
    const idx = i * 4;
    out[idx] = Math.round(r * 255);
    out[idx + 1] = Math.round(g * 255);
    out[idx + 2] = Math.round(b * 255);
    out[idx + 3] = 255;
  }
  return out;
}

/**
 * Builds a full 32-bit float 256×1 RGBA32F LUT buffer for the unified WGSL WebGPU compute pass
 * with zero 8-bit quantization.
 */
export function buildToneCurveLutRgba32F(curve: ParametricToneCurve): Float32Array {
  const out = new Float32Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    const idx = i * 4;
    out[idx] = clamp01(evaluateComposedChannelToneCurve(curve, x, 'red'));
    out[idx + 1] = clamp01(evaluateComposedChannelToneCurve(curve, x, 'green'));
    out[idx + 2] = clamp01(evaluateComposedChannelToneCurve(curve, x, 'blue'));
    out[idx + 3] = 1.0;
  }
  return out;
}

const HSL_BAND_KEYS = [
  'red',
  'orange',
  'yellow',
  'green',
  'aqua',
  'blue',
  'purple',
  'magenta',
] as const;

export function compileDevelopShaderUniforms(recipe: DevelopRecipe): CompiledShaderUniforms {
  const ev = clamp(recipe.basicTone.exposureEv, -5, 5);
  const linearStageActive =
    Math.abs(ev) > 1e-4 ||
    recipe.whiteBalance.temperatureKelvin !== NEUTRAL_KELVIN ||
    recipe.whiteBalance.tint !== 0;

  const calActive = isCameraCalibrationModified(recipe.cameraCalibration);
  const calMat = computeCalibrationPrimaryMatrix(recipe.cameraCalibration);
  // Column-major order for GLSL `mat3`: [c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2]
  const calMatrixColMajor = [
    calMat[0][0],
    calMat[1][0],
    calMat[2][0],
    calMat[0][1],
    calMat[1][1],
    calMat[2][1],
    calMat[0][2],
    calMat[1][2],
    calMat[2][2],
  ];

  const hslShifts: number[] = [];
  let hslActive = false;
  for (const key of HSL_BAND_KEYS) {
    const ch = recipe.hslMixer[key];
    if (ch.hue !== 0 || ch.saturation !== 0 || ch.luminance !== 0) {
      hslActive = true;
    }
    hslShifts.push(ch.hue, ch.saturation, ch.luminance);
  }

  const cg = recipe.colorGrading;
  const colorGradingActive = isColorGradingModified(cg);
  const shd = compileColorWheel(cg.shadows);
  const mid = compileColorWheel(cg.midtones);
  const hi = compileColorWheel(cg.highlights);
  const balanceShift = (cg.balance / 100) * 0.25;
  const cgPivot = clamp(0.5 - balanceShift, 0.18, 0.82);
  const cgSpread = 0.14 + clamp01(cg.blending / 100) * 0.36;

  const texture = clamp(recipe.basicTone.texture, -100, 100);
  const clarity = clamp(recipe.basicTone.clarity, -100, 100);
  const luminanceNoiseReduction = clamp(recipe.detailOptics.luminanceNoiseReduction, 0, 100);
  const colorNoiseReduction = clamp(recipe.detailOptics.colorNoiseReduction, 0, 100);
  const sharpeningAmount = clamp(recipe.detailOptics.sharpeningAmount, 0, 150);
  const grainAmount = clamp(recipe.detailOptics.grainAmount, 0, 100);
  const grainSize = clamp(recipe.detailOptics.grainSize, 0, 100);
  const spatialStageActive =
    Math.abs(texture) > 1e-4 ||
    Math.abs(clarity) > 1e-4 ||
    luminanceNoiseReduction > 1e-4 ||
    colorNoiseReduction > 1e-4 ||
    sharpeningAmount > 1e-4 ||
    grainAmount > 1e-4;

  return {
    linearStageActive,
    exposureGain: Math.pow(2, ev),
    wbGains: computeWhiteBalanceGains(recipe.whiteBalance),
    calActive,
    calShadowsTint: recipe.cameraCalibration.shadowsTint / 100,
    calMatrixColMajor,
    blackAndWhite: recipe.basicTone.blackAndWhite,
    shadows: recipe.basicTone.shadows,
    highlights: recipe.basicTone.highlights,
    whites: recipe.basicTone.whites,
    blacks: recipe.basicTone.blacks,
    dehaze: recipe.basicTone.dehaze,
    contrast: recipe.basicTone.contrast,
    vibrance: recipe.basicTone.vibrance,
    saturation: recipe.basicTone.saturation,
    toneCurveActive: isToneCurveModified(recipe.toneCurve),
    hslActive,
    hslShifts,
    colorGradingActive,
    cgShdTint: shd.tint,
    cgShdLum: shd.lum,
    cgMidTint: mid.tint,
    cgMidLum: mid.lum,
    cgHiTint: hi.tint,
    cgHiLum: hi.lum,
    cgPivot,
    cgSpread,
    vignetteAmount: recipe.detailOptics.vignetteAmount,
    vignetteMidpoint: recipe.detailOptics.vignetteMidpoint ?? DEFAULT_VIGNETTE_MIDPOINT,
    vignetteFeather: recipe.detailOptics.vignetteFeather ?? DEFAULT_VIGNETTE_FEATHER,
    spatialStageActive,
    texture,
    clarity,
    luminanceNoiseReduction,
    colorNoiseReduction,
    sharpeningAmount,
    grainAmount,
    grainSize,
  };
}

/**
 * Computes UV scale/offset so a WebGL quad matches CSS `object-cover` or `object-contain`
 * (optionally narrowed to a normalized `cropRect`) to the exact sub-pixel.
 */
export function computeObjectFitUvTransform(
  imgWidth: number,
  imgHeight: number,
  boxWidth: number,
  boxHeight: number,
  fit: 'cover' | 'contain',
  cropRect?: CropRect,
): ObjectFitUvTransform {
  if (imgWidth <= 0 || imgHeight <= 0 || boxWidth <= 0 || boxHeight <= 0) {
    return { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 };
  }
  const rectLeft = cropRect ? clamp(cropRect.left, 0, 0.95) : 0;
  const rectTop = cropRect ? clamp(cropRect.top, 0, 0.95) : 0;
  const rectW = cropRect ? clamp(cropRect.width, 0.05, 1 - rectLeft) : 1;
  const rectH = cropRect ? clamp(cropRect.height, 0.05, 1 - rectTop) : 1;

  const effectiveW = imgWidth * rectW;
  const effectiveH = imgHeight * rectH;
  const imgAspect = effectiveW / effectiveH;
  const boxAspect = boxWidth / boxHeight;

  let baseScaleX = 1;
  let baseScaleY = 1;
  let baseOffsetX = 0;
  let baseOffsetY = 0;

  if (fit === 'cover') {
    if (imgAspect > boxAspect) {
      baseScaleX = boxAspect / imgAspect;
      baseOffsetX = (1 - baseScaleX) * 0.5;
    } else {
      baseScaleY = imgAspect / boxAspect;
      baseOffsetY = (1 - baseScaleY) * 0.5;
    }
  } else if (imgAspect > boxAspect) {
    baseScaleY = imgAspect / boxAspect;
    baseOffsetY = (1 - baseScaleY) * 0.5;
  } else {
    baseScaleX = boxAspect / imgAspect;
    baseOffsetX = (1 - baseScaleX) * 0.5;
  }

  return {
    scaleX: baseScaleX * rectW,
    scaleY: baseScaleY * rectH,
    offsetX: rectLeft + baseOffsetX * rectW,
    offsetY: rectTop + baseOffsetY * rectH,
  };
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / Math.max(edge1 - edge0, 1e-6));
  return t * t * (3 - 2 * t);
}

function gaussian(x: number, mean: number, sigma: number): number {
  const d = (x - mean) / Math.max(sigma, 1e-5);
  return Math.exp(-0.5 * d * d);
}

function hueDistanceDeg(a: number, b: number): number {
  const diff = Math.abs(a - b) % 360;
  return diff > 180 ? 360 - diff : diff;
}

export function rgbToHsl(rgb: readonly [number, number, number]): [number, number, number] {
  const r = clamp01(rgb[0]);
  const g = clamp01(rgb[1]);
  const b = clamp01(rgb[2]);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = 0.5 * (max + min);
  const d = max - min;
  if (d <= 1e-6) return [0, 0, l];
  const s = l > 0.5 ? d / Math.max(2 - max - min, 1e-6) : d / Math.max(max + min, 1e-6);
  let h =
    Math.abs(max - r) <= 1e-6
      ? (g - b) / d + (g < b ? 6 : 0)
      : Math.abs(max - g) <= 1e-6
        ? (b - r) / d + 2
        : (r - g) / d + 4;
  h = ((h * 60) % 360 + 360) % 360;
  return [h, clamp01(s), clamp01(l)];
}

export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s <= 1e-6) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hk = (((h % 360) + 360) % 360) / 360;
  const hueToRgb = (tRaw: number): number => {
    let t = tRaw;
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 0.5) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [hueToRgb(hk + 1 / 3), hueToRgb(hk), hueToRgb(hk - 1 / 3)];
}

function applyBasicTone(
  rgbIn: [number, number, number],
  tone: BasicToneSettings,
): [number, number, number] {
  let rgb: [number, number, number] = [...rgbIn];
  const luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  let delta = 0;

  if (tone.shadows !== 0) {
    const w = 1 - smoothstep(0.05, 0.55, luma);
    delta += (tone.shadows / 100) * 0.28 * w * w;
  }
  if (tone.highlights !== 0) {
    const w = smoothstep(0.45, 0.95, luma);
    delta += (tone.highlights / 100) * 0.28 * w * w;
  }
  if (tone.whites !== 0) {
    delta += (tone.whites / 100) * 0.22 * smoothstep(0.65, 1.0, luma);
  }
  if (tone.blacks !== 0) {
    delta += (tone.blacks / 100) * 0.22 * (1 - smoothstep(0.0, 0.35, luma));
  }

  if (Math.abs(delta) > 1e-6) {
    const targetLuma = clamp01(luma + delta);
    if (luma > 1e-4) {
      const ratio = targetLuma / luma;
      rgb = [clamp01(rgb[0] * ratio), clamp01(rgb[1] * ratio), clamp01(rgb[2] * ratio)];
    } else {
      rgb = [targetLuma, targetLuma, targetLuma];
    }
  }

  if (tone.dehaze !== 0) {
    const dh = tone.dehaze / 100;
    const lift = -dh * 0.06;
    const scale = 1 + dh * 0.18;
    rgb = [
      clamp01((rgb[0] + lift) * scale),
      clamp01((rgb[1] + lift) * scale),
      clamp01((rgb[2] + lift) * scale),
    ];
  }

  if (tone.contrast !== 0) {
    const c = tone.contrast / 100;
    const factor = c >= 0 ? 1 + c * 0.85 : 1 + c * 0.65;
    rgb = [
      clamp01((rgb[0] - 0.5) * factor + 0.5),
      clamp01((rgb[1] - 0.5) * factor + 0.5),
      clamp01((rgb[2] - 0.5) * factor + 0.5),
    ];
  }

  return rgb;
}

const HSL_BAND_SPECS: readonly (readonly [number, number])[] = [
  [0, 30],
  [30, 30],
  [60, 35],
  [120, 50],
  [180, 50],
  [240, 45],
  [285, 35],
  [325, 35],
];

function applyHslMixer(
  rgb: [number, number, number],
  mixer: HslColorMixer,
): [number, number, number] {
  const [h, s, l] = rgbToHsl(rgb);
  if (s < 0.015) return rgb;
  let hueDelta = 0;
  let satDelta = 0;
  let lumDelta = 0;
  for (let i = 0; i < HSL_BAND_KEYS.length; i++) {
    const shift = mixer[HSL_BAND_KEYS[i]];
    if (shift.hue === 0 && shift.saturation === 0 && shift.luminance === 0) continue;
    const [center, halfWidth] = HSL_BAND_SPECS[i];
    const dist = hueDistanceDeg(h, center);
    if (dist < halfWidth) {
      const weight = 0.5 * (1 + Math.cos((Math.PI * dist) / halfWidth));
      hueDelta += (shift.hue / 100) * 28 * weight;
      satDelta += (shift.saturation / 100) * weight;
      lumDelta += (shift.luminance / 100) * 0.25 * weight;
    }
  }
  const nextH = (((h + hueDelta) % 360) + 360) % 360;
  const nextS = clamp01(s * (1 + satDelta));
  const nextL = clamp01(l + lumDelta * s);
  return hslToRgb(nextH, nextS, nextL);
}

function applyBlackAndWhiteMix(
  rgb: [number, number, number],
  mixer: HslColorMixer,
  hslActive: boolean,
): [number, number, number] {
  const baseLuma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  if (!hslActive) return [baseLuma, baseLuma, baseLuma];
  const [h, s] = rgbToHsl(rgb);
  if (s < 0.015) return [baseLuma, baseLuma, baseLuma];
  let lumDelta = 0;
  for (let i = 0; i < HSL_BAND_KEYS.length; i++) {
    const shift = mixer[HSL_BAND_KEYS[i]];
    if (shift.luminance === 0) continue;
    const [center, halfWidth] = HSL_BAND_SPECS[i];
    const dist = hueDistanceDeg(h, center);
    if (dist < halfWidth) {
      const weight = 0.5 * (1 + Math.cos((Math.PI * dist) / halfWidth));
      lumDelta += (shift.luminance / 100) * 0.45 * weight;
    }
  }
  const bw = clamp01(baseLuma + lumDelta * s);
  return [bw, bw, bw];
}

/**
 * Pure TypeScript reference implementation of `CompiledDevelopGrade::grade_pixel_in_space` from Rust
 * (`internal/infrastructure/src/develop/color_math.rs`) for automated cross-engine parity tests.
 */
export function evaluateColorGradePixel(
  pixel: readonly [number, number, number],
  recipe: DevelopRecipe,
  nx = 0,
  ny = 0,
  inputIsLinear = false,
  inputIsWideGamut = false,
): [number, number, number] {
  const u = compileDevelopShaderUniforms(recipe);

  let rgb: [number, number, number];
  if (inputIsLinear) {
    let lin: [number, number, number] = u.linearStageActive
      ? [
          Math.max(0, pixel[0] * u.wbGains[0] * u.exposureGain),
          Math.max(0, pixel[1] * u.wbGains[1] * u.exposureGain),
          Math.max(0, pixel[2] * u.wbGains[2] * u.exposureGain),
        ]
      : [Math.max(0, pixel[0]), Math.max(0, pixel[1]), Math.max(0, pixel[2])];

    if (u.highlights < 0 || u.whites < 0) {
      lin = recoverHdrHighlights(lin, u.highlights, u.whites);
    }
    if (inputIsWideGamut) {
      lin = compressRec2020ToLinearSrgb(lin);
    }
    rgb = [
      linearToSrgb(softClip(lin[0])),
      linearToSrgb(softClip(lin[1])),
      linearToSrgb(softClip(lin[2])),
    ];
  } else {
    rgb = u.linearStageActive
      ? [
          linearToSrgb(softClip(Math.max(0, srgbToLinear(pixel[0]) * u.wbGains[0] * u.exposureGain))),
          linearToSrgb(softClip(Math.max(0, srgbToLinear(pixel[1]) * u.wbGains[1] * u.exposureGain))),
          linearToSrgb(softClip(Math.max(0, srgbToLinear(pixel[2]) * u.wbGains[2] * u.exposureGain))),
        ]
      : [clamp01(pixel[0]), clamp01(pixel[1]), clamp01(pixel[2])];
  }

  if (u.calActive) {
    const m = computeCalibrationPrimaryMatrix(recipe.cameraCalibration);
    rgb = [
      clamp01(m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2]),
      clamp01(m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2]),
      clamp01(m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2]),
    ];
    if (Math.abs(u.calShadowsTint) > 1e-5) {
      const luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
      const sw = Math.pow(1 - smoothstep(0, 0.55, luma), 2);
      const delta = u.calShadowsTint * 0.08 * sw;
      rgb = [
        clamp01(rgb[0] + delta * 0.5),
        clamp01(rgb[1] - delta),
        clamp01(rgb[2] + delta * 0.5),
      ];
    }
  }

  rgb = applyBasicTone(rgb, recipe.basicTone);

  if (u.toneCurveActive) {
    rgb = [
      evaluateComposedChannelToneCurve(recipe.toneCurve, rgb[0], 'red'),
      evaluateComposedChannelToneCurve(recipe.toneCurve, rgb[1], 'green'),
      evaluateComposedChannelToneCurve(recipe.toneCurve, rgb[2], 'blue'),
    ];
  }

  if (u.blackAndWhite) {
    rgb = applyBlackAndWhiteMix(rgb, recipe.hslMixer, u.hslActive);
  } else {
    if (u.vibrance !== 0 || u.saturation !== 0) {
      const [h, s0, l] = rgbToHsl(rgb);
      let s = s0;
      if (u.vibrance !== 0) {
        const vib = u.vibrance / 100;
        const skinProx = gaussian(hueDistanceDeg(h, 28), 0, 18);
        const skinGuard = vib > 0 ? 1 - 0.6 * skinProx : 1;
        s = clamp01(s + vib * 0.55 * (1 - s) * skinGuard);
      }
      if (u.saturation !== 0) {
        s = clamp01(s * (1 + u.saturation / 100));
      }
      rgb = hslToRgb(h, s, l);
    }

    if (u.hslActive) {
      rgb = applyHslMixer(rgb, recipe.hslMixer);
    }
  }

  if (u.colorGradingActive) {
    const luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    let shdW = Math.pow(1 - smoothstep(u.cgPivot - u.cgSpread, u.cgPivot + u.cgSpread * 0.65, luma), 1.5);
    let hiW = Math.pow(smoothstep(u.cgPivot - u.cgSpread * 0.65, u.cgPivot + u.cgSpread, luma), 1.5);
    let midW = Math.max(0, 1 - shdW - hiW);
    const sum = Math.max(1, shdW + midW + hiW);
    shdW /= sum;
    midW /= sum;
    hiW /= sum;

    const shdEnv = 1 - smoothstep(0.75, 1.0, luma);
    const midEnv = smoothstep(0, 0.12, luma) * (1 - smoothstep(0.88, 1.0, luma));
    const hiEnv = smoothstep(0.05, 0.3, luma);
    const lumShift = shdW * u.cgShdLum + midW * u.cgMidLum + hiW * u.cgHiLum;

    rgb = [
      clamp01(rgb[0] + u.cgShdTint[0] * shdW * shdEnv + u.cgMidTint[0] * midW * midEnv + u.cgHiTint[0] * hiW * hiEnv + lumShift),
      clamp01(rgb[1] + u.cgShdTint[1] * shdW * shdEnv + u.cgMidTint[1] * midW * midEnv + u.cgHiTint[1] * hiW * hiEnv + lumShift),
      clamp01(rgb[2] + u.cgShdTint[2] * shdW * shdEnv + u.cgMidTint[2] * midW * midEnv + u.cgHiTint[2] * hiW * hiEnv + lumShift),
    ];
  }

  if (u.vignetteAmount !== 0) {
    const radius = Math.hypot(nx, ny);
    const mid = clamp(u.vignetteMidpoint / 100, 0.05, 0.98) * 1.15;
    const f = clamp(u.vignetteFeather / 100, 0.02, 1.0);
    const inner = mid * (1 - 0.65 * f);
    const outer = Math.min(1.65, mid + 0.12 + 0.72 * f);
    const mask = smoothstep(inner, outer, radius);
    const factor = clamp(1 + (u.vignetteAmount / 100) * 0.75 * mask, 0.15, 1.85);
    rgb = [clamp01(rgb[0] * factor), clamp01(rgb[1] * factor), clamp01(rgb[2] * factor)];
  }

  return [clamp01(rgb[0]), clamp01(rgb[1]), clamp01(rgb[2])];
}

export interface DevelopSvgFilterTables {
  readonly rTable: string;
  readonly gTable: string;
  readonly bTable: string;
  readonly saturate: number;
  readonly vignetteOverlay: string | null;
}

/**
 * Precomputes per-channel sRGB SVG `<feComponentTransfer>` LUT tables + saturation matrix +
 * vignette overlay matching Rust `color_math.rs` (including per-channel RGB curves, B&W treatment,
 * and 3-way color grading tints) for fast GPU-composited grid & filmstrip thumbnails.
 */
export function buildDevelopSvgFilterTables(
  recipe: DevelopRecipe,
  steps = 33,
): DevelopSvgFilterTables | null {
  if (!isRecipeModified(recipe)) {
    return null;
  }

  const u = compileDevelopShaderUniforms(recipe);
  const count = Math.max(2, steps);
  const rVals: string[] = new Array(count);
  const gVals: string[] = new Array(count);
  const bVals: string[] = new Array(count);
  const tone = recipe.basicTone;
  const channelNames: readonly ('red' | 'green' | 'blue')[] = ['red', 'green', 'blue'];

  const evalChannel = (x: number, channelIdx: 0 | 1 | 2): number => {
    let y = u.linearStageActive
      ? linearToSrgb(
          softClip(Math.max(0, srgbToLinear(x) * u.wbGains[channelIdx] * u.exposureGain)),
        )
      : x;

    let delta = 0;
    if (tone.shadows !== 0) {
      const w = 1 - smoothstep(0.05, 0.55, y);
      delta += (tone.shadows / 100) * 0.28 * w * w;
    }
    if (tone.highlights !== 0) {
      const w = smoothstep(0.45, 0.95, y);
      delta += (tone.highlights / 100) * 0.28 * w * w;
    }
    if (tone.whites !== 0) {
      delta += (tone.whites / 100) * 0.22 * smoothstep(0.65, 1.0, y);
    }
    if (tone.blacks !== 0) {
      delta += (tone.blacks / 100) * 0.22 * (1 - smoothstep(0.0, 0.35, y));
    }
    if (Math.abs(delta) > 1e-6) {
      y = clamp01(y + delta);
    }

    if (tone.dehaze !== 0) {
      const dh = tone.dehaze / 100;
      y = clamp01((y - dh * 0.06) * (1 + dh * 0.18));
    }

    const effectiveContrast = tone.contrast + tone.clarity * 0.2 + tone.texture * 0.08;
    if (Math.abs(effectiveContrast) > 1e-4) {
      const c = clamp(effectiveContrast, -120, 120) / 100;
      const factor = c >= 0 ? 1 + c * 0.85 : 1 + c * 0.65;
      y = clamp01((y - 0.5) * factor + 0.5);
    }

    if (u.toneCurveActive) {
      y = evaluateComposedChannelToneCurve(recipe.toneCurve, y, channelNames[channelIdx]);
    }

    if (u.colorGradingActive) {
      let shdW = Math.pow(1 - smoothstep(u.cgPivot - u.cgSpread, u.cgPivot + u.cgSpread * 0.65, y), 1.5);
      let hiW = Math.pow(smoothstep(u.cgPivot - u.cgSpread * 0.65, u.cgPivot + u.cgSpread, y), 1.5);
      let midW = Math.max(0, 1 - shdW - hiW);
      const sum = Math.max(1, shdW + midW + hiW);
      shdW /= sum;
      midW /= sum;
      hiW /= sum;
      const shdEnv = 1 - smoothstep(0.75, 1.0, y);
      const midEnv = smoothstep(0, 0.12, y) * (1 - smoothstep(0.88, 1.0, y));
      const hiEnv = smoothstep(0.05, 0.3, y);
      const lumShift = shdW * u.cgShdLum + midW * u.cgMidLum + hiW * u.cgHiLum;
      y = clamp01(
        y +
          u.cgShdTint[channelIdx] * shdW * shdEnv +
          u.cgMidTint[channelIdx] * midW * midEnv +
          u.cgHiTint[channelIdx] * hiW * hiEnv +
          lumShift,
      );
    }

    return clamp01(y);
  };

  for (let i = 0; i < count; i++) {
    const x = i / (count - 1);
    rVals[i] = evalChannel(x, 0).toFixed(4);
    gVals[i] = evalChannel(x, 1).toFixed(4);
    bVals[i] = evalChannel(x, 2).toFixed(4);
  }

  const saturate = u.blackAndWhite
    ? 0
    : clamp(1 + (tone.saturation + tone.vibrance * 0.45) / 100, 0, 2.2);
  const vig = u.vignetteAmount;
  const vigStrength = Math.min(0.8, Math.abs(vig) / 115);
  const innerPct = Math.round(clamp(u.vignetteMidpoint * (1 - 0.35 * (u.vignetteFeather / 100)), 18, 65));
  const vignetteOverlay =
    vig < 0
      ? `radial-gradient(circle, transparent ${innerPct}%, rgba(0,0,0,${vigStrength.toFixed(2)}) 100%)`
      : vig > 0
        ? `radial-gradient(circle, transparent ${innerPct}%, rgba(255,255,255,${(vigStrength * 0.65).toFixed(2)}) 100%)`
        : null;

  return {
    rTable: rVals.join(' '),
    gTable: gVals.join(' '),
    bTable: bVals.join(' '),
    saturate,
    vignetteOverlay,
  };
}
