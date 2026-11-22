import {
  type CropRect,
  type DevelopRecipe,
} from './develop';
import {
  compileDevelopShaderUniforms,
  computeCalibrationPrimaryMatrix,
  type ObjectFitUvTransform,
} from './developShaderMath';
import {
  isBackgroundRemovalActive,
  type CompiledWebGlMaskData,
  type MaskLayer,
} from './masking';

/** Strict 16-byte aligned uniform sizes matching `shaders/develop/*.wgsl` and `wgpu_engine.rs`. */
export const GRADE_UNIFORM_FLOATS = 29 * 4; // 464 bytes
export const MASK_UNIFORM_FLOATS = 17 * 4; // 272 bytes
export const SPATIAL_UNIFORM_FLOATS = 5 * 4; // 80 bytes
export const PRESENT_UNIFORM_FLOATS = 2 * 4; // 32 bytes

export function detectWebGpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
}

export interface PackGradeUniformsInput {
  readonly outWidth: number;
  readonly outHeight: number;
  readonly srcWidth: number;
  readonly srcHeight: number;
  readonly cropRect?: CropRect;
  readonly inputIsLinear: boolean;
  readonly inputIsWideGamut: boolean;
  readonly recipe: DevelopRecipe;
  readonly includeFineRotation?: boolean;
  readonly bgMatteReady: boolean;
}

/**
 * Packs `GradeUniforms` (29 × `vec4<f32>` = 464 bytes) matching `shaders/develop/grade_pass.wgsl`
 * and `internal/infrastructure/src/develop/wgpu_engine.rs::GradeUniforms` byte-for-byte.
 */
export function packGradeUniforms(input: PackGradeUniformsInput): Float32Array {
  const out = new Float32Array(GRADE_UNIFORM_FLOATS);
  const { recipe } = input;
  const u = compileDevelopShaderUniforms(recipe);
  const calMat = computeCalibrationPrimaryMatrix(recipe.cameraCalibration);
  const crop = recipe.cropTransform;
  const qt = (((crop.quarterTurns ?? 0) % 4) + 4) % 4;
  const rotRad =
    input.includeFineRotation && Math.abs(crop.rotationDeg) > 1e-3
      ? (crop.rotationDeg * Math.PI) / 180
      : 0;

  const cr = input.cropRect ?? { left: 0, top: 0, width: 1, height: 1 };

  const bg = recipe.backgroundRemoval;
  const bgActive = Boolean(input.bgMatteReady && bg && isBackgroundRemovalActive(bg));
  const bgOpaqueMode =
    bgActive && bg
      ? bg.mode === 'solid_color'
        ? 2.0
        : bg.mode === 'gradient'
          ? 3.0
          : 0.0
      : 0.0;

  let i = 0;
  // [0] dims: (outW, outH, srcW, srcH)
  out[i++] = input.outWidth;
  out[i++] = input.outHeight;
  out[i++] = input.srcWidth;
  out[i++] = input.srcHeight;

  // [1] cropRect: (left, top, width, height)
  out[i++] = cr.left;
  out[i++] = cr.top;
  out[i++] = cr.width;
  out[i++] = cr.height;

  // [2] orient: (flipH, flipV, quarterTurns, rotationRad)
  out[i++] = crop.flipHorizontal ? 1.0 : 0.0;
  out[i++] = crop.flipVertical ? 1.0 : 0.0;
  out[i++] = qt;
  out[i++] = rotRad;

  // [3] linearFlags: (inputLinear, inputWideGamut, linearActive, exposureGain)
  out[i++] = input.inputIsLinear ? 1.0 : 0.0;
  out[i++] = input.inputIsWideGamut ? 1.0 : 0.0;
  out[i++] = u.linearStageActive ? 1.0 : 0.0;
  out[i++] = u.exposureGain;

  // [4] wbAndBw: (wbR, wbG, wbB, blackAndWhite)
  out[i++] = u.wbGains[0];
  out[i++] = u.wbGains[1];
  out[i++] = u.wbGains[2];
  out[i++] = u.blackAndWhite ? 1.0 : 0.0;

  // [5] stageFlags: (calActive, calShadowsTint, toneCurveActive, hslActive)
  out[i++] = u.calActive ? 1.0 : 0.0;
  out[i++] = u.calShadowsTint;
  out[i++] = u.toneCurveActive ? 1.0 : 0.0;
  out[i++] = u.hslActive ? 1.0 : 0.0;

  // [6..8] calRow0..2 (row-major!)
  out[i++] = calMat[0][0];
  out[i++] = calMat[0][1];
  out[i++] = calMat[0][2];
  out[i++] = 0.0;

  out[i++] = calMat[1][0];
  out[i++] = calMat[1][1];
  out[i++] = calMat[1][2];
  out[i++] = 0.0;

  out[i++] = calMat[2][0];
  out[i++] = calMat[2][1];
  out[i++] = calMat[2][2];
  out[i++] = 0.0;

  // [9] basicToneA: (shadows, highlights, whites, blacks)
  out[i++] = u.shadows;
  out[i++] = u.highlights;
  out[i++] = u.whites;
  out[i++] = u.blacks;

  // [10] basicToneB: (dehaze, contrast, vibrance, saturation)
  out[i++] = u.dehaze;
  out[i++] = u.contrast;
  out[i++] = u.vibrance;
  out[i++] = u.saturation;

  // [11..18] hsl[8]: (hue, saturation, luminance, 0)
  for (let b = 0; b < 8; b++) {
    out[i++] = u.hslShifts[b * 3] ?? 0;
    out[i++] = u.hslShifts[b * 3 + 1] ?? 0;
    out[i++] = u.hslShifts[b * 3 + 2] ?? 0;
    out[i++] = 0.0;
  }

  // [19] cgShd: (tintR, tintG, tintB, lum)
  out[i++] = u.cgShdTint[0];
  out[i++] = u.cgShdTint[1];
  out[i++] = u.cgShdTint[2];
  out[i++] = u.cgShdLum;

  // [20] cgMid
  out[i++] = u.cgMidTint[0];
  out[i++] = u.cgMidTint[1];
  out[i++] = u.cgMidTint[2];
  out[i++] = u.cgMidLum;

  // [21] cgHi
  out[i++] = u.cgHiTint[0];
  out[i++] = u.cgHiTint[1];
  out[i++] = u.cgHiTint[2];
  out[i++] = u.cgHiLum;

  // [22] cgMeta: (cgActive, cgPivot, cgSpread, 0)
  out[i++] = u.colorGradingActive ? 1.0 : 0.0;
  out[i++] = u.cgPivot;
  out[i++] = u.cgSpread;
  out[i++] = 0.0;

  // [23] vignette: (amount, midpoint, feather, 0)
  out[i++] = u.vignetteAmount;
  out[i++] = u.vignetteMidpoint;
  out[i++] = u.vignetteFeather;
  out[i++] = 0.0;

  // [24] lutMeta: (lutSize, lutIntensity, domainMinR, domainMinG)
  out[i++] = 0.0;
  out[i++] = 0.0;
  out[i++] = 0.0;
  out[i++] = 0.0;

  // [25] lutDomain: (domainMinB, domainMaxR, domainMaxG, domainMaxB)
  out[i++] = 0.0;
  out[i++] = 1.0;
  out[i++] = 1.0;
  out[i++] = 1.0;

  // [26] bgColor1: (r, g, b, bgMode)
  out[i++] = (bg?.colorRgb[0] ?? 255) / 255;
  out[i++] = (bg?.colorRgb[1] ?? 255) / 255;
  out[i++] = (bg?.colorRgb[2] ?? 255) / 255;
  out[i++] = bgOpaqueMode;

  // [27] bgColor2: (r, g, b, gradStyle)
  out[i++] = (bg?.color2Rgb?.[0] ?? 20) / 255;
  out[i++] = (bg?.color2Rgb?.[1] ?? 22) / 255;
  out[i++] = (bg?.color2Rgb?.[2] ?? 27) / 255;
  out[i++] = bg?.gradientStyle === 'linear' ? 1.0 : 0.0;

  // [28] bgParams: (gradAngleDeg, feather, 0, 0)
  out[i++] = bg?.gradientAngleDeg ?? 180;
  out[i++] = bg?.feather ?? 25;
  out[i++] = 0.0;
  out[i++] = 0.0;

  return out;
}

export interface PackMaskUniformsInput {
  readonly outWidth: number;
  readonly outHeight: number;
  readonly layer: MaskLayer;
  readonly compiled: CompiledWebGlMaskData;
  readonly overlayOnly: boolean;
  readonly passMode?: 0 | 1 | 2;
}

/**
 * Packs `MaskUniforms` (17 × `vec4<f32>` = 272 bytes) matching `shaders/develop/mask_pass.wgsl`
 * and `internal/infrastructure/src/develop/wgpu_engine.rs::MaskUniforms` byte-for-byte.
 */
export function packMaskUniforms(input: PackMaskUniformsInput): Float32Array {
  const out = new Float32Array(MASK_UNIFORM_FLOATS);
  const { layer, compiled } = input;
  const spatialScale = Math.max(0.5, Math.min(4.0, Math.max(input.outWidth, input.outHeight) / 1024.0));

  let i = 0;
  // [0] dims: (outW, outH, spatialScale, passMode)
  out[i++] = input.outWidth;
  out[i++] = input.outHeight;
  out[i++] = spatialScale;
  out[i++] = input.passMode ?? 0;

  // [1] cropRect: (0, 0, 1, 1) — Studio preview runs masks in full-frame oriented space
  out[i++] = 0.0;
  out[i++] = 0.0;
  out[i++] = 1.0;
  out[i++] = 1.0;

  // [2] layerFlags: (hasSpatial, layerInvert, layerOpacity, overlayOnly)
  out[i++] = compiled.hasSpatial ? 1.0 : 0.0;
  out[i++] = layer.inverted ? 1.0 : 0.0;
  out[i++] = Math.max(0, Math.min(1, layer.opacity / 100));
  out[i++] = input.overlayOnly ? 1.0 : 0.0;

  // [3..6] rangeMeta[4]
  for (let r = 0; r < 4; r++) {
    const slot = compiled.ranges[r];
    out[i++] = slot ? slot.mode : 0.0;
    out[i++] = slot ? slot.op : 0.0;
    out[i++] = slot ? slot.inverted : 0.0;
    out[i++] = 0.0;
  }

  // [7..10] rangeParamsA[4]
  for (let r = 0; r < 4; r++) {
    const slot = compiled.ranges[r];
    out[i++] = slot ? slot.p0 : 0.0;
    out[i++] = slot ? slot.p1 : 0.0;
    out[i++] = slot ? slot.p2 : 0.0;
    out[i++] = slot ? slot.p3 : 0.0;
  }

  // [11] rangeParamsB: (feather0..3)
  for (let r = 0; r < 4; r++) {
    const slot = compiled.ranges[r];
    out[i++] = slot ? slot.p4 : 0.0;
  }

  const adj = layer.adjustments;
  const linActive =
    adj.temperatureShift !== 0 || adj.tintShift !== 0 || Math.abs(adj.exposureEv) > 1e-4;
  const tn = adj.temperatureShift / 100;
  const gn = adj.tintShift / 100;
  const rG = Math.max(0.25, 1 + tn * 0.28 + gn * 0.06);
  const gG = Math.max(0.25, 1 - gn * 0.14);
  const bG = Math.max(0.25, 1 - tn * 0.28 + gn * 0.06);
  const expGain = Math.pow(2, Math.max(-4, Math.min(4, adj.exposureEv)));

  // [12] linearParams: (linearActive, exposureGain, 0, 0)
  out[i++] = linActive ? 1.0 : 0.0;
  out[i++] = expGain;
  out[i++] = 0.0;
  out[i++] = 0.0;

  // [13] wbGains: (r, g, b, 0)
  out[i++] = rG / gG;
  out[i++] = 1.0;
  out[i++] = bG / gG;
  out[i++] = 0.0;

  // [14] toneA: (highlights, shadows, whites, blacks)
  out[i++] = adj.highlights;
  out[i++] = adj.shadows;
  out[i++] = adj.whites;
  out[i++] = adj.blacks;

  // [15] toneB: (dehaze, contrast, saturation, 0)
  out[i++] = adj.dehaze;
  out[i++] = adj.contrast;
  out[i++] = adj.saturation;
  out[i++] = 0.0;

  // [16] detail: (texture, clarity, sharpness, 0)
  out[i++] = adj.texture;
  out[i++] = adj.clarity;
  out[i++] = adj.sharpness;
  out[i++] = 0.0;

  return out;
}

export interface PackSpatialUniformsInput {
  readonly outWidth: number;
  readonly outHeight: number;
  readonly recipe: DevelopRecipe;
  readonly bgTransparent: boolean;
}

/**
 * Packs `SpatialUniforms` (5 × `vec4<f32>` = 80 bytes) matching `shaders/develop/spatial_pass.wgsl`
 * and `internal/infrastructure/src/develop/wgpu_engine.rs::SpatialUniforms` byte-for-byte.
 */
export function packSpatialUniforms(input: PackSpatialUniformsInput): Float32Array {
  const out = new Float32Array(SPATIAL_UNIFORM_FLOATS);
  const u = compileDevelopShaderUniforms(input.recipe);
  const spatialScale = Math.max(1.0, Math.max(input.outWidth, input.outHeight) / 1024.0);
  const bg = input.recipe.backgroundRemoval;

  let i = 0;
  // [0] dims: (outW, outH, spatialScale, 0)
  out[i++] = input.outWidth;
  out[i++] = input.outHeight;
  out[i++] = spatialScale;
  out[i++] = 0.0;

  // [1] cropRect: (0, 0, 1, 1)
  out[i++] = 0.0;
  out[i++] = 0.0;
  out[i++] = 1.0;
  out[i++] = 1.0;

  // [2] detailA: (nr, cnr, texture, clarity)
  out[i++] = u.luminanceNoiseReduction;
  out[i++] = u.colorNoiseReduction;
  out[i++] = u.texture;
  out[i++] = u.clarity;

  // [3] detailB: (sharpening, grainAmount, grainSize, 0)
  out[i++] = u.sharpeningAmount;
  out[i++] = u.grainAmount;
  out[i++] = u.grainSize;
  out[i++] = 0.0;

  // [4] bgParams: (bgMode, bgFeather, 0, 0)
  out[i++] = input.bgTransparent ? 1.0 : 0.0;
  out[i++] = bg?.feather ?? 25;
  out[i++] = 0.0;
  out[i++] = 0.0;

  return out;
}

export function packPresentUniforms(
  uv: ObjectFitUvTransform,
  showShadowClip: boolean,
  showHighlightClip: boolean,
  showCheckerboard: boolean,
): Float32Array {
  const out = new Float32Array(PRESENT_UNIFORM_FLOATS);
  out[0] = uv.scaleX;
  out[1] = uv.scaleY;
  out[2] = uv.offsetX;
  out[3] = uv.offsetY;
  out[4] = showShadowClip ? 1.0 : 0.0;
  out[5] = showHighlightClip ? 1.0 : 0.0;
  out[6] = showCheckerboard ? 1.0 : 0.0;
  out[7] = 0.0;
  return out;
}
