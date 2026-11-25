/**
 * Non-destructive Spot Heal / Clone, Generative Inpaint Patches & Frequency-Separation Portrait
 * Retouching domain model (Phase 1–3).
 * Mirrors Rust `internal/domain/src/retouch.rs` and shared WGSL compute passes
 * (`shaders/develop/inpaint_pass.wgsl`, `shaders/develop/heal_pass.wgsl`, and
 * `shaders/develop/portrait_retouch_pass.wgsl`).
 */

import type { CropRect, CropTransformSettings } from './develop';
import { decodeMatteRleB64 } from './masking';
import type { YuNetFaceOverlay } from './photo';

export const MAX_HEAL_SPOTS = 256;
export const MAX_HEAL_STROKE_POINTS = 96;
export const MAX_INPAINT_PATCHES = 32;
export const MAX_PORTRAIT_FACE_GUIDES = 4;

export type HealSpotMode = 'heal' | 'clone';

export interface InpaintPatch {
  readonly id: string;
  /** Tight normalized bounding box `(left, top, width, height)` in oriented full-frame UV `[0..1]`. */
  readonly bounds: CropRect;
  readonly patchWidth: number;
  readonly patchHeight: number;
  /** Blend opacity (`0.05..1`, default `1.0`). */
  readonly opacity: number;
  /** PackBits RLE Base64 soft feathered alpha matte (`patchWidth × patchHeight`). */
  readonly maskRleB64: string;
  /** Base64-encoded PNG RGB patch pixels (`patchWidth × patchHeight`). */
  readonly patchImageB64: string;
  /** Human-readable generator label (e.g. `"LaMa AI"` or `"Harmonic Fill"`). */
  readonly engineLabel: string;
}

export interface InpaintStrokeInput {
  /** Brush radius as a fraction of the shorter oriented image side (`0.004..0.25`). */
  readonly radius: number;
  /** Polyline points `[u, v]` in oriented full-frame normalized UV `[0..1]`. */
  readonly points: readonly (readonly [number, number])[];
}

export interface GeneratedInpaintPatchResult {
  readonly patch: InpaintPatch;
  readonly elapsedMs: number;
}

export interface FaceParseMapDto {
  readonly width: number;
  readonly height: number;
  /** Base64-encoded packed RGBA8 semantic channels (`R=skin, G=eyes, B=teeth, A=lips`). */
  readonly rgbaB64: string;
  readonly engineLabel: string;
  readonly elapsedMs: number;
}

export interface FaceParseMapData {
  readonly width: number;
  readonly height: number;
  readonly rgba: Uint8Array;
  readonly engineLabel: string;
}

export interface HealSpot {
  readonly id: string;
  readonly mode: HealSpotMode;
  /** Destination blemish center `(u, v)` in oriented full-frame normalized space `[0..1]`. */
  readonly targetX: number;
  readonly targetY: number;
  /** Donor clean patch center `(u, v)` in oriented full-frame normalized space `[0..1]`. */
  readonly sourceX: number;
  readonly sourceY: number;
  /** Spot/brush radius as a fraction of the shorter oriented image side (`0.002..0.20`). */
  readonly radius: number;
  /** Soft edge feathering (`0..1`, default `0.55`). */
  readonly feather: number;
  /** Blend opacity (`0.05..1`, default `1.0`). */
  readonly opacity: number;
  /** Optional polyline stroke vertices `[u, v]` for stray-hair / linear heal strokes. */
  readonly points?: readonly (readonly [number, number])[];
}

export interface PortraitRetouchSettings {
  readonly enabled: boolean;
  /** Frequency-separation skin smoothing (`0..100`). */
  readonly skinSmoothing: number;
  /** High-frequency pore preservation (`0..100`, default `70`). */
  readonly skinTexturePreserve: number;
  /** Skin redness / tone blotchiness evening (`0..100`). */
  readonly skinToneEvenness: number;
  /** Specular glare & oil sheen control (`0..100`). */
  readonly shineRemoval: number;
  /** Under-eye shadow & dark-circle brightening (`0..100`). */
  readonly underEyeBrighten: number;
  /** Eye clarity, iris contrast & sclera brightening (`0..100`). */
  readonly eyeEnhance: number;
  /** Natural teeth whitening (`0..100`). */
  readonly teethWhiten: number;
  /** Subtle lip vibrancy & definition (`0..100`). */
  readonly lipEnhance: number;
}

export interface RetouchSettings {
  readonly healSpots: readonly HealSpot[];
  readonly inpaintPatches?: readonly InpaintPatch[];
  readonly portrait: PortraitRetouchSettings;
}

export interface PortraitFaceGuide {
  readonly boxLeft: number;
  readonly boxTop: number;
  readonly boxWidth: number;
  readonly boxHeight: number;
  readonly rightEye: readonly [number, number];
  readonly leftEye: readonly [number, number];
  readonly noseTip: readonly [number, number];
  readonly rightMouth: readonly [number, number];
  readonly leftMouth: readonly [number, number];
}

export interface PortraitRetouchPreset {
  readonly id: string;
  readonly label: string;
  readonly summary: string;
  readonly portrait: PortraitRetouchSettings;
}

export const PORTRAIT_RETOUCH_PRESETS: readonly PortraitRetouchPreset[] = [
  {
    id: 'natural-studio',
    label: 'Natural',
    summary: 'Subtle pore-safe skin evenness & gentle eye lift',
    portrait: {
      enabled: true,
      skinSmoothing: 30,
      skinTexturePreserve: 78,
      skinToneEvenness: 25,
      shineRemoval: 22,
      underEyeBrighten: 22,
      eyeEnhance: 18,
      teethWhiten: 20,
      lipEnhance: 10,
    },
  },
  {
    id: 'studio-polish',
    label: 'Studio Polish',
    summary: 'Balanced graduation & corporate headshot finish',
    portrait: {
      enabled: true,
      skinSmoothing: 50,
      skinTexturePreserve: 70,
      skinToneEvenness: 40,
      shineRemoval: 38,
      underEyeBrighten: 36,
      eyeEnhance: 30,
      teethWhiten: 34,
      lipEnhance: 18,
    },
  },
  {
    id: 'editorial-soft',
    label: 'Editorial',
    summary: 'Silky frequency separation with crisp eyes',
    portrait: {
      enabled: true,
      skinSmoothing: 72,
      skinTexturePreserve: 62,
      skinToneEvenness: 55,
      shineRemoval: 50,
      underEyeBrighten: 46,
      eyeEnhance: 42,
      teethWhiten: 42,
      lipEnhance: 26,
    },
  },
];

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, v));
}

export function createDefaultPortraitRetouch(): PortraitRetouchSettings {
  return {
    enabled: true,
    skinSmoothing: 0,
    skinTexturePreserve: 70,
    skinToneEvenness: 0,
    shineRemoval: 0,
    underEyeBrighten: 0,
    eyeEnhance: 0,
    teethWhiten: 0,
    lipEnhance: 0,
  };
}

export function createDefaultRetouchSettings(): RetouchSettings {
  return {
    healSpots: [],
    inpaintPatches: [],
    portrait: createDefaultPortraitRetouch(),
  };
}

export function isPortraitRetouchSliderActive(portrait: PortraitRetouchSettings | undefined): boolean {
  if (!portrait) return false;
  return (
    portrait.skinSmoothing > 0 ||
    portrait.skinToneEvenness > 0 ||
    portrait.shineRemoval > 0 ||
    portrait.underEyeBrighten > 0 ||
    portrait.eyeEnhance > 0 ||
    portrait.teethWhiten > 0 ||
    portrait.lipEnhance > 0
  );
}

export function isPortraitRetouchActive(portrait: PortraitRetouchSettings | undefined): boolean {
  return Boolean(portrait && portrait.enabled && isPortraitRetouchSliderActive(portrait));
}

export function hasEffectiveHealSpots(retouch: RetouchSettings | undefined): boolean {
  if (!retouch || !retouch.healSpots || retouch.healSpots.length === 0) return false;
  return retouch.healSpots.some((s) => s.radius >= 0.002 && s.opacity >= 0.05);
}

export function isInpaintPatchEffective(patch: InpaintPatch | undefined): boolean {
  return Boolean(
    patch &&
      patch.patchWidth > 0 &&
      patch.patchHeight > 0 &&
      patch.opacity >= 0.05 &&
      patch.bounds.width > 1e-4 &&
      patch.bounds.height > 1e-4 &&
      patch.maskRleB64.length > 0 &&
      patch.patchImageB64.length > 0,
  );
}

export function hasEffectiveInpaintPatches(retouch: RetouchSettings | undefined): boolean {
  if (!retouch || !retouch.inpaintPatches || retouch.inpaintPatches.length === 0) return false;
  return retouch.inpaintPatches.some(isInpaintPatchEffective);
}

export function isRetouchActive(retouch: RetouchSettings | undefined): boolean {
  return (
    hasEffectiveHealSpots(retouch) ||
    hasEffectiveInpaintPatches(retouch) ||
    isPortraitRetouchActive(retouch?.portrait)
  );
}

export function hydrateInpaintPatch(raw: Partial<InpaintPatch>, index = 0): InpaintPatch {
  const b = raw.bounds;
  const left = clamp(b?.left ?? 0, 0, 0.999);
  const top = clamp(b?.top ?? 0, 0, 0.999);
  const width = clamp(b?.width ?? 0.05, 0.001, Math.max(0.001, 1 - left));
  const height = clamp(b?.height ?? 0.05, 0.001, Math.max(0.001, 1 - top));
  return {
    id: raw.id && raw.id.trim().length > 0 ? raw.id : `inpaint-${index + 1}`,
    bounds: { left, top, width, height },
    patchWidth: Math.max(0, Math.min(2048, Math.round(raw.patchWidth ?? 0))),
    patchHeight: Math.max(0, Math.min(2048, Math.round(raw.patchHeight ?? 0))),
    opacity: clamp(raw.opacity ?? 1, 0.05, 1),
    maskRleB64: typeof raw.maskRleB64 === 'string' ? raw.maskRleB64 : '',
    patchImageB64: typeof raw.patchImageB64 === 'string' ? raw.patchImageB64 : '',
    engineLabel:
      typeof raw.engineLabel === 'string' && raw.engineLabel.trim().length > 0
        ? raw.engineLabel.trim()
        : 'AI Remove',
  };
}

export function decodeBase64ToBytes(b64: string): Uint8Array | null {
  if (!b64) return null;
  try {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      out[i] = bin.charCodeAt(i);
    }
    return out;
  } catch {
    return null;
  }
}

export function decodeFaceParseMapDto(
  dto: FaceParseMapDto | null | undefined,
): FaceParseMapData | null {
  if (!dto || dto.width <= 0 || dto.height <= 0 || !dto.rgbaB64) return null;
  const bytes = decodeBase64ToBytes(dto.rgbaB64);
  if (!bytes || bytes.length !== dto.width * dto.height * 4) return null;
  return {
    width: dto.width,
    height: dto.height,
    rgba: bytes,
    engineLabel: dto.engineLabel || 'BiSeNet Face Parsing',
  };
}

export interface DecodedInpaintPatchRgba {
  readonly width: number;
  readonly height: number;
  readonly rgba: Uint8Array;
}

/**
 * Decodes a compact `InpaintPatch` (PNG RGB + PackBits RLE soft feather alpha matte) into a
 * ready-to-upload RGBA8 buffer (`rgb = synthesized fill`, `a = feathered mask`) for WebGPU / WebGL2.
 */
export async function decodeInpaintPatchRgba(
  patch: InpaintPatch,
): Promise<DecodedInpaintPatchRgba | null> {
  if (!isInpaintPatchEffective(patch)) return null;
  const w = patch.patchWidth;
  const h = patch.patchHeight;
  const imgBytes = decodeBase64ToBytes(patch.patchImageB64);
  const alphaMask = decodeMatteRleB64(patch.maskRleB64, w * h);
  if (!imgBytes || !alphaMask || alphaMask.length !== w * h) return null;

  try {
    const blob = new Blob([imgBytes as Uint8Array<ArrayBuffer>], { type: 'image/png' });
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx2d = canvas.getContext('2d');
    if (!ctx2d) return null;

    if (typeof createImageBitmap === 'function') {
      const bitmap = await createImageBitmap(blob);
      try {
        ctx2d.drawImage(bitmap, 0, 0, w, h);
      } finally {
        bitmap.close();
      }
    } else {
      const url = URL.createObjectURL(blob);
      try {
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const el = new Image();
          el.onload = () => resolve(el);
          el.onerror = reject;
          el.src = url;
        });
        ctx2d.drawImage(img, 0, 0, w, h);
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    const rgba = new Uint8Array(ctx2d.getImageData(0, 0, w, h).data);
    for (let i = 0; i < w * h; i++) {
      rgba[i * 4 + 3] = alphaMask[i]!;
    }
    return { width: w, height: h, rgba };
  } catch {
    return null;
  }
}

export function hydrateHealSpot(raw: Partial<HealSpot>, index = 0): HealSpot {
  const tx = clamp(raw.targetX ?? 0.5, -0.25, 1.25);
  const ty = clamp(raw.targetY ?? 0.5, -0.25, 1.25);
  const sx = clamp(raw.sourceX ?? clamp(tx + 0.04, 0, 1), -0.25, 1.25);
  const sy = clamp(raw.sourceY ?? ty, -0.25, 1.25);
  const points =
    Array.isArray(raw.points) && raw.points.length > 0
      ? raw.points
          .slice(0, MAX_HEAL_STROKE_POINTS)
          .map((pt) => [clamp(pt[0] ?? tx, -0.25, 1.25), clamp(pt[1] ?? ty, -0.25, 1.25)] as const)
      : undefined;

  return {
    id: raw.id && raw.id.trim().length > 0 ? raw.id : `heal-${index + 1}`,
    mode: raw.mode === 'clone' ? 'clone' : 'heal',
    targetX: tx,
    targetY: ty,
    sourceX: sx,
    sourceY: sy,
    radius: clamp(raw.radius ?? 0.018, 0.002, 0.2),
    feather: clamp(raw.feather ?? 0.55, 0, 1),
    opacity: clamp(raw.opacity ?? 1, 0.05, 1),
    ...(points ? { points } : {}),
  };
}

export function hydrateRetouchSettings(
  raw: Partial<RetouchSettings> | null | undefined,
): RetouchSettings {
  const def = createDefaultRetouchSettings();
  if (!raw) return def;
  const p = raw.portrait;
  return {
    healSpots: Array.isArray(raw.healSpots)
      ? raw.healSpots.slice(0, MAX_HEAL_SPOTS).map((s, idx) => hydrateHealSpot(s, idx))
      : [],
    inpaintPatches: Array.isArray(raw.inpaintPatches)
      ? raw.inpaintPatches
          .slice(0, MAX_INPAINT_PATCHES)
          .map((patch, idx) => hydrateInpaintPatch(patch, idx))
      : [],
    portrait: {
      enabled: p?.enabled ?? def.portrait.enabled,
      skinSmoothing: Math.round(clamp(p?.skinSmoothing ?? 0, 0, 100)),
      skinTexturePreserve: Math.round(clamp(p?.skinTexturePreserve ?? 70, 0, 100)),
      skinToneEvenness: Math.round(clamp(p?.skinToneEvenness ?? 0, 0, 100)),
      shineRemoval: Math.round(clamp(p?.shineRemoval ?? 0, 0, 100)),
      underEyeBrighten: Math.round(clamp(p?.underEyeBrighten ?? 0, 0, 100)),
      eyeEnhance: Math.round(clamp(p?.eyeEnhance ?? 0, 0, 100)),
      teethWhiten: Math.round(clamp(p?.teethWhiten ?? 0, 0, 100)),
      lipEnhance: Math.round(clamp(p?.lipEnhance ?? 0, 0, 100)),
    },
  };
}

/**
 * Exact inverse of `grade_pass.wgsl::orientSourceUv` + forward straighten rotation.
 * Mirrors `internal/domain/src/retouch.rs::source_uv_to_oriented_uv`.
 */
export function sourceUvToOrientedUv(
  su: number,
  sv: number,
  crop: CropTransformSettings,
  oriWidth: number,
  oriHeight: number,
  includeFineRotation: boolean,
): readonly [number, number] {
  const qt = (((crop.quarterTurns ?? 0) % 4) + 4) % 4;
  let px = su;
  let py = sv;
  if (qt === 1) {
    px = 1 - sv;
    py = su;
  } else if (qt === 2) {
    px = 1 - su;
    py = 1 - sv;
  } else if (qt === 3) {
    px = sv;
    py = 1 - su;
  }
  let u = crop.flipHorizontal ? 1 - px : px;
  let v = crop.flipVertical ? 1 - py : py;

  if (includeFineRotation && Math.abs(crop.rotationDeg) > 1e-3) {
    const oriW = Math.max(1, oriWidth);
    const oriH = Math.max(1, oriHeight);
    const cx = (oriW - 1) * 0.5;
    const cy = (oriH - 1) * 0.5;
    const dx = u * oriW - 0.5 - cx;
    const dy = v * oriH - 0.5 - cy;
    const rad = (crop.rotationDeg * Math.PI) / 180;
    const cosA = Math.cos(rad);
    const sinA = Math.sin(rad);
    const rx = dx * cosA - dy * sinA + cx;
    const ry = dx * sinA + dy * cosA + cy;
    u = (rx + 0.5) / oriW;
    v = (ry + 0.5) / oriH;
  }

  return [u, v] as const;
}

/**
 * Converts a photo's `YuNetFaceOverlay` (percentage `[0..100]` raw space) into oriented full-frame
 * `[0..1]` UV coordinates matching `grade_pass.wgsl`.
 */
export function faceOverlayToOrientedGuide(
  overlay: YuNetFaceOverlay | null | undefined,
  crop: CropTransformSettings,
  oriWidth: number,
  oriHeight: number,
  includeFineRotation: boolean,
): PortraitFaceGuide | null {
  if (!overlay || overlay.widthPct <= 0.1 || overlay.heightPct <= 0.1) return null;

  const mapPt = (u: number, v: number) =>
    sourceUvToOrientedUv(u, v, crop, oriWidth, oriHeight, includeFineRotation);

  const l = overlay.leftPct / 100;
  const t = overlay.topPct / 100;
  const w = overlay.widthPct / 100;
  const h = overlay.heightPct / 100;

  // In `YuNetFaceOverlay` (from `build_face_overlay` in `kroma-tau/src/dto.rs`), the 5 landmark
  // points are percentages `[0..100]` RELATIVE to the face bounding box `(l, t, w, h)`.
  const relToFull = (pt: { readonly x: number; readonly y: number }) =>
    mapPt(l + (pt.x / 100) * w, t + (pt.y / 100) * h);

  const corners = [
    mapPt(l, t),
    mapPt(l + w, t),
    mapPt(l, t + h),
    mapPt(l + w, t + h),
  ];
  let minU = Infinity;
  let minV = Infinity;
  let maxU = -Infinity;
  let maxV = -Infinity;
  for (const [cu, cv] of corners) {
    if (cu < minU) minU = cu;
    if (cv < minV) minV = cv;
    if (cu > maxU) maxU = cu;
    if (cv > maxV) maxV = cv;
  }

  return {
    boxLeft: minU,
    boxTop: minV,
    boxWidth: Math.max(0.01, maxU - minU),
    boxHeight: Math.max(0.01, maxV - minV),
    rightEye: relToFull(overlay.rightEye),
    leftEye: relToFull(overlay.leftEye),
    noseTip: relToFull(overlay.noseTip),
    rightMouth: relToFull(overlay.rightMouth),
    leftMouth: relToFull(overlay.leftMouth),
  };
}

export interface CompiledHealGpuBuffers {
  readonly spotCount: number;
  /** 16 floats (64 bytes) per spot matching `HealSpotGpu` in `heal_pass.wgsl`. */
  readonly spotsData: Float32Array;
  /** 4 floats (16 bytes) per segment `(ax, ay, bx, by)` matching `segments` in `heal_pass.wgsl`. */
  readonly segmentsData: Float32Array;
}

/**
 * Compiles non-destructive `HealSpot[]` into tightly packed 16-byte-aligned storage buffers
 * matching `shaders/develop/heal_pass.wgsl` and `internal/infrastructure/src/develop/retouch_engine.rs`.
 */
export function compileHealSpotsForGpu(
  spots: readonly HealSpot[] | undefined,
  oriWidth: number,
  oriHeight: number,
): CompiledHealGpuBuffers {
  const effective = (spots ?? []).filter((s) => s.radius >= 0.002 && s.opacity >= 0.05).slice(0, MAX_HEAL_SPOTS);
  if (effective.length === 0) {
    return {
      spotCount: 0,
      spotsData: new Float32Array(16),
      segmentsData: new Float32Array(4),
    };
  }

  const oriW = Math.max(1, oriWidth);
  const oriH = Math.max(1, oriHeight);
  const minSide = Math.min(oriW, oriH);

  const spotsFloats: number[] = [];
  const segFloats: number[] = [];

  for (const spot of effective) {
    const segStart = segFloats.length / 4;
    const pts =
      spot.points && spot.points.length > 0
        ? spot.points
        : ([[spot.targetX, spot.targetY]] as const);

    let minU = Infinity;
    let minV = Infinity;
    let maxU = -Infinity;
    let maxV = -Infinity;

    if (pts.length === 1) {
      const p = pts[0]!;
      segFloats.push(p[0], p[1], p[0], p[1]);
      minU = maxU = p[0];
      minV = maxV = p[1];
    } else {
      for (let k = 0; k < pts.length - 1; k++) {
        const a = pts[k]!;
        const b = pts[k + 1]!;
        segFloats.push(a[0], a[1], b[0], b[1]);
        minU = Math.min(minU, a[0], b[0]);
        minV = Math.min(minV, a[1], b[1]);
        maxU = Math.max(maxU, a[0], b[0]);
        maxV = Math.max(maxV, a[1], b[1]);
      }
    }

    const segCount = segFloats.length / 4 - segStart;
    const marginU = (spot.radius * minSide) / oriW + 2 / oriW;
    const marginV = (spot.radius * minSide) / oriH + 2 / oriH;

    const deltaU = spot.sourceX - spot.targetX;
    const deltaV = spot.sourceY - spot.targetY;

    // [0] geom: (deltaU, deltaV, radiusNorm, feather)
    spotsFloats.push(deltaU, deltaV, spot.radius, spot.feather);
    // [1] params: (opacity, mode (0=heal, 1=clone), segStart, segCount)
    spotsFloats.push(spot.opacity, spot.mode === 'clone' ? 1.0 : 0.0, segStart, segCount);
    // [2] bounds: (minU, minV, maxU, maxV)
    spotsFloats.push(minU - marginU, minV - marginV, maxU + marginU, maxV + marginV);
    // [3] center: (targetX, targetY, 0, 0)
    spotsFloats.push(spot.targetX, spot.targetY, 0, 0);
  }

  return {
    spotCount: effective.length,
    spotsData: new Float32Array(spotsFloats),
    segmentsData: new Float32Array(segFloats.length > 0 ? segFloats : [0, 0, 0, 0]),
  };
}

/**
 * Smart automatic donor-patch selector when the user clicks a blemish.
 * Evaluates 12 radial candidates around `(targetU, targetV)` while staying inside frame bounds
 * and avoiding facial features (eyes/mouth) if a face guide is present.
 * When `samplePatchStats` is supplied (reading live preview pixels), ranks candidates by
 * color match to the blemish's clean surrounding ring + low internal texture variance.
 */
export function pickAutoHealSourceUv(
  targetU: number,
  targetV: number,
  radius: number,
  oriWidth: number,
  oriHeight: number,
  faceGuide?: PortraitFaceGuide | null,
  samplePatchStats?: (u: number, v: number, radiusNorm: number) => {
    readonly meanR: number;
    readonly meanG: number;
    readonly meanB: number;
    readonly variance: number;
  } | null,
): readonly [number, number] {
  const minSide = Math.min(Math.max(1, oriWidth), Math.max(1, oriHeight));
  const rxUv = (radius * minSide) / Math.max(1, oriWidth);
  const ryUv = (radius * minSide) / Math.max(1, oriHeight);

  // Reference clean ring color just outside the target blemish (when live sampler is available)
  let ringR = 0;
  let ringG = 0;
  let ringB = 0;
  let ringSamples = 0;
  if (samplePatchStats) {
    for (let k = 0; k < 8; k++) {
      const ang = (k * Math.PI) / 4;
      const ru = clamp(targetU + Math.cos(ang) * rxUv * 1.45, 0.01, 0.99);
      const rv = clamp(targetV + Math.sin(ang) * ryUv * 1.45, 0.01, 0.99);
      const st = samplePatchStats(ru, rv, radius * 0.4);
      if (st) {
        ringR += st.meanR;
        ringG += st.meanG;
        ringB += st.meanB;
        ringSamples += 1;
      }
    }
    if (ringSamples > 0) {
      ringR /= ringSamples;
      ringG /= ringSamples;
      ringB /= ringSamples;
    }
  }

  const featurePenalty = (u: number, v: number): number => {
    if (!faceGuide) return 0;
    const eyeDist = Math.max(
      0.02,
      Math.hypot(
        faceGuide.leftEye[0] - faceGuide.rightEye[0],
        faceGuide.leftEye[1] - faceGuide.rightEye[1],
      ),
    );
    const mouthCx = 0.5 * (faceGuide.rightMouth[0] + faceGuide.leftMouth[0]);
    const mouthCy = 0.5 * (faceGuide.rightMouth[1] + faceGuide.leftMouth[1]);
    let pen = 0;
    const avoid = [
      { p: faceGuide.rightEye, r: eyeDist * 0.34 },
      { p: faceGuide.leftEye, r: eyeDist * 0.34 },
      { p: faceGuide.noseTip, r: eyeDist * 0.24 },
      { p: [mouthCx, mouthCy] as const, r: eyeDist * 0.38 },
    ];
    for (const item of avoid) {
      const d = Math.hypot(u - item.p[0], v - item.p[1]);
      if (d < item.r + rxUv) {
        pen += 1.5 * (1 - d / Math.max(1e-4, item.r + rxUv));
      }
    }
    return pen;
  };

  let bestU = clamp(targetU + rxUv * 2.6, 0.03, 0.97);
  let bestV = clamp(targetV - ryUv * 0.5, 0.03, 0.97);
  let bestScore = Infinity;

  const rings = [2.6, 3.8];
  for (const distMult of rings) {
    for (let k = 0; k < 8; k++) {
      const ang = (k * Math.PI) / 4 + (distMult > 3 ? Math.PI / 8 : 0);
      const cu = targetU + Math.cos(ang) * rxUv * distMult;
      const cv = targetV + Math.sin(ang) * ryUv * distMult;
      if (cu < 0.025 || cu > 0.975 || cv < 0.025 || cv > 0.975) continue;

      let score = distMult * 0.03 + featurePenalty(cu, cv);
      if (samplePatchStats && ringSamples > 0) {
        const st = samplePatchStats(cu, cv, radius * 0.85);
        if (st) {
          const colorDiff =
            Math.abs(st.meanR - ringR) +
            Math.abs(st.meanG - ringG) +
            Math.abs(st.meanB - ringB);
          score += colorDiff * 2.4 + Math.sqrt(Math.max(0, st.variance)) * 1.3;
        }
      }

      if (score < bestScore) {
        bestScore = score;
        bestU = cu;
        bestV = cv;
      }
    }
  }

  return [clamp(bestU, 0.01, 0.99), clamp(bestV, 0.01, 0.99)] as const;
}

export function createHealSpot(
  mode: HealSpotMode,
  targetX: number,
  targetY: number,
  sourceX: number,
  sourceY: number,
  radius: number,
  feather: number,
  opacity: number,
  points?: readonly (readonly [number, number])[],
): HealSpot {
  return hydrateHealSpot({
    id: `heal-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    mode,
    targetX,
    targetY,
    sourceX,
    sourceY,
    radius,
    feather,
    opacity,
    points,
  });
}
