/**
 * Non-destructive Local & AI Masking domain model, PackBits RLE+Base64 codec,
 * and spatial mask rasterizer matching Rust `internal/domain/src/masking.rs`
 * and `internal/infrastructure/src/develop/mask_engine.rs` formula-for-formula.
 */

export type MaskCombineOp = 'add' | 'subtract' | 'intersect';
export type AiMaskKind = 'subject' | 'background' | 'face_skin' | 'sky';
export type BackgroundReplaceMode = 'original' | 'transparent' | 'solid_color' | 'gradient';
export type BackgroundGradientStyle = 'radial' | 'linear';

export interface BackgroundRemovalSettings {
  readonly enabled: boolean;
  readonly mode: BackgroundReplaceMode;
  /** Solid backdrop RGB `[0..255]`, or Gradient Color 1 (center / start). */
  readonly colorRgb: readonly [number, number, number];
  /** Gradient Color 2 (outer / end) RGB `[0..255]` used when `mode === 'gradient'`. */
  readonly color2Rgb: readonly [number, number, number];
  /** Gradient geometry (`'radial'` portrait spotlight or `'linear'` angle wash). */
  readonly gradientStyle: BackgroundGradientStyle;
  /** Linear gradient direction in CSS degrees (`0..360`, default `180` = top → bottom). */
  readonly gradientAngleDeg: number;
  /** Edge softness refinement (`0..100`, default `25`). */
  readonly feather: number;
  readonly matteWidth: number;
  readonly matteHeight: number;
  readonly matteRleB64: string;
}

export const STUDIO_BACKDROP_SWATCHES: readonly {
  readonly id: string;
  readonly label: string;
  readonly rgb: readonly [number, number, number];
  readonly hex: string;
}[] = [
  { id: 'white', label: 'Pure White', rgb: [255, 255, 255], hex: '#FFFFFF' },
  { id: 'cream', label: 'Studio Cream', rgb: [244, 239, 230], hex: '#F4EFE6' },
  { id: 'gray', label: 'Mid Gray', rgb: [142, 146, 153], hex: '#8E9299' },
  { id: 'charcoal', label: 'Charcoal', rgb: [43, 45, 49], hex: '#2B2D31' },
  { id: 'navy', label: 'Studio Navy', rgb: [27, 42, 74], hex: '#1B2A4A' },
  { id: 'black', label: 'Deep Black', rgb: [14, 14, 16], hex: '#0E0E10' },
];

export interface StudioGradientPreset {
  readonly id: string;
  readonly label: string;
  readonly style: BackgroundGradientStyle;
  readonly angleDeg: number;
  readonly color1Rgb: readonly [number, number, number];
  readonly color2Rgb: readonly [number, number, number];
}

export const STUDIO_GRADIENT_PRESETS: readonly StudioGradientPreset[] = [
  {
    id: 'studio-charcoal',
    label: 'Studio Charcoal',
    style: 'radial',
    angleDeg: 180,
    color1Rgb: [74, 79, 90],
    color2Rgb: [20, 22, 27],
  },
  {
    id: 'soft-pearl',
    label: 'Soft Pearl',
    style: 'radial',
    angleDeg: 180,
    color1Rgb: [255, 255, 255],
    color2Rgb: [210, 215, 224],
  },
  {
    id: 'classic-blue',
    label: 'Classic Studio Blue',
    style: 'radial',
    angleDeg: 180,
    color1Rgb: [63, 98, 145],
    color2Rgb: [18, 30, 51],
  },
  {
    id: 'warm-sepia',
    label: 'Warm Sepia',
    style: 'radial',
    angleDeg: 180,
    color1Rgb: [110, 88, 71],
    color2Rgb: [33, 25, 20],
  },
  {
    id: 'midnight-teal',
    label: 'Midnight Teal',
    style: 'linear',
    angleDeg: 145,
    color1Rgb: [30, 63, 75],
    color2Rgb: [13, 21, 29],
  },
  {
    id: 'golden-cream',
    label: 'Golden Cream',
    style: 'linear',
    angleDeg: 180,
    color1Rgb: [247, 239, 226],
    color2Rgb: [218, 198, 172],
  },
];

export function createDefaultBackgroundRemoval(): BackgroundRemovalSettings {
  return {
    enabled: false,
    mode: 'original',
    colorRgb: [74, 79, 90],
    color2Rgb: [20, 22, 27],
    gradientStyle: 'radial',
    gradientAngleDeg: 180,
    feather: 25,
    matteWidth: 0,
    matteHeight: 0,
    matteRleB64: '',
  };
}

export function backgroundRemovalPreviewCss(
  bg: Pick<
    BackgroundRemovalSettings,
    'colorRgb' | 'color2Rgb' | 'gradientStyle' | 'gradientAngleDeg'
  >,
): string {
  const hex1 = rgbTupleToHex(bg.colorRgb);
  const hex2 = rgbTupleToHex(bg.color2Rgb);
  if (bg.gradientStyle === 'linear') {
    return `linear-gradient(${Math.round(bg.gradientAngleDeg)}deg, ${hex1}, ${hex2})`;
  }
  return `radial-gradient(circle at 50% 42%, ${hex1}, ${hex2})`;
}

export function isBackgroundRemovalActive(bg: BackgroundRemovalSettings | undefined | null): boolean {
  return Boolean(
    bg &&
      bg.enabled &&
      bg.mode !== 'original' &&
      bg.matteWidth > 0 &&
      bg.matteHeight > 0 &&
      bg.matteRleB64.length > 0,
  );
}

function clampRgbTuple(
  raw: readonly [number, number, number] | undefined,
  fallback: readonly [number, number, number],
): readonly [number, number, number] {
  if (!Array.isArray(raw) || raw.length !== 3) return fallback;
  return [
    Math.max(0, Math.min(255, Math.round(raw[0] ?? fallback[0]))),
    Math.max(0, Math.min(255, Math.round(raw[1] ?? fallback[1]))),
    Math.max(0, Math.min(255, Math.round(raw[2] ?? fallback[2]))),
  ] as const;
}

export function hydrateBackgroundRemoval(
  raw: Partial<BackgroundRemovalSettings> | undefined | null,
): BackgroundRemovalSettings {
  const def = createDefaultBackgroundRemoval();
  if (!raw) return def;
  const mode: BackgroundReplaceMode =
    raw.mode === 'transparent' || raw.mode === 'solid_color' || raw.mode === 'gradient'
      ? raw.mode
      : 'original';
  const gradientStyle: BackgroundGradientStyle =
    raw.gradientStyle === 'linear' ? 'linear' : 'radial';
  const rawAngle =
    typeof raw.gradientAngleDeg === 'number' && Number.isFinite(raw.gradientAngleDeg)
      ? ((raw.gradientAngleDeg % 360) + 360) % 360
      : def.gradientAngleDeg;
  return {
    enabled: Boolean(raw.enabled),
    mode,
    colorRgb: clampRgbTuple(raw.colorRgb, [255, 255, 255]),
    color2Rgb: clampRgbTuple(raw.color2Rgb, def.color2Rgb),
    gradientStyle,
    gradientAngleDeg: rawAngle,
    feather: typeof raw.feather === 'number' && Number.isFinite(raw.feather) ? raw.feather : 25,
    matteWidth: typeof raw.matteWidth === 'number' ? raw.matteWidth : 0,
    matteHeight: typeof raw.matteHeight === 'number' ? raw.matteHeight : 0,
    matteRleB64: typeof raw.matteRleB64 === 'string' ? raw.matteRleB64 : '',
  };
}

export function rgbTupleToHex(rgb: readonly [number, number, number]): string {
  const toHex = (n: number) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0');
  return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`.toUpperCase();
}

export function hexToRgbTuple(hex: string): readonly [number, number, number] {
  const clean = hex.replace('#', '').trim();
  if (clean.length !== 6) return [255, 255, 255];
  const r = Number.parseInt(clean.slice(0, 2), 16);
  const g = Number.parseInt(clean.slice(2, 4), 16);
  const b = Number.parseInt(clean.slice(4, 6), 16);
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return [255, 255, 255];
  return [r, g, b];
}

export interface BrushStroke {
  /** Radius as a fraction of the shorter image dimension (0.004..0.35). */
  readonly size: number;
  /** Soft falloff (0..1). */
  readonly feather: number;
  /** Per-stroke opacity cap (0.05..1). */
  readonly flow: number;
  /** Ordered polyline vertices `[u, v]` in `[0..1]` oriented image space. */
  readonly points: readonly (readonly [number, number])[];
}

export interface LinearGradientComponent {
  readonly type: 'linear_gradient';
  readonly id: string;
  readonly op: MaskCombineOp;
  readonly inverted: boolean;
  readonly startX: number;
  readonly startY: number;
  readonly endX: number;
  readonly endY: number;
  readonly feather: number; // 0..100
}

export interface RadialGradientComponent {
  readonly type: 'radial_gradient';
  readonly id: string;
  readonly op: MaskCombineOp;
  readonly inverted: boolean;
  readonly centerX: number;
  readonly centerY: number;
  readonly radiusX: number;
  readonly radiusY: number;
  readonly rotationDeg: number; // -180..180
  readonly feather: number; // 0..100
}

export interface BrushComponent {
  readonly type: 'brush';
  readonly id: string;
  readonly op: MaskCombineOp;
  readonly inverted: boolean;
  readonly strokes: readonly BrushStroke[];
}

export interface LuminanceRangeComponent {
  readonly type: 'luminance_range';
  readonly id: string;
  readonly op: MaskCombineOp;
  readonly inverted: boolean;
  readonly minLuma: number; // 0..1
  readonly maxLuma: number; // 0..1
  readonly feather: number; // 0..0.5
}

export interface ColorRangeComponent {
  readonly type: 'color_range';
  readonly id: string;
  readonly op: MaskCombineOp;
  readonly inverted: boolean;
  readonly targetHueDeg: number; // 0..360
  readonly targetSat: number; // 0..1
  readonly targetLuma: number; // 0..1
  readonly tolerance: number; // 0.02..1
  readonly feather: number; // 0..1
}

export interface AiMatteComponent {
  readonly type: 'ai_matte';
  readonly id: string;
  readonly op: MaskCombineOp;
  readonly inverted: boolean;
  readonly kind: AiMaskKind;
  readonly feather: number; // 0..100 (default 25)
  readonly matteWidth: number;
  readonly matteHeight: number;
  readonly matteRleB64: string;
}

export type MaskComponent =
  | LinearGradientComponent
  | RadialGradientComponent
  | BrushComponent
  | LuminanceRangeComponent
  | ColorRangeComponent
  | AiMatteComponent;

export interface LocalAdjustments {
  readonly temperatureShift: number; // -100..+100
  readonly tintShift: number; // -100..+100
  readonly exposureEv: number; // -4.0..+4.0
  readonly contrast: number; // -100..+100
  readonly highlights: number; // -100..+100
  readonly shadows: number; // -100..+100
  readonly whites: number; // -100..+100
  readonly blacks: number; // -100..+100
  readonly texture: number; // -100..+100
  readonly clarity: number; // -100..+100
  readonly dehaze: number; // -100..+100
  readonly saturation: number; // -100..+100
  readonly sharpness: number; // -100..+100 (negative softens skin)
}

export interface MaskLayer {
  readonly id: string;
  readonly name: string;
  readonly enabled: boolean;
  readonly inverted: boolean;
  readonly opacity: number; // 0..100
  readonly components: readonly MaskComponent[];
  readonly adjustments: LocalAdjustments;
}

export interface GeneratedAiMaskResult {
  readonly kind: AiMaskKind;
  readonly matteWidth: number;
  readonly matteHeight: number;
  readonly matteRleB64: string;
  readonly engineLabel: string;
  readonly elapsedMs: number;
}

let idCounter = 1;
export function createMaskId(prefix = 'mask'): string {
  idCounter += 1;
  return `${prefix}-${Date.now().toString(36)}-${idCounter.toString(36)}`;
}

export function createDefaultLocalAdjustments(): LocalAdjustments {
  return {
    temperatureShift: 0,
    tintShift: 0,
    exposureEv: 0,
    contrast: 0,
    highlights: 0,
    shadows: 0,
    whites: 0,
    blacks: 0,
    texture: 0,
    clarity: 0,
    dehaze: 0,
    saturation: 0,
    sharpness: 0,
  };
}

export function isLocalAdjustmentsModified(adj: LocalAdjustments): boolean {
  return (
    adj.temperatureShift !== 0 ||
    adj.tintShift !== 0 ||
    Math.abs(adj.exposureEv) > 1e-4 ||
    adj.contrast !== 0 ||
    adj.highlights !== 0 ||
    adj.shadows !== 0 ||
    adj.whites !== 0 ||
    adj.blacks !== 0 ||
    adj.texture !== 0 ||
    adj.clarity !== 0 ||
    adj.dehaze !== 0 ||
    adj.saturation !== 0 ||
    adj.sharpness !== 0
  );
}

export function isMaskLayerEffective(layer: MaskLayer): boolean {
  return (
    layer.enabled &&
    layer.opacity > 0 &&
    layer.components.length > 0 &&
    isLocalAdjustmentsModified(layer.adjustments)
  );
}

export function createLinearGradientComponent(op: MaskCombineOp = 'add'): LinearGradientComponent {
  return {
    type: 'linear_gradient',
    id: createMaskId('lin'),
    op,
    inverted: false,
    startX: 0.5,
    startY: 0.12,
    endX: 0.5,
    endY: 0.62,
    feather: 55,
  };
}

export function createRadialGradientComponent(op: MaskCombineOp = 'add'): RadialGradientComponent {
  return {
    type: 'radial_gradient',
    id: createMaskId('rad'),
    op,
    inverted: false,
    centerX: 0.5,
    centerY: 0.45,
    radiusX: 0.28,
    radiusY: 0.32,
    rotationDeg: 0,
    feather: 50,
  };
}

export function createBrushComponent(op: MaskCombineOp = 'add'): BrushComponent {
  return {
    type: 'brush',
    id: createMaskId('br'),
    op,
    inverted: false,
    strokes: [],
  };
}

export function createLuminanceRangeComponent(op: MaskCombineOp = 'add'): LuminanceRangeComponent {
  return {
    type: 'luminance_range',
    id: createMaskId('lum'),
    op,
    inverted: false,
    minLuma: 0.45,
    maxLuma: 1.0,
    feather: 0.14,
  };
}

export function createColorRangeComponent(op: MaskCombineOp = 'add'): ColorRangeComponent {
  return {
    type: 'color_range',
    id: createMaskId('col'),
    op,
    inverted: false,
    targetHueDeg: 28,
    targetSat: 0.42,
    targetLuma: 0.58,
    tolerance: 0.26,
    feather: 0.35,
  };
}

export function createAiMatteComponent(
  result: GeneratedAiMaskResult,
  op: MaskCombineOp = 'add',
): AiMatteComponent {
  return {
    type: 'ai_matte',
    id: createMaskId('ai'),
    op,
    inverted: false,
    kind: result.kind,
    feather: 25,
    matteWidth: result.matteWidth,
    matteHeight: result.matteHeight,
    matteRleB64: result.matteRleB64,
  };
}

export function aiMaskKindLabel(kind: AiMaskKind): string {
  switch (kind) {
    case 'subject':
      return 'Subject';
    case 'background':
      return 'Background';
    case 'face_skin':
      return 'Face & Skin';
    case 'sky':
      return 'Sky';
  }
}

export function maskComponentLabel(comp: MaskComponent): string {
  switch (comp.type) {
    case 'linear_gradient':
      return 'Linear Gradient';
    case 'radial_gradient':
      return 'Radial Gradient';
    case 'brush':
      return `Brush (${comp.strokes.length})`;
    case 'luminance_range':
      return 'Luminance Range';
    case 'color_range':
      return 'Color Range';
    case 'ai_matte':
      return `AI ${aiMaskKindLabel(comp.kind)}`;
  }
}

export function createMaskLayer(name: string, component: MaskComponent): MaskLayer {
  return {
    id: createMaskId('layer'),
    name,
    enabled: true,
    inverted: false,
    opacity: 100,
    components: [component],
    adjustments: createDefaultLocalAdjustments(),
  };
}

export function hydrateMaskLayers(raw: readonly MaskLayer[] | undefined | null): readonly MaskLayer[] {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  return raw.map((layer, index) => ({
    id: typeof layer.id === 'string' && layer.id.length > 0 ? layer.id : `mask-${index + 1}`,
    name: typeof layer.name === 'string' && layer.name.trim().length > 0 ? layer.name : `Mask ${index + 1}`,
    enabled: layer.enabled ?? true,
    inverted: Boolean(layer.inverted),
    opacity: Math.max(0, Math.min(100, Math.round(layer.opacity ?? 100))),
    components: Array.isArray(layer.components) ? layer.components : [],
    adjustments: {
      ...createDefaultLocalAdjustments(),
      ...(layer.adjustments ?? {}),
    },
  }));
}

// ── PackBits RLE + Base64 Codec (matches Rust `internal/domain/src/masking.rs`) ──

const B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

export function encodeMatteRleB64(bytes: Uint8Array): string {
  const rle: number[] = [];
  const n = bytes.length;
  let i = 0;

  while (i < n) {
    let run = 1;
    while (i + run < n && run < 128 && bytes[i + run] === bytes[i]) {
      run += 1;
    }
    if (run >= 3) {
      rle.push(257 - run);
      rle.push(bytes[i]!);
      i += run;
    } else {
      const start = i;
      let lit = 0;
      while (i < n && lit < 128) {
        if (i + 2 < n && bytes[i] === bytes[i + 1] && bytes[i] === bytes[i + 2]) {
          break;
        }
        i += 1;
        lit += 1;
      }
      if (lit > 0) {
        rle.push(lit - 1);
        for (let k = start; k < start + lit; k++) {
          rle.push(bytes[k]!);
        }
      }
    }
  }

  let out = '';
  for (let j = 0; j < rle.length; j += 3) {
    const b0 = rle[j]!;
    const has1 = j + 1 < rle.length;
    const has2 = j + 2 < rle.length;
    const b1 = has1 ? rle[j + 1]! : 0;
    const b2 = has2 ? rle[j + 2]! : 0;
    const n24 = (b0 << 16) | (b1 << 8) | b2;
    out += B64_CHARS[(n24 >> 18) & 63];
    out += B64_CHARS[(n24 >> 12) & 63];
    out += has1 ? B64_CHARS[(n24 >> 6) & 63] : '=';
    out += has2 ? B64_CHARS[n24 & 63] : '=';
  }
  return out;
}

const B64_LOOKUP = (() => {
  const table = new Int16Array(128).fill(-1);
  for (let i = 0; i < B64_CHARS.length; i++) {
    table[B64_CHARS.charCodeAt(i)] = i;
  }
  return table;
})();

export function decodeMatteRleB64(encoded: string, expectedLen: number): Uint8Array | null {
  if (expectedLen === 0) return new Uint8Array(0);
  const clean = encoded.replace(/\s+/g, '');
  if (clean.length === 0 || clean.length % 4 !== 0) return null;

  const rleBytes: number[] = [];
  for (let i = 0; i < clean.length; i += 4) {
    const c0 = clean.charCodeAt(i);
    const c1 = clean.charCodeAt(i + 1);
    const c2 = clean.charCodeAt(i + 2);
    const c3 = clean.charCodeAt(i + 3);
    const a = c0 < 128 ? B64_LOOKUP[c0]! : -1;
    const b = c1 < 128 ? B64_LOOKUP[c1]! : -1;
    if (a < 0 || b < 0) return null;
    rleBytes.push((a << 2) | (b >> 4));
    if (clean[i + 2] !== '=') {
      const c = c2 < 128 ? B64_LOOKUP[c2]! : -1;
      if (c < 0) return null;
      rleBytes.push(((b << 4) | (c >> 2)) & 0xff);
      if (clean[i + 3] !== '=') {
        const d = c3 < 128 ? B64_LOOKUP[c3]! : -1;
        if (d < 0) return null;
        rleBytes.push(((c << 6) | d) & 0xff);
      }
    }
  }

  const out = new Uint8Array(expectedLen);
  let outPos = 0;
  let i = 0;
  while (i < rleBytes.length) {
    const header = rleBytes[i++]!;
    if (header === 128) continue;
    if (header < 128) {
      const count = header + 1;
      if (i + count > rleBytes.length || outPos + count > expectedLen) return null;
      for (let k = 0; k < count; k++) {
        out[outPos++] = rleBytes[i++]!;
      }
    } else {
      const count = 257 - header;
      if (i >= rleBytes.length || outPos + count > expectedLen) return null;
      const val = rleBytes[i++]!;
      out.fill(val, outPos, outPos + count);
      outPos += count;
    }
  }
  return outPos === expectedLen ? out : null;
}

// ── Spatial Mask Rasterizer + Per-Pixel Range Compiler for WebGL2 ─────────────

function smoothstep(edge0: number, edge1: number, x: number): number {
  const denom = Math.max(edge1 - edge0, 1e-6);
  const t = Math.max(0, Math.min(1, (x - edge0) / denom));
  return t * t * (3 - 2 * t);
}

export function evalLinearGradient(
  u: number,
  v: number,
  c: LinearGradientComponent,
): number {
  const dx = c.endX - c.startX;
  const dy = c.endY - c.startY;
  const len2 = Math.max(dx * dx + dy * dy, 1e-6);
  const t = ((u - c.startX) * dx + (v - c.startY) * dy) / len2;
  const halfSpan = 0.04 + (Math.max(0, Math.min(100, c.feather)) / 100) * 0.46;
  return 1 - smoothstep(0.5 - halfSpan, 0.5 + halfSpan, t);
}

export function evalRadialGradient(
  u: number,
  v: number,
  c: RadialGradientComponent,
): number {
  const px = u - c.centerX;
  const py = v - c.centerY;
  const rad = (c.rotationDeg * Math.PI) / 180;
  const sinR = Math.sin(rad);
  const cosR = Math.cos(rad);
  const rx = px * cosR + py * sinR;
  const ry = -px * sinR + py * cosR;
  const nx = rx / Math.max(c.radiusX, 0.005);
  const ny = ry / Math.max(c.radiusY, 0.005);
  const d = Math.hypot(nx, ny);
  const inner = Math.max(0, Math.min(0.99, 1 - Math.max(0, Math.min(100, c.feather)) / 100));
  return 1 - smoothstep(inner, 1, d);
}

function applyMatteFeather(m: number, feather: number): number {
  const f = Math.max(0, Math.min(100, feather));
  if (Math.abs(f - 25) < 0.5) return m;
  if (f < 25) {
    const k = 1 + ((25 - f) / 25) * 3;
    return Math.max(0, Math.min(1, (m - 0.5) * k + 0.5));
  }
  const t = (f - 25) / 75;
  const soft = smoothstep(0, 1, Math.pow(Math.max(0, Math.min(1, m)), 1 - 0.35 * t));
  return m + (soft - m) * t;
}

export interface CompiledMaskRangeSlot {
  /** 0 = unused, 1 = luminance_range, 2 = color_range */
  readonly mode: number;
  /** 0 = add, 1 = subtract, 2 = intersect */
  readonly op: number;
  readonly inverted: number;
  readonly p0: number;
  readonly p1: number;
  readonly p2: number;
  readonly p3: number;
  readonly p4: number;
}

export interface CompiledWebGlMaskData {
  readonly hasSpatial: boolean;
  readonly spatialWidth: number;
  readonly spatialHeight: number;
  /** Tightly packed RGBA bytes (mask alpha replicated in R,G,B,A) for universal WebGL1/2 upload. */
  readonly spatialRgba: Uint8Array;
  /** Up to 4 full-resolution range components evaluated per-pixel inside the GPU shader. */
  readonly ranges: readonly CompiledMaskRangeSlot[];
}

export function isRangeComponent(
  c: MaskComponent,
): c is LuminanceRangeComponent | ColorRangeComponent {
  return c.type === 'luminance_range' || c.type === 'color_range';
}

function encodeCombineOp(op: MaskCombineOp): number {
  if (op === 'subtract') return 1;
  if (op === 'intersect') return 2;
  return 0;
}

/**
 * Compiles a `MaskLayer`'s spatial components (`Linear`, `Radial`, `Brush`, `AiMatte`) into a
 * compact GPU-ready alpha texture (`<= 320px` max-dim, hardware bilinearly filtered), plus
 * uniform slots for full-resolution per-pixel `LuminanceRange` & `ColorRange` evaluation.
 */
export function compileMaskLayerForWebGl(
  layer: MaskLayer,
  orientedW: number,
  orientedH: number,
  maxSpatialDim = 320,
): CompiledWebGlMaskData {
  const spatialComps = layer.components.filter((c) => !isRangeComponent(c));
  const rangeComps = layer.components.filter(isRangeComponent).slice(0, 4);

  const ranges: CompiledMaskRangeSlot[] = [];
  for (let i = 0; i < 4; i++) {
    const r = rangeComps[i];
    if (!r) {
      ranges.push({ mode: 0, op: 0, inverted: 0, p0: 0, p1: 0, p2: 0, p3: 0, p4: 0 });
    } else if (r.type === 'luminance_range') {
      ranges.push({
        mode: 1,
        op: encodeCombineOp(r.op),
        inverted: r.inverted ? 1 : 0,
        p0: r.minLuma,
        p1: r.maxLuma,
        p2: r.feather,
        p3: 0,
        p4: 0,
      });
    } else {
      ranges.push({
        mode: 2,
        op: encodeCombineOp(r.op),
        inverted: r.inverted ? 1 : 0,
        p0: r.targetHueDeg,
        p1: r.targetSat,
        p2: r.targetLuma,
        p3: r.tolerance,
        p4: r.feather,
      });
    }
  }

  if (spatialComps.length === 0) {
    return {
      hasSpatial: false,
      spatialWidth: 1,
      spatialHeight: 1,
      spatialRgba: new Uint8Array([255, 255, 255, 255]),
      ranges,
    };
  }

  const safeW = Math.max(1, orientedW);
  const safeH = Math.max(1, orientedH);
  const scale = Math.min(1, maxSpatialDim / Math.max(safeW, safeH));
  const w = Math.max(16, Math.round(safeW * scale));
  const h = Math.max(16, Math.round(safeH * scale));
  const aspect = safeW / safeH;
  const sx = aspect >= 1 ? aspect : 1;
  const sy = aspect >= 1 ? 1 : 1 / aspect;

  // Pre-decode any AI mattes once before the pixel loop
  const decodedMattes = new Map<string, { mw: number; mh: number; data: Uint8Array }>();
  for (const c of spatialComps) {
    if (c.type === 'ai_matte' && c.matteWidth > 0 && c.matteHeight > 0 && c.matteRleB64) {
      const data = decodeMatteRleB64(c.matteRleB64, c.matteWidth * c.matteHeight);
      if (data) {
        decodedMattes.set(c.id, { mw: c.matteWidth, mh: c.matteHeight, data });
      }
    }
  }

  // Precompute stroke bounding boxes for fast brush culling
  const strokeBounds = new Map<
    string,
    { minU: number; maxU: number; minV: number; maxV: number; stroke: BrushStroke }[]
  >();
  for (const c of spatialComps) {
    if (c.type !== 'brush') continue;
    const boxes: { minU: number; maxU: number; minV: number; maxV: number; stroke: BrushStroke }[] =
      [];
    for (const s of c.strokes) {
      if (s.points.length === 0) continue;
      let minU = Infinity;
      let maxU = -Infinity;
      let minV = Infinity;
      let maxV = -Infinity;
      for (const [pu, pv] of s.points) {
        if (pu < minU) minU = pu;
        if (pu > maxU) maxU = pu;
        if (pv < minV) minV = pv;
        if (pv > maxV) maxV = pv;
      }
      const padU = s.size / sx + 0.01;
      const padV = s.size / sy + 0.01;
      boxes.push({
        minU: minU - padU,
        maxU: maxU + padU,
        minV: minV - padV,
        maxV: maxV + padV,
        stroke: s,
      });
    }
    strokeBounds.set(c.id, boxes);
  }

  const rgba = new Uint8Array(w * h * 4);
  const invW = 1 / w;
  const invH = 1 / h;

  for (let y = 0; y < h; y++) {
    const v = (y + 0.5) * invH;
    const pv = v * sy;
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) * invW;
      const pu = u * sx;
      let acc = 0;

      for (let ci = 0; ci < spatialComps.length; ci++) {
        const comp = spatialComps[ci]!;
        let val = 0;
        if (comp.type === 'linear_gradient') {
          val = evalLinearGradient(u, v, comp);
        } else if (comp.type === 'radial_gradient') {
          val = evalRadialGradient(u, v, comp);
        } else if (comp.type === 'brush') {
          const boxes = strokeBounds.get(comp.id);
          if (boxes && boxes.length > 0) {
            let brAcc = 0;
            for (let si = 0; si < boxes.length; si++) {
              const box = boxes[si]!;
              if (u < box.minU || u > box.maxU || v < box.minV || v > box.maxV) continue;
              const stroke = box.stroke;
              const radius = Math.max(0.004, stroke.size);
              const radius2 = radius * radius;
              let minD2 = Infinity;
              const pts = stroke.points;
              if (pts.length === 1) {
                const dx = pu - pts[0]![0] * sx;
                const dy = pv - pts[0]![1] * sy;
                minD2 = dx * dx + dy * dy;
              } else {
                for (let k = 0; k < pts.length - 1; k++) {
                  const ax = pts[k]![0] * sx;
                  const ay = pts[k]![1] * sy;
                  const bx = pts[k + 1]![0] * sx;
                  const by = pts[k + 1]![1] * sy;
                  const abx = bx - ax;
                  const aby = by - ay;
                  const ab2 = abx * abx + aby * aby;
                  const t =
                    ab2 > 1e-8
                      ? Math.max(0, Math.min(1, ((pu - ax) * abx + (pv - ay) * aby) / ab2))
                      : 0;
                  const dx = pu - (ax + abx * t);
                  const dy = pv - (ay + aby * t);
                  const d2 = dx * dx + dy * dy;
                  if (d2 < minD2) minD2 = d2;
                }
              }
              if (minD2 < radius2) {
                const dist = Math.sqrt(minD2);
                const inner = radius * Math.max(0, Math.min(0.98, 1 - stroke.feather));
                const dab = (1 - smoothstep(inner, radius, dist)) * Math.max(0.05, Math.min(1, stroke.flow));
                brAcc = Math.min(1, brAcc + dab * (1 - brAcc));
              }
            }
            val = brAcc;
          }
        } else if (comp.type === 'ai_matte') {
          const decoded = decodedMattes.get(comp.id);
          if (decoded) {
            const { mw, mh, data } = decoded;
            const fx = Math.max(0, Math.min(mw - 1, u * (mw - 1)));
            const fy = Math.max(0, Math.min(mh - 1, v * (mh - 1)));
            const x0 = Math.floor(fx);
            const y0 = Math.floor(fy);
            const x1 = Math.min(mw - 1, x0 + 1);
            const y1 = Math.min(mh - 1, y0 + 1);
            const tx = fx - x0;
            const ty = fy - y0;
            const c00 = data[y0 * mw + x0]! / 255;
            const c10 = data[y0 * mw + x1]! / 255;
            const c01 = data[y1 * mw + x0]! / 255;
            const c11 = data[y1 * mw + x1]! / 255;
            const rawM = c00 + (c10 - c00) * tx + (c01 + (c11 - c01) * tx - (c00 + (c10 - c00) * tx)) * ty;
            val = applyMatteFeather(rawM, comp.feather);
          }
        }

        if (comp.inverted) val = 1 - val;

        if (ci === 0) {
          acc = comp.op === 'subtract' ? Math.max(0, Math.min(1, 1 - val)) : Math.max(0, Math.min(1, val));
        } else if (comp.op === 'add') {
          acc = Math.min(1, acc + val);
        } else if (comp.op === 'subtract') {
          acc = Math.max(0, acc - val);
        } else {
          acc = Math.max(0, Math.min(1, acc * val));
        }
      }

      const byte = Math.round(Math.max(0, Math.min(1, acc)) * 255);
      const o = (y * w + x) * 4;
      rgba[o] = byte;
      rgba[o + 1] = byte;
      rgba[o + 2] = byte;
      rgba[o + 3] = 255;
    }
  }

  return { hasSpatial: true, spatialWidth: w, spatialHeight: h, spatialRgba: rgba, ranges };
}

/**
 * Generates a deterministic cache signature for a mask layer's spatial geometry so dragging
 * local adjustment sliders (Exposure, Shadows, Clarity, etc.) never re-rasterizes the mask.
 */
export function spatialMaskGeometrySignature(
  layer: MaskLayer,
  orientedW: number,
  orientedH: number,
): string {
  const spatial = layer.components.filter((c) => !isRangeComponent(c));
  if (spatial.length === 0) return 'none';
  return `${orientedW}x${orientedH}:${JSON.stringify(spatial)}`;
}
