/**
 * Pure domain types & zero-copy binary parsers for Kroma's high-bit-depth Studio Develop
 * working buffers:
 * - `KR32` (`RGBA32F` 32-bit IEEE 754 float, scene-referred Linear Rec.2020 for WebGPU)
 * - `KR16` (`RGBA16F` 16-bit IEEE 754 half-float for WebGL2 fallback)
 */

export const KR16_FLAG_LINEAR = 0x01;
export const KR16_FLAG_RAW_DEMOSAIC = 0x02;
export const KR_FLAG_WIDE_GAMUT_REC2020 = 0x04;

export type WorkingBufferPrecision = 'f32' | 'f16';
export type DevelopBufferLoadPriority = 'active' | 'prefetch';

export interface DevelopWorkingBuffer {
  readonly precision: WorkingBufferPrecision;
  readonly width: number;
  readonly height: number;
  /** `true` for unclipped scene-linear Camera RAW floats (`> 1.0` highlight headroom preserved). */
  readonly isLinear: boolean;
  /** `true` when linear RGB channels are encoded in wide-gamut Linear Rec.2020 (D65). */
  readonly isWideGamut: boolean;
  /** `true` when decoded directly from camera RAW sensor data via Bayer demosaicing. */
  readonly isRawDemosaic: boolean;
  readonly sourceBitDepth: number;
  /** Zero-copy view of packed IEEE 754 binary32 RGBA pixels (`width * height * 4` f32 values) for `KR32`. */
  readonly float32Rgba?: Float32Array;
  /** Zero-copy view of packed IEEE 754 binary16 RGBA pixels (`width * height * 4` u16 values) for `KR16`. */
  readonly halfFloatRgba?: Uint16Array;
}

const RAW_EXTENSIONS = new Set([
  '3fr',
  'ari',
  'arw',
  'bay',
  'cap',
  'cr2',
  'cr3',
  'crw',
  'dcr',
  'dcs',
  'dng',
  'drf',
  'eip',
  'erf',
  'fff',
  'iiq',
  'k25',
  'kdc',
  'mdc',
  'mef',
  'mos',
  'mrw',
  'nef',
  'nrw',
  'orf',
  'pef',
  'ptx',
  'pxn',
  'qtk',
  'raf',
  'raw',
  'rw2',
  'rwl',
  'rwz',
  'sr2',
  'srf',
  'srw',
  'sti',
  'x3f',
]);

const HIGH_BIT_DEPTH_CANDIDATE_EXTENSIONS = new Set([...RAW_EXTENSIONS, 'tif', 'tiff', 'png']);
const WEB_NATIVE_IMAGE_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'webp']);

function extractExtensionLower(fileName: string): string {
  const dot = fileName.lastIndexOf('.');
  if (dot < 0 || dot === fileName.length - 1) return '';
  return fileName.slice(dot + 1).toLowerCase();
}

export function isRawFileName(fileName: string): boolean {
  return RAW_EXTENSIONS.has(extractExtensionLower(fileName));
}

/**
 * Returns `true` when Chromium / WebView2 can decode the original file natively (JPEG, PNG, WebP)
 * so the Studio Develop GPU canvas can load full-resolution source bytes (downscaled via
 * `developBitmapCache`) while all other UI tabs use compact `<=1024px` cached preview proxies.
 */
export function isWebNativeFileName(fileName: string): boolean {
  return WEB_NATIVE_IMAGE_EXTENSIONS.has(extractExtensionLower(fileName));
}

/**
 * Returns `true` when the photo may carry >8-bit source data (Camera RAW, TIFF, or potentially
 * 16-bit PNG). Returns `false` for JPEG/WebP so 8-bit web-native shoots skip IPC probing entirely.
 */
export function shouldProbeHighBitDepthBuffer(fileName: string): boolean {
  return HIGH_BIT_DEPTH_CANDIDATE_EXTENSIONS.has(extractExtensionLower(fileName));
}

/**
 * Parses a binary `KR32` payload (`"KR32"` header + packed `rgba32float` pixels) returned by
 * Tauri IPC into a zero-copy `DevelopWorkingBuffer` ready for WebGPU `rgba32float` upload.
 */
export function parseKr32WorkingBuffer(raw: ArrayBuffer): DevelopWorkingBuffer | null {
  if (raw.byteLength < 16) {
    return null;
  }

  const view = new DataView(raw);
  // Magic bytes "KR32" (0x4B, 0x52, 0x33, 0x32)
  if (
    view.getUint8(0) !== 0x4b ||
    view.getUint8(1) !== 0x52 ||
    view.getUint8(2) !== 0x33 ||
    view.getUint8(3) !== 0x32
  ) {
    return null;
  }

  const width = view.getUint32(4, true);
  const height = view.getUint32(8, true);
  const flags = view.getUint8(12);
  const sourceBitDepth = view.getUint8(13) || 32;

  if (width === 0 || height === 0 || width > 16384 || height > 16384) {
    return null;
  }

  const expectedPixelBytes = width * height * 16;
  if (raw.byteLength !== 16 + expectedPixelBytes) {
    return null;
  }

  const float32Rgba = new Float32Array(raw, 16, width * height * 4);
  return {
    precision: 'f32',
    width,
    height,
    isLinear: (flags & KR16_FLAG_LINEAR) !== 0,
    isWideGamut: (flags & KR_FLAG_WIDE_GAMUT_REC2020) !== 0,
    isRawDemosaic: (flags & KR16_FLAG_RAW_DEMOSAIC) !== 0,
    sourceBitDepth,
    float32Rgba,
  };
}

/**
 * Parses a binary `KR16` payload returned by Tauri IPC (`tauri::ipc::Response`) into a zero-copy
 * `DevelopWorkingBuffer` ready for WebGL2 `gl.RGBA16F` / `gl.HALF_FLOAT` texture upload.
 * Returns `null` for empty responses (e.g. 8-bit PNG fallback) or malformed payloads.
 */
export function parseKr16WorkingBuffer(raw: ArrayBuffer): DevelopWorkingBuffer | null {
  if (raw.byteLength < 16) {
    return null;
  }

  const view = new DataView(raw);
  // Magic bytes "KR16" (0x4B, 0x52, 0x31, 0x36)
  if (
    view.getUint8(0) !== 0x4b ||
    view.getUint8(1) !== 0x52 ||
    view.getUint8(2) !== 0x31 ||
    view.getUint8(3) !== 0x36
  ) {
    return null;
  }

  const width = view.getUint32(4, true);
  const height = view.getUint32(8, true);
  const flags = view.getUint8(12);
  const sourceBitDepth = view.getUint8(13) || 16;

  if (width === 0 || height === 0 || width > 16384 || height > 16384) {
    return null;
  }

  const expectedPixelBytes = width * height * 8;
  if (raw.byteLength !== 16 + expectedPixelBytes) {
    return null;
  }

  const halfFloatRgba = new Uint16Array(raw, 16, width * height * 4);
  return {
    precision: 'f16',
    width,
    height,
    isLinear: (flags & KR16_FLAG_LINEAR) !== 0,
    isWideGamut: (flags & KR_FLAG_WIDE_GAMUT_REC2020) !== 0,
    isRawDemosaic: (flags & KR16_FLAG_RAW_DEMOSAIC) !== 0,
    sourceBitDepth,
    halfFloatRgba,
  };
}

/**
 * Detects Kroma's 4-byte `"KRSK"` sentinel returned by the Rust RAW decode scheduler when an
 * older or background prefetch decode was superseded by a newer active photo selection.
 * Callers must treat this as a cancelled load and NEVER cache `null` in the LRU.
 */
export function isSupersededWorkingBuffer(raw: ArrayBuffer): boolean {
  if (raw.byteLength !== 4) {
    return false;
  }
  const view = new DataView(raw);
  // Magic bytes "KRSK" (0x4B, 0x52, 0x53, 0x4B)
  return (
    view.getUint8(0) === 0x4b &&
    view.getUint8(1) === 0x52 &&
    view.getUint8(2) === 0x53 &&
    view.getUint8(3) === 0x4b
  );
}

/**
 * Auto-detecting zero-copy parser supporting both `KR32` (32-bit float WebGPU)
 * and `KR16` (16-bit half-float WebGL2) binary payloads.
 */
export function parseDevelopWorkingBuffer(raw: ArrayBuffer): DevelopWorkingBuffer | null {
  return parseKr32WorkingBuffer(raw) ?? parseKr16WorkingBuffer(raw);
}
