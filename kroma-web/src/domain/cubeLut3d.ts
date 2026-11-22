import { invoke } from '@tauri-apps/api/core';
import { isTauriDesktopEnvironment } from '../adapters/gateways/tauriStudioGateway';

export interface ParsedCubeLut3D {
  readonly size: number;
  readonly domainMin: readonly [number, number, number];
  readonly domainMax: readonly [number, number, number];
  /** Flattened `vec4<f32>` table (`size * size * size * 4` floats `[r, g, b, 1.0]`) for WebGPU `storage` binding 4. */
  readonly rgbaTable: Float32Array;
}

const parsedLutCache = new Map<string, ParsedCubeLut3D>();
const inFlightLutLoads = new Map<string, Promise<ParsedCubeLut3D | null>>();

function parseRgbTriplet(line: string, context: string): [number, number, number] {
  const parts = line.trim().split(/\s+/);
  if (parts.length < 3) {
    throw new Error(`Missing RGB triplet in ${context}`);
  }
  const r = Number.parseFloat(parts[0]!);
  const g = Number.parseFloat(parts[1]!);
  const b = Number.parseFloat(parts[2]!);
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) {
    throw new Error(`Invalid RGB float values in ${context}`);
  }
  return [r, g, b];
}

/**
 * Parses a standard Adobe `.cube` 3D Lookup Table file into a GPU-ready `Float32Array` (`vec4<f32>`).
 * Matches `internal/infrastructure/src/develop/cube_lut.rs::CubeLut3D::parse` 1:1.
 */
export function parseCubeLut3D(content: string): ParsedCubeLut3D {
  let size: number | null = null;
  let domainMin: [number, number, number] = [0, 0, 0];
  let domainMax: [number, number, number] = [1, 1, 1];
  const triplets: number[] = [];

  for (const rawLine of content.split(/\r?\n/)) {
    const line = (rawLine.split('#')[0] ?? '').trim();
    if (!line) continue;
    if (line.startsWith('TITLE')) continue;

    if (line.startsWith('LUT_3D_SIZE')) {
      const rest = line.slice('LUT_3D_SIZE'.length).trim();
      const parsed = Number.parseInt(rest, 10);
      if (!Number.isInteger(parsed) || parsed < 2 || parsed > 129) {
        throw new Error(`Unsupported LUT_3D_SIZE "${rest}" (expected 2..129)`);
      }
      size = parsed;
      continue;
    }

    if (line.startsWith('DOMAIN_MIN')) {
      domainMin = parseRgbTriplet(line.slice('DOMAIN_MIN'.length), 'DOMAIN_MIN');
      continue;
    }

    if (line.startsWith('DOMAIN_MAX')) {
      domainMax = parseRgbTriplet(line.slice('DOMAIN_MAX'.length), 'DOMAIN_MAX');
      continue;
    }

    if (line.startsWith('LUT_1D_SIZE')) {
      throw new Error('1D .cube LUTs are not supported; please select a 3D .cube LUT');
    }

    if (/^[A-Za-z_]/.test(line)) {
      continue;
    }

    const [r, g, b] = parseRgbTriplet(line, 'LUT entry');
    triplets.push(r, g, b, 1.0);
  }

  if (size === null) {
    throw new Error('Missing LUT_3D_SIZE header in .cube file');
  }

  const expectedFloats = size * size * size * 4;
  if (triplets.length !== expectedFloats) {
    throw new Error(
      `Expected ${size * size * size} RGB triplets for ${size}³ LUT, found ${triplets.length / 4}`,
    );
  }

  return {
    size,
    domainMin,
    domainMax,
    rgbaTable: new Float32Array(triplets),
  };
}

/**
 * Generates Kroma's 4 built-in 17×17×17 3D `.cube` film/studio profiles deterministically,
 * matching `CubeLut3D::builtin_by_name` in `internal/infrastructure/src/develop/cube_lut.rs` 1:1.
 */
export function buildBuiltinCubeLut3D(nameOrPath: string): ParsedCubeLut3D | null {
  const lower = nameOrPath.trim().toLowerCase();
  const kind = lower.includes('kodak_portra_400')
    ? 1
    : lower.includes('fuji_pro_400h')
      ? 2
      : lower.includes('kroma_studio_neutral')
        ? 3
        : lower.includes('ilford_hp5_mono')
          ? 4
          : 0;
  if (kind === 0) return null;

  const size = 17;
  const maxIdx = size - 1;
  const rgbaTable = new Float32Array(size * size * size * 4);
  let ptr = 0;

  for (let bIdx = 0; bIdx < size; bIdx++) {
    const b = bIdx / maxIdx;
    for (let gIdx = 0; gIdx < size; gIdx++) {
      const g = gIdx / maxIdx;
      for (let rIdx = 0; rIdx < size; rIdx++) {
        const r = rIdx / maxIdx;
        const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let mr = r;
        let mg = g;
        let mb = b;

        if (kind === 1) {
          const lift = (x: number) => {
            const s = x * x * (3 - 2 * x);
            return (x * 0.35 + s * 0.65) * 0.96 + 0.025;
          };
          mr = Math.min(1, Math.max(0, lift(r) * 1.035 + 0.012 * (1 - luma)));
          mg = Math.min(1, Math.max(0, lift(g) * 1.005 + 0.004 * luma));
          mb = Math.min(1, Math.max(0, lift(b) * 0.945 + 0.015 * (1 - luma)));
        } else if (kind === 2) {
          const lift = (x: number) => {
            const s = x * x * (3 - 2 * x);
            return (x * 0.42 + s * 0.58) * 0.97 + 0.02;
          };
          const shd = (1 - luma) * (1 - luma);
          mr = Math.min(1, Math.max(0, lift(r) * 0.985 + 0.008 * luma));
          mg = Math.min(1, Math.max(0, lift(g) * 1.018 + 0.012 * shd));
          mb = Math.min(1, Math.max(0, lift(b) * 1.015 + 0.018 * shd));
        } else if (kind === 3) {
          const sCurve = (x: number) => {
            const s = x * x * (3 - 2 * x);
            return x * 0.45 + s * 0.55;
          };
          const sr = sCurve(r);
          const sg = sCurve(g);
          const sb = sCurve(b);
          const sl = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb;
          mr = Math.min(1, Math.max(0, sl + (sr - sl) * 1.06));
          mg = Math.min(1, Math.max(0, sl + (sg - sl) * 1.06));
          mb = Math.min(1, Math.max(0, sl + (sb - sl) * 1.06));
        } else {
          const pan = Math.min(1, Math.max(0, 0.26 * r + 0.64 * g + 0.1 * b));
          const s = pan * pan * (3 - 2 * pan);
          const film = (pan * 0.28 + s * 0.72) * 0.95 + 0.025;
          mr = film;
          mg = film;
          mb = film;
        }

        rgbaTable[ptr++] = mr;
        rgbaTable[ptr++] = mg;
        rgbaTable[ptr++] = mb;
        rgbaTable[ptr++] = 1.0;
      }
    }
  }

  return {
    size,
    domainMin: [0, 0, 0],
    domainMax: [1, 1, 1],
    rgbaTable,
  };
}

export function cacheParsedCubeLut3D(cubePath: string, parsed: ParsedCubeLut3D): void {
  parsedLutCache.set(cubePath, parsed);
}

export function getCachedCubeLut3D(cubePath: string): ParsedCubeLut3D | null {
  const existing = parsedLutCache.get(cubePath);
  if (existing) return existing;
  const builtin = buildBuiltinCubeLut3D(cubePath);
  if (builtin) {
    parsedLutCache.set(cubePath, builtin);
    return builtin;
  }
  return null;
}

export async function ensureCubeLut3DLoaded(cubePath: string): Promise<ParsedCubeLut3D | null> {
  const sync = getCachedCubeLut3D(cubePath);
  if (sync) return sync;
  if (!isTauriDesktopEnvironment()) return null;

  const inFlight = inFlightLutLoads.get(cubePath);
  if (inFlight) return inFlight;

  const req = invoke<string>('read_cube_lut_file', { cubePath })
    .then((content) => {
      const parsed = parseCubeLut3D(content);
      parsedLutCache.set(cubePath, parsed);
      return parsed;
    })
    .catch(() => null)
    .finally(() => {
      inFlightLutLoads.delete(cubePath);
    });

  inFlightLutLoads.set(cubePath, req);
  return req;
}
