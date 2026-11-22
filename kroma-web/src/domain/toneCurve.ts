import type { ParametricToneCurve } from './develop';

export type ToneCurveChannel = 'rgb' | 'red' | 'green' | 'blue';

export interface ToneCurvePoint {
  readonly x: number; // Normalized input luminance [0.0, 1.0]
  readonly y: number; // Normalized output luminance [0.0, 1.0]
}

export const DEFAULT_TONE_CURVE_POINTS: readonly ToneCurvePoint[] = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
];

export const MAX_TONE_CURVE_POINTS = 12;
export const MIN_POINT_X_GAP = 0.025;

export interface ToneCurveShapePreset {
  readonly id: string;
  readonly label: string;
  readonly points: readonly ToneCurvePoint[];
}

export const TONE_CURVE_SHAPES: readonly ToneCurveShapePreset[] = [
  {
    id: 'linear',
    label: 'Linear',
    points: DEFAULT_TONE_CURVE_POINTS,
  },
  {
    id: 's-curve',
    label: 'S-Curve',
    points: [
      { x: 0, y: 0 },
      { x: 0.25, y: 0.2 },
      { x: 0.5, y: 0.5 },
      { x: 0.75, y: 0.81 },
      { x: 1, y: 1 },
    ],
  },
  {
    id: 'matte',
    label: 'Film Matte',
    points: [
      { x: 0, y: 0.085 },
      { x: 0.22, y: 0.21 },
      { x: 0.55, y: 0.56 },
      { x: 0.85, y: 0.87 },
      { x: 1, y: 0.96 },
    ],
  },
];

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function roundPoint(value: number): number {
  return Math.round(clamp01(value) * 1000) / 1000;
}

export function normalizeToneCurvePoints(
  rawPoints: readonly ToneCurvePoint[] | undefined,
): readonly ToneCurvePoint[] {
  if (!rawPoints || rawPoints.length < 2) {
    return DEFAULT_TONE_CURVE_POINTS;
  }
  const cleaned = rawPoints
    .map((p) => ({ x: roundPoint(p.x), y: roundPoint(p.y) }))
    .sort((a, b) => a.x - b.x)
    .slice(0, MAX_TONE_CURVE_POINTS);

  const deduped: ToneCurvePoint[] = [];
  for (const pt of cleaned) {
    const prev = deduped[deduped.length - 1];
    if (!prev || pt.x - prev.x >= 0.005) {
      deduped.push(pt);
    }
  }
  return deduped.length >= 2 ? deduped : DEFAULT_TONE_CURVE_POINTS;
}

export function getEffectiveCurvePoints(
  curve: ParametricToneCurve,
  channel: ToneCurveChannel = 'rgb',
): readonly ToneCurvePoint[] {
  if (channel === 'red') return normalizeToneCurvePoints(curve.redPoints);
  if (channel === 'green') return normalizeToneCurvePoints(curve.greenPoints);
  if (channel === 'blue') return normalizeToneCurvePoints(curve.bluePoints);
  return normalizeToneCurvePoints(curve.points);
}

export function setCurveChannelPoints(
  curve: ParametricToneCurve,
  channel: ToneCurveChannel,
  nextPoints: readonly ToneCurvePoint[] | undefined,
): ParametricToneCurve {
  const stored = hasCustomCurvePoints(nextPoints) ? nextPoints : undefined;
  if (channel === 'red') return { ...curve, redPoints: stored };
  if (channel === 'green') return { ...curve, greenPoints: stored };
  if (channel === 'blue') return { ...curve, bluePoints: stored };
  return { ...curve, points: stored };
}

export function hasCustomCurvePoints(points: readonly ToneCurvePoint[] | undefined): boolean {
  if (!points || points.length < 2) return false;
  if (points.length === 2) {
    const [p0, p1] = points;
    return (
      Math.abs(p0.x) > 1e-4 ||
      Math.abs(p0.y) > 1e-4 ||
      Math.abs(p1.x - 1) > 1e-4 ||
      Math.abs(p1.y - 1) > 1e-4
    );
  }
  return true;
}

export function isToneCurveChannelModified(
  curve: ParametricToneCurve,
  channel: ToneCurveChannel,
): boolean {
  if (channel === 'red') return hasCustomCurvePoints(curve.redPoints);
  if (channel === 'green') return hasCustomCurvePoints(curve.greenPoints);
  if (channel === 'blue') return hasCustomCurvePoints(curve.bluePoints);
  return (
    curve.highlights !== 0 ||
    curve.lights !== 0 ||
    curve.darks !== 0 ||
    curve.shadows !== 0 ||
    hasCustomCurvePoints(curve.points)
  );
}

export function isToneCurveModified(curve: ParametricToneCurve): boolean {
  return (
    isToneCurveChannelModified(curve, 'rgb') ||
    hasCustomCurvePoints(curve.redPoints) ||
    hasCustomCurvePoints(curve.greenPoints) ||
    hasCustomCurvePoints(curve.bluePoints)
  );
}

export function insertToneCurvePoint(
  currentPoints: readonly ToneCurvePoint[],
  x: number,
  y: number,
): { readonly points: readonly ToneCurvePoint[]; readonly index: number } {
  const pts = [...normalizeToneCurvePoints(currentPoints)];
  const clampedX = roundPoint(x);
  const clampedY = roundPoint(y);

  // If clicking very close horizontally to an existing point, grab that point instead
  const nearIdx = pts.findIndex((p) => Math.abs(p.x - clampedX) < MIN_POINT_X_GAP);
  if (nearIdx !== -1) {
    const updated = moveToneCurvePoint(pts, nearIdx, clampedX, clampedY);
    return { points: updated, index: nearIdx };
  }

  if (pts.length >= MAX_TONE_CURVE_POINTS) {
    // Cap reached — select nearest existing point
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const d = Math.hypot(pts[i].x - clampedX, pts[i].y - clampedY);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    return { points: pts, index: bestIdx };
  }

  let insertIdx = pts.findIndex((p) => p.x > clampedX);
  if (insertIdx === -1) insertIdx = pts.length;

  pts.splice(insertIdx, 0, { x: clampedX, y: clampedY });
  return { points: pts, index: insertIdx };
}

export function moveToneCurvePoint(
  currentPoints: readonly ToneCurvePoint[],
  index: number,
  nextX: number,
  nextY: number,
): readonly ToneCurvePoint[] {
  const pts = [...normalizeToneCurvePoints(currentPoints)];
  if (index < 0 || index >= pts.length) return pts;

  const minX = index === 0 ? 0 : pts[index - 1].x + MIN_POINT_X_GAP;
  const maxX = index === pts.length - 1 ? 1 : pts[index + 1].x - MIN_POINT_X_GAP;
  const clampedX = roundPoint(Math.min(maxX, Math.max(minX, nextX)));
  const clampedY = roundPoint(nextY);

  pts[index] = { x: clampedX, y: clampedY };
  return pts;
}

export function removeToneCurvePoint(
  currentPoints: readonly ToneCurvePoint[],
  index: number,
): readonly ToneCurvePoint[] {
  const pts = [...normalizeToneCurvePoints(currentPoints)];
  if (index < 0 || index >= pts.length) return pts;

  // Reset endpoints to their linear corners instead of deleting them
  if (index === 0) {
    pts[0] = { x: 0, y: 0 };
    return pts;
  }
  if (index === pts.length - 1) {
    pts[index] = { x: 1, y: 1 };
    return pts;
  }

  pts.splice(index, 1);
  return pts;
}

/**
 * Evaluates a Fritsch-Carlson monotone cubic Hermite spline (identical math to Rust `color_math.rs`).
 * Guarantees C1 smoothness, exact interpolation through every control point, and zero overshoot.
 */
export function evaluateMonotoneCubicSpline(
  rawPoints: readonly ToneCurvePoint[],
  xInput: number,
): number {
  const x = clamp01(xInput);
  const pts = normalizeToneCurvePoints(rawPoints);
  const n = pts.length;

  if (x <= pts[0].x) return pts[0].y;
  if (x >= pts[n - 1].x) return pts[n - 1].y;

  const h: number[] = new Array(n - 1);
  const delta: number[] = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    const dx = Math.max(1e-5, pts[i + 1].x - pts[i].x);
    h[i] = dx;
    delta[i] = (pts[i + 1].y - pts[i].y) / dx;
  }

  const m: number[] = new Array(n).fill(0);
  m[0] = delta[0];
  m[n - 1] = delta[n - 2];
  for (let i = 1; i < n - 1; i++) {
    m[i] = delta[i - 1] * delta[i] <= 0 ? 0 : 0.5 * (delta[i - 1] + delta[i]);
  }

  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(delta[i]) < 1e-6) {
      m[i] = 0;
      m[i + 1] = 0;
    } else {
      const alpha = m[i] / delta[i];
      const beta = m[i + 1] / delta[i];
      const sq = alpha * alpha + beta * beta;
      if (sq > 9) {
        const tau = 3 / Math.sqrt(sq);
        m[i] = tau * alpha * delta[i];
        m[i + 1] = tau * beta * delta[i];
      }
    }
  }

  let seg = n - 2;
  for (let i = 0; i < n - 1; i++) {
    if (x <= pts[i + 1].x) {
      seg = i;
      break;
    }
  }

  const t = clamp01((x - pts[seg].x) / h[seg]);
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;

  return clamp01(
    h00 * pts[seg].y + h10 * h[seg] * m[seg] + h01 * pts[seg + 1].y + h11 * h[seg] * m[seg + 1],
  );
}

function gaussian(x: number, mean: number, sigma: number): number {
  const z = (x - mean) / sigma;
  return Math.exp(-0.5 * z * z);
}

/**
 * Evaluates the combined Point Curve + Parametric Region Curve at luminance `x` in `[0, 1]`.
 * Matches `build_tone_curve_lut` in `internal/infrastructure/src/develop/color_math.rs`.
 */
export function evaluateCombinedToneCurve(curve: ParametricToneCurve, xInput: number): number {
  const x = clamp01(xInput);
  let y = hasCustomCurvePoints(curve.points)
    ? evaluateMonotoneCubicSpline(curve.points!, x)
    : x;

  if (curve.shadows !== 0) {
    y += (curve.shadows / 100) * 0.16 * gaussian(x, 0.14, 0.14);
  }
  if (curve.darks !== 0) {
    y += (curve.darks / 100) * 0.16 * gaussian(x, 0.36, 0.18);
  }
  if (curve.lights !== 0) {
    y += (curve.lights / 100) * 0.16 * gaussian(x, 0.64, 0.18);
  }
  if (curve.highlights !== 0) {
    y += (curve.highlights / 100) * 0.16 * gaussian(x, 0.86, 0.14);
  }
  return clamp01(y);
}

/**
 * Evaluates a single channel's point curve (`red`, `green`, `blue`) or the combined Master
 * Point + Parametric curve (`rgb`). Matches Rust `build_rgb_tone_curve_luts`.
 */
export function evaluateChannelToneCurve(
  curve: ParametricToneCurve,
  xInput: number,
  channel: ToneCurveChannel = 'rgb',
): number {
  if (channel === 'rgb') {
    return evaluateCombinedToneCurve(curve, xInput);
  }
  const pts =
    channel === 'red' ? curve.redPoints : channel === 'green' ? curve.greenPoints : curve.bluePoints;
  return hasCustomCurvePoints(pts) ? evaluateMonotoneCubicSpline(pts!, xInput) : clamp01(xInput);
}

/**
 * Evaluates the composed Master + Per-Channel tone curve at normalized value `xInput`.
 */
export function evaluateComposedChannelToneCurve(
  curve: ParametricToneCurve,
  xInput: number,
  channel: 'red' | 'green' | 'blue',
): number {
  const master = evaluateCombinedToneCurve(curve, xInput);
  return evaluateChannelToneCurve(curve, master, channel);
}

/**
 * Samples the tone curve into a space-delimited SVG `<feFunc* type="table">` LUT string.
 */
export function sampleToneCurveSvgTableValues(
  curve: ParametricToneCurve,
  steps = 33,
  channel: ToneCurveChannel = 'rgb',
): string {
  const count = Math.max(2, steps);
  const values: string[] = new Array(count);
  for (let i = 0; i < count; i++) {
    const x = i / (count - 1);
    const y =
      channel === 'rgb'
        ? evaluateCombinedToneCurve(curve, x)
        : evaluateComposedChannelToneCurve(curve, x, channel);
    values[i] = y.toFixed(4);
  }
  return values.join(' ');
}

export function buildToneCurveSvgPaths(
  curve: ParametricToneCurve,
  width: number,
  height: number,
  samples = 72,
  channel: ToneCurveChannel = 'rgb',
): { readonly linePath: string; readonly areaPath: string } {
  const count = Math.max(2, samples);
  const coords: string[] = new Array(count);
  for (let i = 0; i < count; i++) {
    const xNorm = i / (count - 1);
    const yNorm = evaluateChannelToneCurve(curve, xNorm, channel);
    const sx = (xNorm * width).toFixed(2);
    const sy = ((1 - yNorm) * height).toFixed(2);
    coords[i] = `${i === 0 ? 'M' : 'L'} ${sx},${sy}`;
  }
  const linePath = coords.join(' ');
  const areaPath = `${linePath} L ${width.toFixed(2)},${height.toFixed(2)} L 0,${height.toFixed(2)} Z`;
  return { linePath, areaPath };
}
