export interface SplashTiltVector {
  /** Pitch in degrees (+ tilts top edge toward viewer when cursor is above center). */
  readonly rotateX: number;
  /** Yaw in degrees (+ tilts right edge toward viewer when cursor is right of center). */
  readonly rotateY: number;
  /** Subtle magnetic translation along X in px. */
  readonly shiftX: number;
  /** Subtle magnetic translation along Y in px. */
  readonly shiftY: number;
  /** Dynamic specular shine horizontal origin in [0..100]%. */
  readonly glareX: number;
  /** Dynamic specular shine vertical origin in [0..100]%. */
  readonly glareY: number;
}

export const NEUTRAL_SPLASH_TILT: SplashTiltVector = {
  rotateX: 0,
  rotateY: 0,
  shiftX: 0,
  shiftY: 0,
  glareX: 50,
  glareY: 32,
};

export const MAX_SPLASH_TILT_DEG = 24;
const MAX_SHIFT_PX = 9;
const RESPONSE_RADIUS_PX = 220;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Maps pointer position relative to the logo center into responsive, bounded 3D physics tilt
 * and specular highlight coordinates using a smooth `tanh` falloff curve.
 */
export function computeCursorTilt(
  pointerX: number,
  pointerY: number,
  centerX: number,
  centerY: number,
): SplashTiltVector {
  if (
    !Number.isFinite(pointerX) ||
    !Number.isFinite(pointerY) ||
    !Number.isFinite(centerX) ||
    !Number.isFinite(centerY)
  ) {
    return NEUTRAL_SPLASH_TILT;
  }

  const nx = Math.tanh((pointerX - centerX) / RESPONSE_RADIUS_PX);
  const ny = Math.tanh((pointerY - centerY) / RESPONSE_RADIUS_PX);

  return {
    rotateX: -ny * MAX_SPLASH_TILT_DEG,
    rotateY: nx * MAX_SPLASH_TILT_DEG,
    shiftX: nx * MAX_SHIFT_PX,
    shiftY: ny * MAX_SHIFT_PX,
    glareX: clamp(50 + nx * 44, 6, 94),
    glareY: clamp(50 + ny * 44, 6, 94),
  };
}
