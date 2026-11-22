import type { Transition } from 'framer-motion';

/**
 * Optical darkroom motion tokens tuned for high-volume photographic workflows:
 * zero sluggish bounce, instantaneous hardware-detent feedback, GPU-composited only.
 */
export const OPTICAL_EASE = [0.22, 1, 0.36, 1] as const;

export const TACTILE_SPRING: Transition = {
  type: 'spring',
  stiffness: 520,
  damping: 38,
  mass: 0.65,
};

export const SOFT_SPRING: Transition = {
  type: 'spring',
  stiffness: 380,
  damping: 34,
  mass: 0.8,
};

export const PANEL_TRANSITION: Transition = {
  duration: 0.18,
  ease: OPTICAL_EASE,
};

export const FAST_FADE: Transition = {
  duration: 0.14,
  ease: OPTICAL_EASE,
};
