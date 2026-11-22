import {
  AnimatePresence,
  motion,
  useMotionTemplate,
  useMotionValue,
  useSpring,
  useTransform,
} from 'framer-motion';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import kromaLogo from '../../assets/kroma-logo.png';
import { formatByteSize } from '../../domain/exportEstimate';
import { DarkroomButton, FAST_FADE } from '../darkroom-kit';
import { AppUpdateModal } from './AppUpdateModal';
import { computeCursorTilt, NEUTRAL_SPLASH_TILT } from './splashTilt';

const MIN_DISPLAY_MS = 2850;
const COMPLETE_HOLD_MS = 340;

// Subtle, lively physics springs (gentle overshoot bounce without feeling cartoonish)
const TILT_SPRING = { stiffness: 240, damping: 14, mass: 0.85 } as const;
const MAGNETIC_SPRING = { stiffness: 200, damping: 16, mass: 0.85 } as const;
const GLARE_SPRING = { stiffness: 180, damping: 20, mass: 0.7 } as const;

function formatTransferSpeed(bytesPerSecond: number): string | null {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond <= 0) return null;
  return `${formatByteSize(bytesPerSecond)}/s`;
}

export function SplashScreen() {
  const {
    isBootReady,
    bootPhase,
    modelManifestStatus,
    modelDownloadProgress,
    modelDownloadError,
    startupUpdate,
    runtimeInfo,
    retryModelDownload,
    continueWithoutModels,
    dismissStartupUpdate,
  } = useStudio();
  const [visible, setVisible] = useState(true);
  const [minWindowElapsed, setMinWindowElapsed] = useState(false);
  const [timedProgress, setTimedProgress] = useState(12);
  const [stageIndex, setStageIndex] = useState(0);
  const logoRef = useRef<HTMLDivElement>(null);

  // Seamlessly remove the static pre-React HTML boot splash on the exact first painted React frame
  // and guarantee the Tauri window is revealed.
  useLayoutEffect(() => {
    const bootSplash = document.getElementById('boot-splash');
    if (bootSplash) {
      bootSplash.remove();
    }
    const tauriInternals = (
      window as Window & {
        __TAURI_INTERNALS__?: { invoke?: (cmd: string) => Promise<unknown> };
      }
    ).__TAURI_INTERNALS__;
    void tauriInternals?.invoke?.('reveal_main_window')?.catch(() => {});
  }, []);

  // Physics-driven cursor tilt & specular reflection values
  const rawRotateX = useMotionValue(NEUTRAL_SPLASH_TILT.rotateX);
  const rawRotateY = useMotionValue(NEUTRAL_SPLASH_TILT.rotateY);
  const rawShiftX = useMotionValue(NEUTRAL_SPLASH_TILT.shiftX);
  const rawShiftY = useMotionValue(NEUTRAL_SPLASH_TILT.shiftY);
  const rawGlareX = useMotionValue(NEUTRAL_SPLASH_TILT.glareX);
  const rawGlareY = useMotionValue(NEUTRAL_SPLASH_TILT.glareY);

  const rotateX = useSpring(rawRotateX, TILT_SPRING);
  const rotateY = useSpring(rawRotateY, TILT_SPRING);
  const shiftX = useSpring(rawShiftX, MAGNETIC_SPRING);
  const shiftY = useSpring(rawShiftY, MAGNETIC_SPRING);
  const glareX = useSpring(rawGlareX, GLARE_SPRING);
  const glareY = useSpring(rawGlareY, GLARE_SPRING);

  // Subtle inner artwork parallax inside the 3D glass tile
  const logoParallaxX = useTransform(shiftX, (v) => v * 0.4);
  const logoParallaxY = useTransform(shiftY, (v) => v * 0.4);

  // Dynamic drop shadow shifting opposite the tilt angle for true physical elevation
  const shadowOffsetX = useTransform(rotateY, (deg) => -deg * 0.75);
  const shadowOffsetY = useTransform(rotateX, (deg) => 22 + deg * 0.65);
  const dynamicTileShadow = useMotionTemplate`${shadowOffsetX}px ${shadowOffsetY}px 48px rgba(0, 0, 0, 0.72), 0 0 0 1px rgba(245, 158, 11, 0.14)`;

  const specularOverlay = useMotionTemplate`radial-gradient(circle 100px at ${glareX}% ${glareY}%, rgba(255, 255, 255, 0.30), rgba(245, 158, 11, 0.10) 42%, transparent 72%)`;
  const rimHighlight = useMotionTemplate`radial-gradient(circle 130px at ${glareX}% ${glareY}%, rgba(255, 193, 116, 0.42), transparent 68%)`;
  const ambientGlowX = useTransform(shiftX, (v) => v * 3.2);
  const ambientGlowY = useTransform(shiftY, (v) => v * 3.2);

  // Listen globally on window so 3D cursor tilt reacts immediately across the full viewport
  useEffect(() => {
    if (!visible) return;

    const resetTilt = () => {
      rawRotateX.set(NEUTRAL_SPLASH_TILT.rotateX);
      rawRotateY.set(NEUTRAL_SPLASH_TILT.rotateY);
      rawShiftX.set(NEUTRAL_SPLASH_TILT.shiftX);
      rawShiftY.set(NEUTRAL_SPLASH_TILT.shiftY);
      rawGlareX.set(NEUTRAL_SPLASH_TILT.glareX);
      rawGlareY.set(NEUTRAL_SPLASH_TILT.glareY);
    };

    const onPointerMove = (event: PointerEvent) => {
      const rect = logoRef.current?.getBoundingClientRect();
      const centerX = rect ? rect.left + rect.width * 0.5 : window.innerWidth * 0.5;
      const centerY = rect ? rect.top + rect.height * 0.5 : window.innerHeight * 0.5;
      const tilt = computeCursorTilt(event.clientX, event.clientY, centerX, centerY);
      rawRotateX.set(tilt.rotateX);
      rawRotateY.set(tilt.rotateY);
      rawShiftX.set(tilt.shiftX);
      rawShiftY.set(tilt.shiftY);
      rawGlareX.set(tilt.glareX);
      rawGlareY.set(tilt.glareY);
    };

    window.addEventListener('pointermove', onPointerMove, { passive: true });
    document.documentElement.addEventListener('mouseleave', resetTilt);
    window.addEventListener('blur', resetTilt);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      document.documentElement.removeEventListener('mouseleave', resetTilt);
      window.removeEventListener('blur', resetTilt);
    };
  }, [visible, rawRotateX, rawRotateY, rawShiftX, rawShiftY, rawGlareX, rawGlareY]);

  // Composed, smooth startup staging (~2.85s minimum window on fast launches)
  useEffect(() => {
    const t1 = window.setTimeout(() => {
      setStageIndex(1);
      setTimedProgress((prev) => Math.max(prev, 38));
    }, 620);
    const t2 = window.setTimeout(() => {
      setStageIndex(2);
      setTimedProgress((prev) => Math.max(prev, 70));
    }, 1420);
    const t3 = window.setTimeout(() => {
      setTimedProgress((prev) => Math.max(prev, 90));
    }, 2200);
    const tMin = window.setTimeout(() => {
      setMinWindowElapsed(true);
    }, MIN_DISPLAY_MS);

    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.clearTimeout(t3);
      window.clearTimeout(tMin);
    };
  }, []);

  // Advance to 100% and dissolve smoothly only once boot is fully ready (models + update gate passed)
  useEffect(() => {
    if (!isBootReady || bootPhase !== 'ready' || !minWindowElapsed) return;
    setStageIndex(3);
    setTimedProgress(100);
    const dismissTimer = window.setTimeout(() => {
      setVisible(false);
    }, COMPLETE_HOLD_MS);
    return () => window.clearTimeout(dismissTimer);
  }, [isBootReady, bootPhase, minWindowElapsed]);

  // Allow Esc to skip only once startup & update gate have completely finished
  useEffect(() => {
    if (!visible) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isBootReady && bootPhase === 'ready') {
        setVisible(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [visible, isBootReady, bootPhase]);

  const isDownloadingModels = bootPhase === 'downloading_models';
  const hasModelsError = bootPhase === 'models_error';
  const isUpdatePromptOpen = bootPhase === 'update_prompt' && startupUpdate !== null;

  const downloadPercent = (() => {
    if (!modelDownloadProgress) return 4;
    if (modelDownloadProgress.overallTotalBytes > 0) {
      return Math.min(
        99,
        Math.max(
          2,
          Math.round(
            (modelDownloadProgress.overallDownloadedBytes /
              modelDownloadProgress.overallTotalBytes) *
              100,
          ),
        ),
      );
    }
    if (modelDownloadProgress.totalBytes > 0) {
      return Math.min(
        99,
        Math.max(
          2,
          Math.round((modelDownloadProgress.downloadedBytes / modelDownloadProgress.totalBytes) * 100),
        ),
      );
    }
    return 8;
  })();

  const progress = isDownloadingModels
    ? downloadPercent
    : hasModelsError
      ? 0
      : isUpdatePromptOpen
        ? 94
        : timedProgress;

  const statusText = (() => {
    if (isDownloadingModels) {
      if (modelDownloadProgress?.modelName) {
        return `Downloading ${modelDownloadProgress.modelName} (${modelDownloadProgress.index}/${modelDownloadProgress.totalCount})`;
      }
      const pendingCount =
        (modelManifestStatus?.missingCount ?? 0) + (modelManifestStatus?.outdatedCount ?? 0);
      return pendingCount > 0
        ? `Preparing AI models (${pendingCount} to download)…`
        : 'Downloading AI models…';
    }
    if (hasModelsError) {
      return 'AI model download paused';
    }
    if (isUpdatePromptOpen) {
      return `Update v${startupUpdate.latestVersion} available`;
    }
    if (bootPhase === 'checking_models') {
      return 'Verifying AI models…';
    }
    if (bootPhase === 'checking_updates') {
      return 'Checking for updates…';
    }
    if (stageIndex >= 3) {
      return runtimeInfo?.modelsReady ? 'AI Ready' : 'Ready';
    }
    if (stageIndex === 2) {
      return 'Preparing workspace…';
    }
    if (stageIndex === 1) {
      return runtimeInfo?.modelsReady ? 'Initializing AI…' : 'Preparing studio…';
    }
    return 'Starting Kroma…';
  })();

  const downloadDetailLine = (() => {
    if (!isDownloadingModels) return null;
    if (!modelDownloadProgress) {
      const totalPending = modelManifestStatus?.pendingDownloadBytes ?? 0;
      return totalPending > 0 ? `Connecting to GitHub Releases · ${formatByteSize(totalPending)}` : 'Connecting…';
    }
    const parts: string[] = [`${downloadPercent}%`];
    if (modelDownloadProgress.overallTotalBytes > 0) {
      parts.push(
        `${formatByteSize(modelDownloadProgress.overallDownloadedBytes)} / ${formatByteSize(modelDownloadProgress.overallTotalBytes)}`,
      );
    } else if (modelDownloadProgress.totalBytes > 0) {
      parts.push(
        `${formatByteSize(modelDownloadProgress.downloadedBytes)} / ${formatByteSize(modelDownloadProgress.totalBytes)}`,
      );
    }
    const speed = formatTransferSpeed(modelDownloadProgress.bytesPerSecond);
    if (speed) {
      parts.push(speed);
    }
    return parts.join(' · ');
  })();

  return (
    <>
      <AnimatePresence>
        {visible && (
          <motion.div
            role="status"
            aria-live="polite"
            aria-label="Loading Kroma"
            initial={{ opacity: 1 }}
            exit={{
              opacity: 0,
              scale: 1.025,
              transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] },
            }}
            className="fixed inset-0 z-[120] flex flex-col items-center justify-center select-none overflow-hidden px-6"
            style={{
              background:
                'radial-gradient(ellipse 80% 62% at 50% 34%, #1e1c1a 0%, #141519 46%, #0b0c0f 100%)',
            }}
          >
            {/* Layered Atmospheric Darkroom Gradient Wash (Warm Safelight Top + Cool Emerald Depth) */}
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0"
              style={{
                background:
                  'radial-gradient(circle 540px at 50% 38%, rgba(245, 158, 11, 0.11), transparent 68%), radial-gradient(circle 480px at 66% 70%, rgba(16, 185, 129, 0.08), transparent 70%)',
              }}
            />

            {/* Cursor-Reactive Soft Ambient Glow Behind the Emblem */}
            <motion.div
              aria-hidden="true"
              style={{
                x: ambientGlowX,
                y: ambientGlowY,
                background:
                  'radial-gradient(circle, rgba(245, 158, 11, 0.22) 0%, rgba(16, 185, 129, 0.11) 46%, transparent 72%)',
              }}
              className="pointer-events-none absolute w-[360px] h-[360px] rounded-full opacity-55 blur-[76px]"
            />

            {/* Centered Minimal Composition: Logo → Status → Progress Bar */}
            <div className="relative z-10 flex flex-col items-center w-full max-w-[420px]">
              {/* Seamless Handoff From Static Pre-React Boot Splash (zero double-fade blink) */}
              <motion.div
                initial={false}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ type: 'spring', stiffness: 230, damping: 16, mass: 0.9 }}
              >
                {/* Gentle Resting Float */}
                <motion.div
                  animate={{ y: [0, -5, 0] }}
                  transition={{
                    duration: 3.4,
                    repeat: Infinity,
                    ease: 'easeInOut',
                  }}
                >
                  {/* True 3D Cursor-Tracking Physics Tilt & Specular Shine Tile */}
                  <motion.div
                    ref={logoRef}
                    whileTap={{ scale: 0.93 }}
                    transition={{ type: 'spring', stiffness: 360, damping: 14 }}
                    style={{
                      transformPerspective: 680,
                      rotateX,
                      rotateY,
                      x: shiftX,
                      y: shiftY,
                      boxShadow: dynamicTileShadow,
                    }}
                    className="relative w-[108px] h-[108px] rounded-[28px] bg-gradient-to-b from-[#20232a] via-[#17191f] to-[#111317] border border-white/[0.11] flex items-center justify-center cursor-pointer"
                  >
                    {/* Clipped Inner Surface for Optical Glare & Sheen */}
                    <div className="relative w-full h-full rounded-[27px] overflow-hidden flex items-center justify-center">
                      <motion.div
                        aria-hidden="true"
                        className="pointer-events-none absolute inset-0"
                        style={{ background: rimHighlight }}
                      />

                      <motion.img
                        src={kromaLogo}
                        alt="Kroma"
                        draggable={false}
                        style={{ x: logoParallaxX, y: logoParallaxY }}
                        className="w-[106%] h-[106%] max-w-none object-cover select-none pointer-events-none"
                      />

                      <motion.div
                        aria-hidden="true"
                        className="pointer-events-none absolute inset-0"
                        style={{ background: specularOverlay }}
                      />

                      <motion.div
                        aria-hidden="true"
                        initial={{ x: '-145%' }}
                        animate={{ x: '165%' }}
                        transition={{
                          duration: 1.5,
                          repeat: Infinity,
                          repeatDelay: 1.9,
                          ease: [0.25, 0.1, 0.25, 1],
                        }}
                        className="pointer-events-none absolute -inset-y-6 w-[60%] rotate-[22deg]"
                        style={{
                          background:
                            'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.26) 50%, transparent 100%)',
                        }}
                      />
                    </div>
                  </motion.div>
                </motion.div>
              </motion.div>

              {/* Clean Status Line */}
              <div className="mt-7 min-h-[20px] flex items-center justify-center text-center">
                <AnimatePresence mode="wait">
                  <motion.span
                    key={statusText}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.2, ease: 'easeOut' }}
                    className="tabular-nums-mono text-[11.5px] tracking-[0.04em] text-ink-secondary"
                  >
                    {statusText}
                  </motion.span>
                </AnimatePresence>
              </div>

              {/* Minimal Precision Progress Bar */}
              {!hasModelsError && (
                <div
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={progress}
                  className={`mt-3 h-[3px] rounded-full bg-white/[0.08] overflow-hidden relative transition-all duration-300 ${
                    isDownloadingModels ? 'w-[280px]' : 'w-[224px]'
                  }`}
                >
                  <motion.div
                    initial={{ width: '0%' }}
                    animate={{ width: `${progress}%` }}
                    transition={{ type: 'spring', stiffness: 110, damping: 22, mass: 0.85 }}
                    className="h-full rounded-full bg-gradient-to-r from-amber-core via-amber-soft to-amber-core relative"
                  >
                    <span className="absolute right-0 top-1/2 -translate-y-1/2 w-5 h-full bg-white/60 blur-[2px] rounded-full" />
                  </motion.div>
                </div>
              )}

              {/* Live Model Download Transfer Telemetry */}
              <AnimatePresence>
                {isDownloadingModels && downloadDetailLine && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={FAST_FADE}
                    className="mt-3 flex flex-col items-center gap-2.5"
                  >
                    <span className="tabular-nums-mono text-[10.5px] text-ink-telemetry">
                      {downloadDetailLine}
                    </span>
                    <button
                      type="button"
                      onClick={() => void continueWithoutModels()}
                      className="text-[11px] text-ink-telemetry hover:text-ink-primary underline underline-offset-2 transition-colors cursor-pointer"
                    >
                      Skip for now (offline mode)
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Model Download Error Recovery Panel */}
              <AnimatePresence>
                {hasModelsError && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 4 }}
                    transition={FAST_FADE}
                    className="mt-5 w-full p-4 rounded-xl bg-[#15171c]/95 border border-hairline-strong shadow-[0_18px_48px_rgba(0,0,0,0.65)] flex flex-col gap-3 text-left"
                  >
                    <div className="flex items-start gap-2.5">
                      <span className="material-symbols-outlined text-[18px] text-amber-soft shrink-0 mt-0.5">
                        cloud_off
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="font-display text-[12.5px] font-semibold text-ink-primary">
                          Could not download required AI models
                        </div>
                        <p className="mt-1 text-[11.5px] leading-snug text-ink-secondary break-words">
                          {modelDownloadError ??
                            'Check your internet connection and verify the models-v1 release assets on GitHub.'}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center justify-end gap-2 pt-1">
                      <DarkroomButton
                        variant="ghost"
                        size="xs"
                        onClick={() => void continueWithoutModels()}
                      >
                        Continue Offline
                      </DarkroomButton>
                      <DarkroomButton
                        variant="primary"
                        size="xs"
                        icon="refresh"
                        onClick={() => void retryModelDownload()}
                      >
                        Retry Download
                      </DarkroomButton>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Pre-Landing Update Gate Modal (renders above SplashScreen before entering Landing Page) */}
      <AppUpdateModal
        isOpen={isUpdatePromptOpen}
        update={startupUpdate}
        mode="startup"
        onDismiss={dismissStartupUpdate}
      />
    </>
  );
}
