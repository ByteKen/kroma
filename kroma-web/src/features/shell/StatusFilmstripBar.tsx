import { AnimatePresence, motion } from 'framer-motion';
import { useStudio } from '../../app/StudioContext';
import { KROMA_FALLBACK_VERSION } from '../../domain/appUpdates';
import { formatShortcutCombo, isMacPlatform } from '../../domain/platform';
import { DarkroomButton, DarkroomTooltip, FAST_FADE, TelemetryPill } from '../darkroom-kit';

/**
 * Slim 30px darkroom status strip along the bottom edge:
 * - Shows live scan progress, quiet bottom-left action status (+ inline Undo pill), and app version info.
 */
export function StatusFilmstripBar() {
  const {
    isDesktopRuntime,
    session,
    runtimeInfo,
    isScanningFolder,
    scanProgress,
    thumbnailRenderProgress,
    developViewportStatus,
    activeWorkspace,
    manualMatchSplitOpen,
    statusBanner,
    canUndo,
    canRedo,
    undoLabel,
    redoLabel,
    undoOrganizationAction,
    redoOrganizationAction,
    canUndoDevelop,
    undoDevelopLabel,
    undoDevelopAction,
  } = useStudio();

  const isMac = isMacPlatform();
  const undoKey = formatShortcutCombo('Ctrl+Z', isMac);
  const redoKey = formatShortcutCombo('Ctrl+Shift+Z', isMac);
  const appVersion = runtimeInfo?.appVersion ?? KROMA_FALLBACK_VERSION;

  const defaultStatus = session
    ? `Loaded ${session.telemetry.totalPhotos} photo(s) in ${session.telemetry.indexElapsedSeconds}s${
        session.telemetry.indexErrorCount > 0
          ? ` • ${session.telemetry.indexErrorCount} skipped`
          : ''
      }`
    : 'Ready • Open a project to begin';

  const statusText = scanProgress?.message ?? statusBanner ?? defaultStatus;
  const progressPct =
    scanProgress && scanProgress.total > 0
      ? Math.round((scanProgress.completed / scanProgress.total) * 100)
      : null;

  const isStudio = activeWorkspace === 'develop';
  const activeCanUndo = isStudio ? canUndoDevelop : canUndo;
  const activeUndoLabel = isStudio ? undoDevelopLabel : undoLabel;
  const triggerActiveUndo = () => {
    if (isStudio) {
      undoDevelopAction();
    } else {
      void undoOrganizationAction();
    }
  };

  const isFloatingDockVisible =
    Boolean(session) && activeWorkspace === 'culling' && !manualMatchSplitOpen;
  const thumbProgressPct =
    thumbnailRenderProgress && thumbnailRenderProgress.total > 0
      ? Math.min(
          100,
          Math.max(
            0,
            Math.round((thumbnailRenderProgress.completed / thumbnailRenderProgress.total) * 100),
          ),
        )
      : null;
  const backendTag = developViewportStatus?.backend === 'webgpu' ? 'WEBGPU' : 'WEBGL2';
  const developTelemetryLabel = developViewportStatus
    ? developViewportStatus.state === 'decoding'
      ? developViewportStatus.precision === 'f32'
        ? 'Decoding 32-bit RAW…'
        : 'Decoding 16-bit RAW…'
      : developViewportStatus.precision === 'f32'
        ? developViewportStatus.isRawDemosaic
          ? `32-BIT RAW • ${backendTag}${developViewportStatus.width && developViewportStatus.height ? ` (${developViewportStatus.width}×${developViewportStatus.height})` : ''}`
          : `32-BIT FLOAT • ${backendTag}${developViewportStatus.width && developViewportStatus.height ? ` (${developViewportStatus.width}×${developViewportStatus.height})` : ''}`
        : developViewportStatus.isRawDemosaic
          ? `16-BIT RAW • ${backendTag}${developViewportStatus.width && developViewportStatus.height ? ` (${developViewportStatus.width}×${developViewportStatus.height})` : ''}`
          : `16-BIT FLOAT • ${backendTag}${developViewportStatus.width && developViewportStatus.height ? ` (${developViewportStatus.width}×${developViewportStatus.height})` : ''}`
    : null;
  const developTelemetryTooltip = developViewportStatus
    ? developViewportStatus.state === 'decoding'
      ? `Demosaicing ${developViewportStatus.precision === 'f32' ? '32-bit' : '16-bit'} scene-linear wide-gamut Rec.2020 RAW for Studio…`
      : `${developViewportStatus.precision === 'f32' ? '32-bit' : '16-bit'} ${developViewportStatus.isRawDemosaic ? 'Scene-Linear Wide-Gamut Rec.2020 RAW' : 'High-Bit-Depth Float'} • ${backendTag}${developViewportStatus.width && developViewportStatus.height ? ` (${developViewportStatus.width}×${developViewportStatus.height})` : ''}`
    : '';
  const appInfoLabel = isDesktopRuntime ? `Kroma v${appVersion}` : `Kroma v${appVersion} • Preview`;
  const appInfoTooltip = isDesktopRuntime
    ? [
        `Kroma v${appVersion}`,
        runtimeInfo?.developGpuBackendLabel && runtimeInfo.developGpuBackendLabel !== 'CPU'
          ? runtimeInfo.developGpuBackendLabel
          : null,
        runtimeInfo?.backendLabel && runtimeInfo.backendLabel !== 'CPU'
          ? runtimeInfo.backendLabel
          : null,
      ]
        .filter(Boolean)
        .join(' • ')
    : `Kroma v${appVersion} • Browser Preview`;

  return (
    <footer className="h-[30px] shrink-0 bg-surface-lowest border-t border-hairline px-3.5 flex items-center justify-between gap-3 z-50">
      {/* Left: Live Project Status, Scan Progress, or Quiet Action Feedback + Quick Undo */}
      <div className="flex items-center gap-2 min-w-0 flex-1 tabular-nums-mono text-[10.5px] overflow-hidden">
        <span
          className={`material-symbols-outlined text-[14px] shrink-0 ${
            statusBanner ? 'text-amber-soft' : 'text-emerald-focus'
          }`}
        >
          {isScanningFolder ? 'hourglass_top' : statusBanner ? 'bolt' : 'check_circle'}
        </span>
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={statusText}
            initial={{ opacity: 0, y: 2 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -2 }}
            transition={FAST_FADE}
            className={`truncate ${
              scanProgress || statusBanner ? 'text-amber-soft' : 'text-ink-secondary'
            }`}
          >
            {statusText}
          </motion.span>
        </AnimatePresence>

        {statusBanner && activeCanUndo && !isScanningFolder && (
          <DarkroomTooltip
            content={activeUndoLabel ? `Undo ${activeUndoLabel} (${undoKey})` : `Undo (${undoKey})`}
            side="top"
          >
            <button
              type="button"
              onClick={triggerActiveUndo}
              className="px-1.5 py-0.5 rounded bg-surface-active hover:bg-amber-core hover:text-amber-ink border border-amber-core/45 text-amber-soft tabular-nums-mono text-[9.5px] font-semibold transition-colors cursor-pointer shrink-0 leading-none"
            >
              Undo
            </button>
          </DarkroomTooltip>
        )}

        {progressPct !== null && (
          <div className="w-24 h-1.5 rounded-full bg-canvas border border-hairline overflow-hidden shrink-0">
            <div
              className="h-full bg-amber-core transition-all duration-150"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        )}
      </div>

      {/* Right: Undo/Redo (when Floating Dock is hidden) + 32-Bit Decode & Preview Progress + App Version */}
      <div className="flex items-center gap-2 shrink-0">
        {session && !isFloatingDockVisible && (canUndo || canRedo) && (
          <div className="flex items-center gap-1">
            <DarkroomButton
              variant="ghost"
              size="xs"
              icon="undo"
              tooltipSide="top"
              disabled={!canUndo || isScanningFolder}
              onClick={() => void undoOrganizationAction()}
              title={undoLabel ? `Undo ${undoLabel} (${undoKey})` : `Undo (${undoKey})`}
              aria-label="Undo organization change"
              className="h-5 px-1"
            />
            <DarkroomButton
              variant="ghost"
              size="xs"
              icon="redo"
              tooltipSide="top"
              disabled={!canRedo || isScanningFolder}
              onClick={() => void redoOrganizationAction()}
              title={redoLabel ? `Redo ${redoLabel} (${redoKey})` : `Redo (${redoKey})`}
              aria-label="Redo organization change"
              className="h-5 px-1"
            />
          </div>
        )}

        {thumbnailRenderProgress?.active && thumbProgressPct !== null && (
          <DarkroomTooltip
            content={`Rendering 32-bit linear grid & filmstrip previews (${thumbnailRenderProgress.completed}/${thumbnailRenderProgress.total}) — change anytime in Settings → Performance`}
            side="top"
          >
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-surface-active/80 border border-amber-core/35 tabular-nums-mono text-[9.5px] text-amber-soft shrink-0">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-core animate-pulse shrink-0" />
              <span>
                32-bit Previews {thumbnailRenderProgress.completed}/{thumbnailRenderProgress.total}
              </span>
              <div className="w-10 h-1 rounded-full bg-canvas border border-hairline overflow-hidden shrink-0">
                <div
                  className="h-full bg-amber-core transition-all duration-150"
                  style={{ width: `${thumbProgressPct}%` }}
                />
              </div>
            </div>
          </DarkroomTooltip>
        )}

        {isStudio && developViewportStatus && developTelemetryLabel && (
          <DarkroomTooltip content={developTelemetryTooltip} side="top">
            {developViewportStatus.state === 'decoding' ? (
              <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-surface-active/80 border border-emerald-focus/35 tabular-nums-mono text-[9.5px] text-emerald-bright shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-focus animate-pulse shrink-0" />
                <span>{developTelemetryLabel}</span>
              </div>
            ) : (
              <TelemetryPill
                tone={developViewportStatus.isRawDemosaic ? 'emerald' : 'amber'}
                dot
                className="py-0.5 px-2 text-[9.5px] shrink-0"
              >
                {developTelemetryLabel}
              </TelemetryPill>
            )}
          </DarkroomTooltip>
        )}

        <DarkroomTooltip content={appInfoTooltip} side="top">
          <TelemetryPill
            tone="neutral"
            className="hidden md:inline-flex py-0.5 px-2 text-[9.5px] shrink-0"
          >
            <span className="text-ink-telemetry">{appInfoLabel}</span>
          </TelemetryPill>
        </DarkroomTooltip>
      </div>
    </footer>
  );
}
