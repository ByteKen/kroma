import { AnimatePresence, motion } from 'framer-motion';
import { useEffect } from 'react';
import kromaLogo from '../../assets/kroma-logo.png';
import type { AppUpdateCheckResult } from '../../domain/appUpdates';
import { DarkroomButton, FAST_FADE, MarkdownView, SOFT_SPRING } from '../darkroom-kit';

interface AppUpdateModalProps {
  readonly isOpen: boolean;
  readonly update: AppUpdateCheckResult | null;
  readonly mode: 'startup' | 'manual';
  readonly onDismiss: () => void;
}

function formatPublishedDate(iso: string | null): string | null {
  if (!iso) return null;
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export function AppUpdateModal({ isOpen, update, mode, onDismiss }: AppUpdateModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onDismiss();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, onDismiss]);

  const publishedLabel = formatPublishedDate(update?.publishedAt ?? null);
  const latestVersion = update?.latestVersion ?? '';
  const currentVersion = update?.currentVersion ?? '';
  const changelogSourceLabel =
    update?.changelogSource === 'changelog_file'
      ? 'CHANGELOG.md'
      : update?.changelogSource === 'release_body'
        ? 'Release Notes'
        : null;

  return (
    <AnimatePresence>
      {isOpen && update && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Kroma v${latestVersion} Update Available`}
          className="fixed inset-0 z-[140] flex items-center justify-center p-4 sm:p-6 select-none"
        >
          {/* Dimmed backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={FAST_FADE}
            onClick={mode === 'manual' ? onDismiss : undefined}
            className="absolute inset-0 bg-black/72 backdrop-blur-[6px]"
          />

          {/* Centered Update Dialog */}
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 8 }}
            transition={SOFT_SPRING}
            className="relative z-10 w-full max-w-[560px] max-h-[82vh] rounded-2xl bg-[#15171c] border border-hairline-strong shadow-[0_28px_80px_rgba(0,0,0,0.82)] flex flex-col overflow-hidden"
          >
            {/* Header */}
            <header className="shrink-0 px-6 pt-5 pb-4 border-b border-hairline flex items-start justify-between gap-4">
              <div className="flex items-center gap-3.5 min-w-0">
                <img
                  src={kromaLogo}
                  alt="Kroma"
                  draggable={false}
                  className="w-11 h-11 rounded-xl object-cover shrink-0 shadow-[0_6px_18px_rgba(0,0,0,0.48)] border border-white/10"
                />
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h2 className="font-display text-[17px] font-semibold tracking-tight text-ink-primary">
                      Update Available
                    </h2>
                    <span className="tabular-nums-mono text-[10.5px] px-2 py-0.5 rounded-full bg-amber-core/15 border border-amber-core/40 text-amber-soft font-semibold">
                      v{currentVersion} → v{latestVersion}
                    </span>
                  </div>
                  <div className="mt-1 flex items-center gap-2 text-[11.5px] text-ink-secondary">
                    <span className="truncate">{update.releaseName ?? `Kroma v${latestVersion}`}</span>
                    {publishedLabel && (
                      <>
                        <span className="text-ink-telemetry">•</span>
                        <span className="tabular-nums-mono text-[11px] text-ink-telemetry shrink-0">
                          {publishedLabel}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {mode === 'manual' && (
                <button
                  type="button"
                  onClick={onDismiss}
                  aria-label="Close update notes"
                  className="w-7 h-7 rounded-lg flex items-center justify-center text-ink-telemetry hover:text-ink-primary hover:bg-surface-elevated transition-colors cursor-pointer shrink-0"
                >
                  <span className="material-symbols-outlined text-[16px]">close</span>
                </button>
              )}
            </header>

            {/* Scrollable Markdown Changelog Body */}
            <div className="flex-1 min-h-0 overflow-y-auto px-6 py-4 select-text">
              <div className="flex items-center justify-between gap-2 mb-3">
                <span className="tabular-nums-mono text-[10px] uppercase tracking-[0.14em] text-ink-telemetry font-medium">
                  What’s Changed
                </span>
                {changelogSourceLabel && (
                  <span className="tabular-nums-mono text-[9.5px] px-1.5 py-0.5 rounded bg-surface border border-hairline text-ink-telemetry">
                    {changelogSourceLabel}
                  </span>
                )}
              </div>

              {update.changelogMarkdown ? (
                <MarkdownView markdown={update.changelogMarkdown} />
              ) : (
                <p className="text-[12.5px] leading-relaxed text-ink-secondary">
                  A newer version of Kroma (<strong className="text-ink-primary">v{latestVersion}</strong>) is
                  available on GitHub Releases. Open the release page to inspect full details and
                  download the latest desktop installer.
                </p>
              )}
            </div>

            {/* Footer Actions */}
            <footer className="shrink-0 px-6 py-3.5 border-t border-hairline bg-surface/70 flex items-center justify-between gap-3">
              <span className="text-[11px] text-ink-telemetry truncate">
                {mode === 'startup'
                  ? 'You can update now or continue into Kroma.'
                  : 'Downloads open in your default browser.'}
              </span>

              <div className="flex items-center gap-2 shrink-0">
                <DarkroomButton variant="ghost" size="sm" onClick={onDismiss}>
                  {mode === 'startup' ? 'Remind Me Later' : 'Close'}
                </DarkroomButton>
                <a
                  href={update.releaseUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 h-8 px-3.5 rounded-lg bg-amber-core text-amber-ink font-display text-[12px] font-bold hover:brightness-110 active:scale-[0.98] transition-all shadow-[0_2px_10px_rgba(245,158,11,0.28)]"
                >
                  <span className="material-symbols-outlined text-[15px]">download</span>
                  Download Update
                </a>
              </div>
            </footer>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
