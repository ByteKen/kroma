import { memo, useEffect, useMemo, useRef } from 'react';
import { useStudio } from '../../app/StudioContext';
import kromaLogo from '../../assets/kroma-logo.png';
import { formatShortcutCombo, isMacPlatform, isSharedSystemFolder } from '../../domain/platform';
import { DarkroomTooltip } from '../darkroom-kit';

export const TopCommandBar = memo(function TopCommandBar() {
  const {
    session,
    activeWorkspace,
    settingsOpen,
    searchQuery,
    setSearchQuery,
    setActiveWorkspace,
    openParentFolder,
    isScanningFolder,
    isWindowMaximized,
    minimizeWindow,
    toggleMaximizeWindow,
    exitApp,
  } = useStudio();

  const searchInputRef = useRef<HTMLInputElement>(null);
  const isMac = useMemo(() => isMacPlatform(), []);
  const searchShortcutLabel = formatShortcutCombo('Ctrl+K', isMac);

  const isHomeView = !session || activeWorkspace === 'home';

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        isHomeView &&
        !settingsOpen &&
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === 'k'
      ) {
        event.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isHomeView, settingsOpen]);

  const totalLoadedPhotos = session
    ? session.clusters.reduce((sum, c) => sum + c.photos.length, 0)
    : 0;

  return (
    <header
      data-tauri-drag-region
      className="h-[52px] shrink-0 bg-surface-low border-b border-hairline pl-4 pr-2 flex items-center justify-between gap-4 z-50 select-none"
    >
      {/* Left: macOS Traffic Lights (macOS only) + Brand (Home) + Active Project Breadcrumb */}
      <div className="flex items-center gap-3 min-w-0">
        {isMac && (
          <div className="group/traffic flex items-center gap-2 pr-1.5 shrink-0">
            <DarkroomTooltip content="Close" side="bottom">
              <button
                type="button"
                onClick={() => void exitApp()}
                aria-label="Close Kroma"
                className="w-3 h-3 rounded-full bg-[#ff5f57] border border-black/25 flex items-center justify-center cursor-pointer"
              >
                <svg
                  width="6"
                  height="6"
                  viewBox="0 0 6 6"
                  aria-hidden="true"
                  className="opacity-0 group-hover/traffic:opacity-100 transition-opacity text-[#4d0000]"
                >
                  <path d="M1 1l4 4M5 1L1 5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                </svg>
              </button>
            </DarkroomTooltip>
            <DarkroomTooltip content="Minimize" side="bottom">
              <button
                type="button"
                onClick={() => void minimizeWindow()}
                aria-label="Minimize window"
                className="w-3 h-3 rounded-full bg-[#febc2e] border border-black/25 flex items-center justify-center cursor-pointer"
              >
                <svg
                  width="6"
                  height="6"
                  viewBox="0 0 6 6"
                  aria-hidden="true"
                  className="opacity-0 group-hover/traffic:opacity-100 transition-opacity text-[#5c3c00]"
                >
                  <path d="M1 3h4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                </svg>
              </button>
            </DarkroomTooltip>
            <DarkroomTooltip content={isWindowMaximized ? 'Restore' : 'Zoom'} side="bottom">
              <button
                type="button"
                onClick={() => void toggleMaximizeWindow()}
                aria-label={isWindowMaximized ? 'Restore window' : 'Zoom window'}
                className="w-3 h-3 rounded-full bg-[#28c840] border border-black/25 flex items-center justify-center cursor-pointer"
              >
                <svg
                  width="6"
                  height="6"
                  viewBox="0 0 6 6"
                  aria-hidden="true"
                  className="opacity-0 group-hover/traffic:opacity-100 transition-opacity text-[#004d0d]"
                >
                  <path d="M1.2 4.8L4.8 1.2M2.5 1.2h2.3v2.3M3.5 4.8H1.2V2.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                </svg>
              </button>
            </DarkroomTooltip>
          </div>
        )}

        <DarkroomTooltip content="Home & Recent Projects" side="bottom">
          <button
            type="button"
            onClick={() => setActiveWorkspace('home')}
            aria-label="Go to Home and Recent Projects"
            aria-current={isHomeView ? 'page' : undefined}
            className={`flex items-center gap-2 px-2 py-1 ${isMac ? '' : '-ml-2'} rounded-lg transition-colors shrink-0 cursor-pointer ${
              isHomeView
                ? 'bg-surface-active/80 text-amber-core'
                : 'text-amber-core hover:bg-surface-elevated'
            }`}
          >
            <img
              src={kromaLogo}
              alt=""
              aria-hidden="true"
              draggable={false}
              className="w-[22px] h-[22px] rounded-[5px] object-cover shrink-0 shadow-[0_1px_4px_rgba(0,0,0,0.45)]"
            />
            <span className="font-display text-[14px] font-bold tracking-[0.14em] text-ink-primary uppercase">
              KROMA
            </span>
          </button>
        </DarkroomTooltip>

        <div className="h-4 w-px bg-hairline shrink-0" />

        {session ? (
          <div className="flex items-center gap-2.5 min-w-0">
            <button
              type="button"
              onClick={() => {
                if (activeWorkspace === 'home') {
                  setActiveWorkspace(
                    session.sessionMode !== 'single' && session.isClustered ? 'culling' : 'develop',
                  );
                }
              }}
              className={`flex items-center gap-2 min-w-0 text-left ${
                activeWorkspace === 'home'
                  ? 'hover:text-amber-soft cursor-pointer transition-colors'
                  : 'cursor-default'
              }`}
            >
              <span className="font-display text-[13px] font-semibold text-ink-primary truncate">
                {session.telemetry.projectName}
              </span>
              <span className="tabular-nums-mono text-[11px] text-ink-telemetry shrink-0">
                {session.sessionMode === 'single'
                  ? 'Single Photo'
                  : session.sessionMode === 'files'
                    ? `${totalLoadedPhotos} selected${session.isClustered ? ` • ${session.clusters.length} groups` : ''}`
                    : `${totalLoadedPhotos} photos${session.isClustered ? ` • ${session.clusters.length} groups` : ' • Unclustered'}`}
              </span>
            </button>

            {(session.sessionMode === 'single' || session.sessionMode === 'files') &&
              session.folderPath &&
              !isSharedSystemFolder(session.folderPath) && (
                <DarkroomTooltip
                  content={`Load all photos in ${session.folderPath}`}
                  side="bottom"
                >
                  <button
                    type="button"
                    disabled={isScanningFolder}
                    onClick={() => void openParentFolder()}
                    className="h-6 px-2 rounded-md bg-surface hover:bg-surface-active border border-hairline hover:border-amber-core/50 text-ink-secondary hover:text-amber-soft tabular-nums-mono text-[10px] flex items-center gap-1 transition-colors cursor-pointer shrink-0 disabled:opacity-40"
                  >
                    <span className="material-symbols-outlined text-[13px]">folder_open</span>
                    <span className="hidden md:inline">Open Parent Folder</span>
                  </button>
                </DarkroomTooltip>
              )}
          </div>
        ) : (
          <span className="font-display text-[12px] font-medium text-ink-secondary">
            Home &amp; Recent Projects
          </span>
        )}
      </div>

      {/* Center: Recent Projects Search Bar (only shown on Home); drag spacer inside projects */}
      {isHomeView ? (
        <div className="flex-1 max-w-[380px] h-8 px-3 rounded-lg bg-canvas border border-hairline focus-within:border-amber-core transition-colors flex items-center gap-2">
          <span className="material-symbols-outlined text-[15px] text-ink-telemetry shrink-0">
            search
          </span>
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search recent projects..."
            className="bg-transparent text-ink-primary placeholder:text-ink-telemetry text-[12px] flex-1 min-w-0 outline-none border-none"
          />
          {searchQuery ? (
            <DarkroomTooltip content="Clear search" side="bottom">
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="text-ink-telemetry hover:text-ink-primary cursor-pointer flex items-center"
                aria-label="Clear search"
              >
                <span className="material-symbols-outlined text-[14px]">close</span>
              </button>
            </DarkroomTooltip>
          ) : (
            <kbd className="tabular-nums-mono text-[10px] bg-surface px-1.5 py-0.5 rounded border border-hairline text-ink-telemetry shrink-0">
              {searchShortcutLabel}
            </kbd>
          )}
        </div>
      ) : (
        <div
          data-tauri-drag-region
          onDoubleClick={() => void toggleMaximizeWindow()}
          className="flex-1 h-full min-w-[32px]"
        />
      )}

      {/* Right: Windows/Linux Frameless Window Controls (hidden on macOS where traffic lights sit top-left) */}
      {isMac ? (
        <div className="w-2 shrink-0" aria-hidden="true" />
      ) : (
        <div className="flex items-center gap-1 shrink-0">
          <DarkroomTooltip content="Minimize" side="bottom">
            <button
              type="button"
              onClick={() => void minimizeWindow()}
              aria-label="Minimize window"
              className="w-9 h-8 rounded-md flex items-center justify-center text-ink-secondary hover:text-ink-primary hover:bg-surface-elevated transition-colors cursor-pointer"
            >
              <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">
                <path d="M1.5 5.5h8" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
              </svg>
            </button>
          </DarkroomTooltip>

          <DarkroomTooltip content={isWindowMaximized ? 'Restore' : 'Maximize'} side="bottom">
            <button
              type="button"
              onClick={() => void toggleMaximizeWindow()}
              aria-label={isWindowMaximized ? 'Restore window' : 'Maximize window'}
              className="w-9 h-8 rounded-md flex items-center justify-center text-ink-secondary hover:text-ink-primary hover:bg-surface-elevated transition-colors cursor-pointer"
            >
              {isWindowMaximized ? (
                <svg width="11" height="11" viewBox="0 0 11 11" fill="none" aria-hidden="true">
                  <rect
                    x="1.5"
                    y="3"
                    width="6.5"
                    height="6.5"
                    rx="1"
                    stroke="currentColor"
                    strokeWidth="1.2"
                  />
                  <path
                    d="M3.5 3V2.2A1 1 0 0 1 4.5 1.2h4.3a1 1 0 0 1 1 1v4.3a1 1 0 0 1-1 1H8"
                    stroke="currentColor"
                    strokeWidth="1.2"
                  />
                </svg>
              ) : (
                <svg width="11" height="11" viewBox="0 0 11 11" fill="none" aria-hidden="true">
                  <rect
                    x="1.75"
                    y="1.75"
                    width="7.5"
                    height="7.5"
                    rx="1"
                    stroke="currentColor"
                    strokeWidth="1.25"
                  />
                </svg>
              )}
            </button>
          </DarkroomTooltip>

          <DarkroomTooltip content="Close" side="bottom">
            <button
              type="button"
              onClick={() => void exitApp()}
              aria-label="Close Kroma"
              className="w-9 h-8 rounded-md flex items-center justify-center text-ink-secondary hover:text-white hover:bg-crimson-clip transition-colors cursor-pointer"
            >
              <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">
                <path
                  d="M2 2l7 7M9 2L2 9"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </DarkroomTooltip>
        </div>
      )}
    </header>
  );
});
