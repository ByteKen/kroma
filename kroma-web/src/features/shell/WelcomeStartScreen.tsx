import { motion } from 'framer-motion';
import { useMemo, useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import { formatShortcutCombo, isMacPlatform } from '../../domain/platform';
import {
  filterRecentProjects,
  formatRelativeOpenedAt,
  normalizeProjectPathKey,
  type RecentProjectEntry,
  type RecentProjectsFilter,
  type RecentProjectsSort,
} from '../../domain/recentProjects';
import {
  DarkroomButton,
  DarkroomTooltip,
  SegmentedControl,
  SOFT_SPRING,
  type SegmentedOption,
} from '../darkroom-kit';

type RecentsViewMode = 'grid' | 'list';

const VIEW_MODES: readonly SegmentedOption<RecentsViewMode>[] = [
  { id: 'grid', icon: 'grid_view', title: 'Contact sheet' },
  { id: 'list', icon: 'view_list', title: 'Table view' },
];

const SORT_CYCLE: readonly { readonly id: RecentProjectsSort; readonly label: string }[] = [
  { id: 'recent', label: 'Last Opened' },
  { id: 'name', label: 'Name' },
  { id: 'photos', label: 'Frame Count' },
];

function ContactSheetCover({
  thumbnails,
  projectName,
}: {
  readonly thumbnails: readonly string[];
  readonly projectName: string;
}) {
  if (thumbnails.length >= 3) {
    return (
      <div className="grid grid-cols-12 grid-rows-2 gap-[2px] w-full h-full bg-surface-lowest">
        <div className="col-span-8 row-span-2 overflow-hidden bg-surface-lowest">
          <img
            src={thumbnails[0]}
            alt={projectName}
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-[1.03]"
          />
        </div>
        <div className="col-span-4 overflow-hidden bg-surface-lowest">
          <img
            src={thumbnails[1]}
            alt=""
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-[1.05]"
          />
        </div>
        <div className="col-span-4 overflow-hidden bg-surface-lowest">
          <img
            src={thumbnails[2]}
            alt=""
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-[1.05]"
          />
        </div>
      </div>
    );
  }

  if (thumbnails.length === 2) {
    return (
      <div className="grid grid-cols-2 gap-[2px] w-full h-full bg-surface-lowest">
        {thumbnails.map((src, i) => (
          <div key={i} className="overflow-hidden bg-surface-lowest">
            <img
              src={src}
              alt={i === 0 ? projectName : ''}
              loading="lazy"
              decoding="async"
              className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-[1.03]"
            />
          </div>
        ))}
      </div>
    );
  }

  if (thumbnails.length === 1) {
    return (
      <img
        src={thumbnails[0]}
        alt={projectName}
        loading="lazy"
        decoding="async"
        className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-[1.03]"
      />
    );
  }

  return (
    <div className="w-full h-full bg-surface-lowest flex items-center justify-center text-ink-telemetry">
      <span className="material-symbols-outlined text-[26px] opacity-40">folder_open</span>
    </div>
  );
}

/**
 * Edge-to-edge Industrial Darkroom Project Browser (inspired by DaVinci Resolve / Capture One / Adobe Home):
 * - Flush left library rail for launching folders, filtering recents, and inspecting local engine specs.
 * - Full-bleed contact-sheet / data-table project browser with zero generic SaaS cards or dead padding.
 */
export function WelcomeStartScreen() {
  const {
    session,
    runtimeInfo,
    recentProjects,
    isScanningFolder,
    scanProgress,
    searchQuery,
    setSearchQuery,
    sfaceThreshold,
    reviewWarningThreshold,
    openShootFolder,
    openPhotoFiles,
    openRecentShootFolder,
    removeRecentProjectEntry,
    togglePinRecentProjectEntry,
    setActiveWorkspace,
  } = useStudio();

  const isMac = useMemo(() => isMacPlatform(), []);
  const openShortcutLabel = formatShortcutCombo('Ctrl+O', isMac);
  const openPhotosShortcutLabel = formatShortcutCombo('Ctrl+Shift+O', isMac);

  const [viewMode, setViewMode] = useState<RecentsViewMode>('grid');
  const [filterTab, setFilterTab] = useState<RecentProjectsFilter>('all');
  const [sortMode, setSortMode] = useState<RecentProjectsSort>('recent');

  const modelsReady = runtimeInfo?.modelsReady ?? false;
  const rawCount = runtimeInfo?.supportedRawCount ?? 29;
  const pct =
    scanProgress && scanProgress.total > 0
      ? Math.round((scanProgress.completed / scanProgress.total) * 100)
      : null;

  const activeFolderKey = session?.folderPath
    ? normalizeProjectPathKey(session.folderPath)
    : null;

  const counts = useMemo(
    () => ({
      all: recentProjects.length,
      pinned: recentProjects.filter((p) => p.isPinned).length,
      clustered: recentProjects.filter((p) => p.isClustered).length,
      unclustered: recentProjects.filter((p) => !p.isClustered).length,
    }),
    [recentProjects],
  );

  const visibleRecents = useMemo(
    () => filterRecentProjects(recentProjects, searchQuery, filterTab, sortMode),
    [recentProjects, searchQuery, filterTab, sortMode],
  );

  const cycleSort = () => {
    const idx = SORT_CYCLE.findIndex((s) => s.id === sortMode);
    setSortMode(SORT_CYCLE[(idx + 1) % SORT_CYCLE.length].id);
  };

  const currentSortLabel = SORT_CYCLE.find((s) => s.id === sortMode)?.label ?? 'Last Opened';

  const libraryTabs: readonly {
    readonly id: RecentProjectsFilter;
    readonly icon: string;
    readonly label: string;
    readonly count: number;
  }[] = [
      { id: 'all', icon: 'schedule', label: 'Recent Projects', count: counts.all },
      { id: 'pinned', icon: 'keep', label: 'Pinned', count: counts.pinned },
      { id: 'clustered', icon: 'face', label: 'Organized', count: counts.clustered },
      { id: 'unclustered', icon: 'folder_open', label: 'Unclustered', count: counts.unclustered },
    ];

  const activeTabLabel = libraryTabs.find((t) => t.id === filterTab)?.label ?? 'Recent Projects';

  const emptyStateCopy: Record<RecentProjectsFilter, { readonly title: string; readonly hint: string }> = {
    all: {
      title: 'No matching projects',
      hint: 'Clear your search query or open a local folder to start a project.',
    },
    pinned: {
      title: 'No pinned projects yet',
      hint: 'Click the pin icon on any recent project card to keep it pinned here.',
    },
    clustered: {
      title: 'No organized projects yet',
      hint: 'Projects appear here automatically after running Organize by Face.',
    },
    unclustered: {
      title: 'No unclustered projects',
      hint: 'Opened projects waiting for face organization appear here.',
    },
  };

  const handleOpenRecent = (project: RecentProjectEntry) => {
    if (isScanningFolder) return;
    if (activeFolderKey && activeFolderKey === project.id && session) {
      setActiveWorkspace(session.isClustered ? 'culling' : 'develop');
      return;
    }
    void openRecentShootFolder(project.folderPath);
  };

  const showOpenProjectTile = filterTab === 'all' && searchQuery.trim().length === 0;

  return (
    <div className="flex-1 min-w-0 flex overflow-hidden bg-canvas">
      {/* LEFT LIBRARY SIDEBAR — Flush Native Desktop Project Navigation */}
      <aside
        aria-label="Project Library Navigation"
        className="w-[224px] shrink-0 bg-surface-low border-r border-hairline flex flex-col justify-between overflow-hidden"
      >
        <div className="flex flex-col overflow-y-auto">
          {session && (
            <div className="p-3 border-b border-hairline">
              <button
                type="button"
                onClick={() =>
                  setActiveWorkspace(
                    session.sessionMode !== 'single' && session.isClustered ? 'culling' : 'develop',
                  )
                }
                className="w-full p-2.5 rounded-lg bg-surface border border-emerald-focus/45 hover:border-emerald-bright/70 text-left flex items-center justify-between gap-2 transition-colors cursor-pointer group"
              >
                <div className="min-w-0 flex flex-col">
                  <span className="tabular-nums-mono text-[9px] uppercase tracking-wider text-emerald-bright font-semibold">
                    {session.sessionMode === 'single' ? 'Active Photo' : 'Active Project'}
                  </span>
                  <span className="font-display text-[12px] font-bold text-ink-primary truncate">
                    {session.telemetry.projectName}
                  </span>
                </div>
                <span className="material-symbols-outlined text-[16px] text-emerald-bright group-hover:translate-x-0.5 transition-transform shrink-0">
                  arrow_forward
                </span>
              </button>
            </div>
          )}

          {/* Library Filters */}
          <nav className="p-2.5 flex flex-col gap-0.5" aria-label="Project Library Filters">
            <div className="px-2 pb-1.5 pt-0.5 tabular-nums-mono text-[9.5px] uppercase tracking-[0.12em] text-ink-telemetry">
              Catalog
            </div>
            {libraryTabs.map((tab) => {
              const isActive = filterTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setFilterTab(tab.id)}
                  className={`h-8 px-2.5 rounded-md flex items-center justify-between gap-2 text-[12px] transition-colors cursor-pointer ${isActive
                      ? 'bg-surface-active text-ink-primary font-semibold border border-hairline-strong'
                      : 'text-ink-secondary hover:bg-surface hover:text-ink-primary border border-transparent'
                    }`}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <span
                      className={`material-symbols-outlined text-[16px] shrink-0 ${isActive ? 'text-amber-soft' : 'text-ink-telemetry'
                        }`}
                    >
                      {tab.icon}
                    </span>
                    <span className="truncate">{tab.label}</span>
                  </span>
                  <span className="tabular-nums-mono text-[10px] text-ink-telemetry shrink-0">
                    {tab.count}
                  </span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Hardware / Engine Telemetry Spec Plate (Utilitarian Instrument Look) */}
        <div className="p-3 bg-surface-lowest/70 border-t border-hairline flex flex-col gap-1.5 tabular-nums-mono text-[10px]">
          <div className="flex items-center justify-between">
            <span className="text-ink-telemetry">STATUS</span>
            <span className="flex items-center gap-1.5 text-ink-primary">
              <span
                className={`w-1.5 h-1.5 rounded-full ${modelsReady ? 'bg-emerald-focus' : 'bg-tangerine-alert'
                  }`}
              />
              {modelsReady ? 'AI Ready' : 'Offline'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-ink-telemetry">CODECS</span>
            <span className="text-ink-secondary">{rawCount} RAW • PNG • JPG</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-ink-telemetry">CUTOFF / REV</span>
            <span className="text-ink-secondary">
              {sfaceThreshold.toFixed(2)} / {Math.round(reviewWarningThreshold * 100)}%
            </span>
          </div>
        </div>
      </aside>

      {/* MAIN EDGE-TO-EDGE PROJECT BROWSER */}
      <section
        aria-label="Recent Projects Browser"
        className="flex-1 min-w-0 flex flex-col overflow-hidden bg-canvas"
      >
        {/* Flush Desktop Browser Toolbar */}
        <div className="h-11 px-5 bg-surface border-b border-hairline flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <h1 className="font-display text-[14px] font-bold text-ink-primary tracking-tight truncate">
              {activeTabLabel}
            </h1>
            <span className="tabular-nums-mono text-[11px] text-ink-telemetry shrink-0">
              {visibleRecents.length} {visibleRecents.length === 1 ? 'project' : 'projects'}
            </span>
            {searchQuery.trim().length > 0 && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="px-2 py-0.5 rounded bg-surface-active border border-hairline tabular-nums-mono text-[10px] text-amber-soft hover:border-amber-core/50 cursor-pointer flex items-center gap-1"
              >
                <span>&ldquo;{searchQuery.trim()}&rdquo;</span>
                <span className="material-symbols-outlined text-[12px]">close</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <DarkroomButton
              variant="subtle"
              size="xs"
              icon="add_photo_alternate"
              disabled={isScanningFolder}
              onClick={() => void openPhotoFiles()}
              title={`Open one or more specific photos (${openPhotosShortcutLabel})`}
              className="h-7 px-2.5"
            >
              Open Photo(s)
            </DarkroomButton>

            <DarkroomButton
              variant="subtle"
              size="xs"
              onClick={cycleSort}
              title="Change sort order"
              className="h-7 px-2.5 tabular-nums-mono text-[10.5px]"
            >
              <span>{currentSortLabel}</span>
              <span className="material-symbols-outlined text-[14px]">unfold_more</span>
            </DarkroomButton>

            <SegmentedControl
              layoutId="home-projects-view-mode"
              options={VIEW_MODES}
              value={viewMode}
              onChange={setViewMode}
            />
          </div>
        </div>

        {/* Live Folder Loading Bar */}
        {isScanningFolder && scanProgress && (
          <div className="px-5 py-2 bg-surface-low border-b border-hairline flex items-center gap-3 shrink-0">
            <span className="material-symbols-outlined text-[15px] text-amber-soft animate-pulse">
              hourglass_top
            </span>
            <span className="tabular-nums-mono text-[11px] text-amber-soft truncate flex-1">
              {scanProgress.message}
            </span>
            <div className="w-40 h-1.5 rounded-full bg-canvas border border-hairline overflow-hidden shrink-0">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct ?? 15}%` }}
                transition={SOFT_SPRING}
                className="h-full bg-amber-core"
              />
            </div>
            {pct !== null && (
              <span className="tabular-nums-mono text-[11px] text-ink-primary font-semibold w-9 text-right">
                {pct}%
              </span>
            )}
          </div>
        )}

        {/* Main Scrollable Light-Table Project Canvas */}
        <div
          className="flex-1 min-h-0 overflow-y-auto p-5"
          style={{
            backgroundImage:
              'radial-gradient(rgba(255,255,255,0.035) 1px, transparent 1px)',
            backgroundSize: '24px 24px',
          }}
        >
          {viewMode === 'list' && visibleRecents.length > 0 ? (
            /* Enterprise Columnar Data Table (Capture One / Bridge / Figma List View) */
            <div className="rounded-lg bg-surface border border-hairline overflow-hidden">
              <div className="grid grid-cols-[minmax(220px,1.5fr)_minmax(180px,1.8fr)_88px_110px_96px_68px] items-center gap-3 px-4 h-8 bg-surface-low border-b border-hairline tabular-nums-mono text-[9.5px] uppercase tracking-[0.1em] text-ink-telemetry">
                <span>Project</span>
                <span>Directory</span>
                <span className="text-right">Frames</span>
                <span>Status</span>
                <span className="text-right">Opened</span>
                <span className="text-right">Actions</span>
              </div>

              <div className="divide-y divide-hairline">
                {visibleRecents.map((project) => {
                  const isOpenNow = activeFolderKey === project.id;
                  return (
                    <div
                      key={project.id}
                      onClick={() => handleOpenRecent(project)}
                      className="group grid grid-cols-[minmax(220px,1.5fr)_minmax(180px,1.8fr)_88px_110px_96px_68px] items-center gap-3 px-4 py-2.5 hover:bg-surface-elevated transition-colors cursor-pointer"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="w-11 h-8 rounded overflow-hidden bg-surface-lowest border border-hairline shrink-0">
                          {project.coverThumbnails[0] ? (
                            <img
                              src={project.coverThumbnails[0]}
                              alt={project.projectName}
                              loading="lazy"
                              decoding="async"
                              className="w-full h-full object-cover object-top"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-ink-telemetry">
                              <span className="material-symbols-outlined text-[15px]">folder</span>
                            </div>
                          )}
                        </div>
                        <div className="min-w-0 flex items-center gap-1.5">
                          <span className="font-display text-[12.5px] font-semibold text-ink-primary truncate group-hover:text-amber-soft transition-colors">
                            {project.projectName}
                          </span>
                          {project.isPinned && (
                            <span className="material-symbols-outlined text-[13px] text-amber-soft shrink-0">
                              keep
                            </span>
                          )}
                        </div>
                      </div>

                      <span className="tabular-nums-mono text-[10.5px] text-ink-telemetry truncate">
                        {project.folderPath}
                      </span>

                      <span className="tabular-nums-mono text-[11px] text-ink-secondary text-right">
                        {project.photoCount.toLocaleString()}
                      </span>

                      <div className="flex items-center">
                        {isOpenNow ? (
                          <span className="tabular-nums-mono text-[10px] text-amber-soft font-medium">
                            ● Open
                          </span>
                        ) : project.isClustered ? (
                          <span className="tabular-nums-mono text-[10px] text-emerald-bright">
                            ● {project.clusterCount} subjects
                          </span>
                        ) : (
                          <span className="tabular-nums-mono text-[10px] text-ink-telemetry">
                            ○ Unclustered
                          </span>
                        )}
                      </div>

                      <span className="tabular-nums-mono text-[10.5px] text-ink-telemetry text-right">
                        {formatRelativeOpenedAt(project.lastOpenedAtIso)}
                      </span>

                      <div
                        className="flex items-center justify-end gap-0.5"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <DarkroomTooltip content={project.isPinned ? 'Unpin' : 'Pin'} side="top">
                          <button
                            type="button"
                            onClick={() => togglePinRecentProjectEntry(project.folderPath)}
                            aria-label={project.isPinned ? 'Unpin project' : 'Pin project'}
                            className={`w-7 h-7 rounded flex items-center justify-center transition-colors cursor-pointer ${project.isPinned
                                ? 'text-amber-soft bg-surface-active'
                                : 'text-ink-telemetry hover:text-ink-primary hover:bg-surface-active'
                              }`}
                          >
                            <span className="material-symbols-outlined text-[14px]">keep</span>
                          </button>
                        </DarkroomTooltip>
                        <DarkroomTooltip content="Remove" side="top">
                          <button
                            type="button"
                            onClick={() => removeRecentProjectEntry(project.folderPath)}
                            aria-label={`Remove ${project.projectName}`}
                            className="w-7 h-7 rounded text-ink-telemetry hover:text-crimson-clip hover:bg-surface-active flex items-center justify-center transition-colors cursor-pointer"
                          >
                            <span className="material-symbols-outlined text-[14px]">close</span>
                          </button>
                        </DarkroomTooltip>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            /* Contact-Sheet Project Grid (DaVinci Resolve / Capture One / Figma Recents) */
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {/* Unified Optical Viewfinder "Open Folder | Open Photo(s)" Tile (always first in All view) */}
              {showOpenProjectTile && (
                <div className="relative rounded-lg overflow-hidden bg-surface-low/90 border border-hairline hover:border-hairline-strong transition-colors flex flex-col">
                  <div className="relative aspect-[16/10] bg-surface-lowest/90 grid grid-cols-2 divide-x divide-hairline/70 overflow-hidden">
                    <button
                      type="button"
                      disabled={isScanningFolder}
                      onClick={() => void openShootFolder()}
                      title={`Open a shoot folder (${openShortcutLabel})`}
                      className="group/folder relative flex flex-col items-center justify-center gap-2 px-2 hover:bg-surface/75 transition-colors cursor-pointer disabled:opacity-50 focus-visible:outline-none focus-visible:bg-surface/75"
                    >
                      {/* Left Viewfinder Corner Reticles */}
                      <span className="pointer-events-none absolute top-3 left-3 w-3 h-3 border-t border-l border-hairline-strong group-hover/folder:border-amber-core transition-colors" />
                      <span className="pointer-events-none absolute bottom-3 left-3 w-3 h-3 border-b border-l border-hairline-strong group-hover/folder:border-amber-core transition-colors" />

                      <div className="w-9 h-9 rounded-lg bg-surface border border-hairline-strong group-hover/folder:border-amber-core/70 group-hover/folder:text-amber-core text-ink-secondary flex items-center justify-center transition-colors">
                        <span className="material-symbols-outlined text-[20px]">
                          create_new_folder
                        </span>
                      </div>
                      <span className="font-display text-[11.5px] font-semibold text-ink-primary group-hover/folder:text-amber-soft transition-colors whitespace-nowrap">
                        Open Folder
                      </span>
                    </button>

                    <button
                      type="button"
                      disabled={isScanningFolder}
                      onClick={() => void openPhotoFiles()}
                      title={`Open one or more photos (${openPhotosShortcutLabel})`}
                      className="group/photo relative flex flex-col items-center justify-center gap-2 px-2 hover:bg-surface/75 transition-colors cursor-pointer disabled:opacity-50 focus-visible:outline-none focus-visible:bg-surface/75"
                    >
                      {/* Right Viewfinder Corner Reticles */}
                      <span className="pointer-events-none absolute top-3 right-3 w-3 h-3 border-t border-r border-hairline-strong group-hover/photo:border-amber-core transition-colors" />
                      <span className="pointer-events-none absolute bottom-3 right-3 w-3 h-3 border-b border-r border-hairline-strong group-hover/photo:border-amber-core transition-colors" />

                      <div className="w-9 h-9 rounded-lg bg-surface border border-hairline-strong group-hover/photo:border-amber-core/70 group-hover/photo:text-amber-core text-ink-secondary flex items-center justify-center transition-colors">
                        <span className="material-symbols-outlined text-[20px]">
                          add_photo_alternate
                        </span>
                      </div>
                      <span className="font-display text-[11.5px] font-semibold text-ink-primary group-hover/photo:text-amber-soft transition-colors whitespace-nowrap">
                        Open Photo(s)
                      </span>
                    </button>
                  </div>

                  <div className="px-3 py-2 border-t border-hairline flex-1 flex flex-col justify-center gap-1">
                    <button
                      type="button"
                      tabIndex={-1}
                      disabled={isScanningFolder}
                      onClick={() => void openShootFolder()}
                      className="flex items-center justify-between gap-2 text-left text-ink-telemetry hover:text-amber-soft transition-colors cursor-pointer disabled:opacity-50"
                    >
                      <span className="tabular-nums-mono text-[10px] truncate">
                        Folder · Full Shoot &amp; AI
                      </span>
                      <kbd className="tabular-nums-mono text-[9.5px] px-1.5 py-0.5 rounded bg-canvas border border-hairline text-ink-telemetry shrink-0">
                        {openShortcutLabel}
                      </kbd>
                    </button>
                    <button
                      type="button"
                      tabIndex={-1}
                      disabled={isScanningFolder}
                      onClick={() => void openPhotoFiles()}
                      className="flex items-center justify-between gap-2 text-left text-ink-telemetry hover:text-amber-soft transition-colors cursor-pointer disabled:opacity-50"
                    >
                      <span className="tabular-nums-mono text-[10px] truncate">
                        Photo(s) · Single or Multi
                      </span>
                      <kbd className="tabular-nums-mono text-[9.5px] px-1.5 py-0.5 rounded bg-canvas border border-hairline text-ink-telemetry shrink-0">
                        {openPhotosShortcutLabel}
                      </kbd>
                    </button>
                  </div>
                </div>
              )}

              {/* User's Real Recent Shoot Cards */}
              {visibleRecents.map((project) => {
                const isOpenNow = activeFolderKey === project.id;
                return (
                  <article
                    key={project.id}
                    onClick={() => handleOpenRecent(project)}
                    className={`group rounded-lg overflow-hidden bg-surface border transition-colors cursor-pointer flex flex-col ${isOpenNow
                        ? 'border-amber-core ring-1 ring-amber-core/30'
                        : 'border-hairline hover:border-hairline-strong'
                      }`}
                  >
                    <div className="relative aspect-[16/10] bg-surface-lowest overflow-hidden">
                      <ContactSheetCover
                        thumbnails={project.coverThumbnails}
                        projectName={project.projectName}
                      />

                      {/* Crisp Contact-Sheet Telemetry Overlays */}
                      <div className="absolute bottom-2 left-2 z-10 flex items-center gap-1.5">
                        <span className="px-1.5 py-0.5 rounded bg-canvas/85 backdrop-blur border border-white/10 tabular-nums-mono text-[9.5px] text-ink-primary">
                          {project.photoCount.toLocaleString()} FR
                        </span>
                        {project.isClustered && (
                          <span className="px-1.5 py-0.5 rounded bg-canvas/85 backdrop-blur border border-emerald-focus/45 tabular-nums-mono text-[9.5px] text-emerald-bright">
                            {project.clusterCount} SUBJ
                          </span>
                        )}
                        {isOpenNow && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-core text-amber-ink font-display text-[9px] font-bold uppercase tracking-wider">
                            OPEN
                          </span>
                        )}
                      </div>

                      {/* Pinned Status Ribbon (shown on cover only when pinned) */}
                      {project.isPinned && (
                        <div className="absolute top-2 right-2 z-10 px-1.5 py-0.5 rounded bg-amber-core text-amber-ink flex items-center gap-1 shadow-sm">
                          <span className="material-symbols-outlined text-[11px]">keep</span>
                          <span className="tabular-nums-mono text-[9px] font-bold uppercase tracking-wider">
                            Pinned
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="px-3 py-2.5 border-t border-hairline flex flex-col gap-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-display text-[12.5px] font-bold text-ink-primary truncate group-hover:text-amber-soft transition-colors">
                          {project.projectName}
                        </span>
                        <div
                          className="flex items-center gap-0.5 shrink-0"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <DarkroomTooltip
                            content={project.isPinned ? 'Unpin project' : 'Pin project'}
                            side="top"
                          >
                            <button
                              type="button"
                              onClick={() => togglePinRecentProjectEntry(project.folderPath)}
                              aria-label={project.isPinned ? 'Unpin project' : 'Pin project'}
                              className={`w-6 h-6 rounded flex items-center justify-center transition-colors cursor-pointer ${project.isPinned
                                  ? 'text-amber-core bg-amber-core/15 hover:bg-amber-core/25'
                                  : 'text-ink-telemetry hover:text-amber-soft hover:bg-surface-active'
                                }`}
                            >
                              <span className="material-symbols-outlined text-[14px]">keep</span>
                            </button>
                          </DarkroomTooltip>
                          <DarkroomTooltip content="Remove from recents" side="top">
                            <button
                              type="button"
                              onClick={() => removeRecentProjectEntry(project.folderPath)}
                              aria-label={`Remove ${project.projectName}`}
                              className="w-6 h-6 rounded text-ink-telemetry/70 hover:text-crimson-clip hover:bg-surface-active flex items-center justify-center transition-colors cursor-pointer"
                            >
                              <span className="material-symbols-outlined text-[14px]">close</span>
                            </button>
                          </DarkroomTooltip>
                        </div>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <span className="tabular-nums-mono text-[10px] text-ink-telemetry truncate">
                          {project.folderPath}
                        </span>
                        <span className="tabular-nums-mono text-[10px] text-ink-telemetry shrink-0">
                          {formatRelativeOpenedAt(project.lastOpenedAtIso)}
                        </span>
                      </div>
                    </div>
                  </article>
                );
              })}

              {/* Filtered Empty State (only when a specific sub-filter or search has 0 matches) */}
              {visibleRecents.length === 0 && !showOpenProjectTile && (
                <div className="col-span-full py-16 flex flex-col items-center justify-center text-center gap-2">
                  <span className="material-symbols-outlined text-[24px] text-ink-telemetry">
                    {filterTab === 'pinned' ? 'keep' : 'filter_list_off'}
                  </span>
                  <span className="font-display text-[13px] font-semibold text-ink-primary">
                    {emptyStateCopy[filterTab].title}
                  </span>
                  <span className="text-[12px] text-ink-telemetry max-w-[320px]">
                    {emptyStateCopy[filterTab].hint}
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setFilterTab('all');
                      setSearchQuery('');
                    }}
                    className="mt-1 tabular-nums-mono text-[11px] text-amber-soft hover:underline cursor-pointer"
                  >
                    Back to Recent Projects
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
