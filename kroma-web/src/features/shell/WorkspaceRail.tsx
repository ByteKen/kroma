import { AnimatePresence, motion } from 'framer-motion';
import { memo, useMemo, useState } from 'react';
import { useStudio, type WorkspaceView } from '../../app/StudioContext';
import { isRecipeModified } from '../../domain/develop';
import { isFaceOnlySlateCode } from '../../domain/identity';
import { formatShortcutCombo } from '../../domain/platform';
import { DarkroomTooltip, FAST_FADE, SOFT_SPRING, TACTILE_SPRING } from '../darkroom-kit';

interface NavItem {
  readonly id: WorkspaceView;
  readonly label: string;
  readonly icon: string;
  readonly badge?: string;
}

const WORKSPACE_ITEMS: readonly NavItem[] = [
  { id: 'develop', label: 'Studio', icon: 'tune', badge: '1' },
  { id: 'culling', label: 'Face Organizer', icon: 'filter_vintage', badge: '2' },
  { id: 'slate_ocr', label: 'Slate OCR', icon: 'document_scanner', badge: '3' },
  { id: 'review_queue', label: 'Needs Review', icon: 'fact_check' },
  { id: 'export', label: 'Export', icon: 'account_tree', badge: '4' },
];

const SINGLE_PHOTO_WORKSPACE_ITEMS: readonly NavItem[] = [
  { id: 'develop', label: 'Studio', icon: 'tune', badge: '1' },
  { id: 'export', label: 'Export', icon: 'account_tree', badge: '2' },
];

export const WorkspaceRail = memo(function WorkspaceRail() {
  const {
    session,
    activeWorkspace,
    settingsOpen,
    setActiveWorkspace,
    toggleSettings,
    ambiguousReviewQueue,
    slateOcrResults,
  } = useStudio();
  const [expanded, setExpanded] = useState(false);

  const { editedCount, subjectCount, namedSlateCount } = useMemo(() => {
    if (!session) return { editedCount: 0, subjectCount: 0, namedSlateCount: 0 };
    let edited = 0;
    let subjects = 0;
    let namedSlates = 0;
    for (const cluster of session.clusters) {
      if (session.isClustered && !cluster.anchor.id.startsWith('_')) {
        subjects += 1;
        if (
          !isFaceOnlySlateCode(cluster.anchor.slateCode) ||
          cluster.photos.some((p) => p.isSlateAnchor)
        ) {
          namedSlates += 1;
        }
      }
      for (const photo of cluster.photos) {
        if (isRecipeModified(photo.developRecipe)) {
          edited += 1;
        }
      }
    }
    return { editedCount: edited, subjectCount: subjects, namedSlateCount: namedSlates };
  }, [session]);

  const reviewCount = ambiguousReviewQueue.length;
  const liveSlateReadCount = useMemo(
    () => slateOcrResults.filter((r) => Boolean(r.proposedName)).length,
    [slateOcrResults],
  );

  const getCollapsedTooltip = (item: NavItem): string => {
    if (!session) return item.label;
    switch (item.id) {
      case 'develop':
        return editedCount > 0 ? `${item.label} • ${editedCount} edited` : item.label;
      case 'culling':
        return session.isClustered ? `${item.label} • ${subjectCount} subjects` : item.label;
      case 'slate_ocr': {
        if (namedSlateCount > 0 && subjectCount > 0) {
          return `${item.label} • ${namedSlateCount}/${subjectCount} named`;
        }
        if (liveSlateReadCount > 0) {
          return `${item.label} • ${liveSlateReadCount} read`;
        }
        return item.label;
      }
      case 'review_queue':
        if (reviewCount > 0) return `${item.label} • ${reviewCount}`;
        return session.isClustered ? `${item.label} • Clear` : item.label;
      default:
        return item.label;
    }
  };

  return (
    <motion.aside
      initial={false}
      animate={{ width: expanded ? 216 : 56 }}
      transition={SOFT_SPRING}
      className="shrink-0 bg-surface-low border-r border-hairline flex flex-col justify-between py-2.5 px-2 overflow-hidden z-40"
    >
      <div className="flex flex-col gap-2">
        {/* Collapse/Expand Toggle */}
        <div className="flex items-center justify-between px-1 pb-1">
          <AnimatePresence initial={false}>
            {expanded && (
              <motion.span
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -6 }}
                transition={FAST_FADE}
                className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry whitespace-nowrap"
              >
                Workspaces
              </motion.span>
            )}
          </AnimatePresence>
          <DarkroomTooltip content={expanded ? 'Collapse rail' : 'Expand rail'} side="right">
            <motion.button
              type="button"
              whileTap={{ scale: 0.92 }}
              onClick={() => setExpanded((prev) => !prev)}
              className="w-8 h-7 mx-auto rounded-md text-ink-secondary hover:bg-surface-active hover:text-ink-primary flex items-center justify-center transition-colors cursor-pointer shrink-0"
              aria-label={expanded ? 'Collapse rail' : 'Expand rail'}
            >
              <span className="material-symbols-outlined text-[18px]">
                {expanded ? 'left_panel_close' : 'left_panel_open'}
              </span>
            </motion.button>
          </DarkroomTooltip>
        </div>

        {/* Workspace Navigation */}
        <nav className="flex flex-col gap-1" aria-label="Studio Workspaces">
          {(session?.sessionMode === 'single' ? SINGLE_PHOTO_WORKSPACE_ITEMS : WORKSPACE_ITEMS).map((item) => {
            const isActive = Boolean(session) && activeWorkspace === item.id;
            const isDisabled = !session;
            const alertCount = item.id === 'review_queue' ? reviewCount : 0;

            const collapsedDotColor =
              !expanded && session
                ? item.id === 'develop' && editedCount > 0
                  ? 'bg-amber-core'
                  : item.id === 'culling' && session.isClustered
                    ? 'bg-emerald-focus'
                    : item.id === 'slate_ocr' &&
                      (namedSlateCount > 0 || slateOcrResults.length > 0)
                      ? namedSlateCount > 0
                        ? 'bg-emerald-focus'
                        : 'bg-amber-core'
                      : item.id === 'review_queue' && alertCount === 0 && session.isClustered
                        ? 'bg-emerald-focus'
                        : null
                : null;

            return (
              <DarkroomTooltip
                key={item.id}
                content={!expanded ? getCollapsedTooltip(item) : undefined}
                side="right"
                className="flex w-full"
              >
                <button
                  type="button"
                  disabled={isDisabled}
                  onClick={() => setActiveWorkspace(item.id)}
                  aria-label={item.label}
                  className={`relative w-full flex items-center gap-2.5 h-9 px-2 rounded-lg text-[12px] transition-colors cursor-pointer disabled:opacity-40 disabled:pointer-events-none ${isActive
                    ? 'text-amber-soft font-semibold'
                    : 'text-ink-secondary hover:bg-surface hover:text-ink-primary'
                    } ${expanded ? 'justify-start' : 'justify-center'}`}
                >
                  {isActive && (
                    <motion.span
                      layoutId="workspace-rail-active-bg"
                      transition={TACTILE_SPRING}
                      className="absolute inset-0 rounded-lg bg-surface-active border border-amber-core/45"
                    />
                  )}
                  {isActive && (
                    <motion.span
                      layoutId="workspace-rail-active-bar"
                      transition={TACTILE_SPRING}
                      className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-amber-core"
                    />
                  )}

                  <span className="relative z-10 material-symbols-outlined text-[19px] shrink-0">
                    {item.icon}
                  </span>

                  {expanded && (
                    <span className="relative z-10 truncate flex-1 text-left whitespace-nowrap">
                      {item.label}
                    </span>
                  )}

                  {expanded && item.badge && (
                    <span className="relative z-10 tabular-nums-mono text-[9px] px-1.5 py-0.5 rounded bg-amber-core/15 text-amber-soft border border-amber-core/30 shrink-0">
                      {item.badge}
                    </span>
                  )}

                  {expanded && alertCount > 0 && (
                    <span className="relative z-10 tabular-nums-mono text-[10px] px-1.5 py-0.5 rounded-full bg-tangerine-alert/20 text-tangerine-alert shrink-0">
                      {alertCount}
                    </span>
                  )}

                  {!expanded && alertCount > 0 && (
                    <span
                      aria-hidden="true"
                      className="pointer-events-none absolute top-0.5 right-0.5 z-20 min-w-[15px] h-[15px] px-1 rounded-full bg-tangerine-alert text-amber-ink font-bold tabular-nums-mono text-[8.5px] leading-none flex items-center justify-center ring-2 ring-surface-low"
                    >
                      {alertCount > 99 ? '99+' : alertCount}
                    </span>
                  )}

                  {!expanded && alertCount === 0 && collapsedDotColor && (
                    <span
                      aria-hidden="true"
                      className={`pointer-events-none absolute top-1.5 right-1.5 z-20 w-2 h-2 rounded-full ring-2 ring-surface-low ${collapsedDotColor}`}
                    />
                  )}
                </button>
              </DarkroomTooltip>
            );
          })}
        </nav>
      </div>

      {/* Bottom Utility Area: Settings Modal Trigger */}
      <div className="flex flex-col gap-2">
        <DarkroomTooltip
          content={!expanded ? `Settings (${formatShortcutCombo('Ctrl+,')})` : undefined}
          side="right"
          className="flex w-full"
        >
          <button
            type="button"
            onClick={toggleSettings}
            aria-label="Settings"
            aria-haspopup="dialog"
            aria-expanded={settingsOpen}
            className={`relative w-full flex items-center gap-2.5 h-9 px-2 rounded-lg text-[12px] transition-colors cursor-pointer ${settingsOpen
              ? 'bg-surface-active border border-amber-core/45 text-amber-soft font-semibold'
              : 'text-ink-secondary hover:bg-surface hover:text-ink-primary'
              } ${expanded ? 'justify-start' : 'justify-center'}`}
          >
            <span className="relative z-10 material-symbols-outlined text-[19px] shrink-0">
              settings
            </span>
            {expanded && (
              <span className="relative z-10 truncate flex-1 text-left whitespace-nowrap">
                Settings
              </span>
            )}
          </button>
        </DarkroomTooltip>
      </div>
    </motion.aside>
  );
});
