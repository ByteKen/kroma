import { useEffect, useMemo, useRef, useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import {
  buildUnorganizedShootTimeline,
  type ShootTimelinePhoto,
  type ShootTimelineSortMode,
} from '../../domain/identity';
import {
  DarkroomButton,
  TelemetryPill,
  VirtualGrid,
  type VirtualGridHandle,
} from '../darkroom-kit';

const SORT_MODES: readonly { id: ShootTimelineSortMode; label: string }[] = [
  { id: 'filename', label: 'Filename' },
  { id: 'timestamp', label: 'Timestamp' },
  { id: 'nearest', label: 'Nearest' },
];

/**
 * Unobstructed Side-by-Side Manual Match Workspace:
 * - Left pane keeps the active subject's full portrait clearly visible at all times.
 * - Right pane opens the entire unorganized shoot folder sorted by filename / capture timestamp,
 *   auto-scrolled directly to the active subject's frame neighborhood for rapid multi-select matching.
 */
export function ManualMatchSplitView() {
  const {
    session,
    activeCluster,
    ambiguousReviewQueue,
    movePhotosToCluster,
    selectNextSingletonCluster,
    setManualMatchSplitOpen,
    openPhotoInDevelop,
  } = useStudio();

  const [sortMode, setSortMode] = useState<ShootTimelineSortMode>('filename');
  const [folderFilterId, setFolderFilterId] = useState<string | null>(null);
  const [onlySingletons, setOnlySingletons] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [pickedIds, setPickedIds] = useState<Set<string>>(new Set());
  const [lastClickedIdx, setLastClickedIdx] = useState<number | null>(null);
  const [referencePhotoId, setReferencePhotoId] = useState<string | null>(null);

  const gridRef = useRef<VirtualGridHandle | null>(null);
  const lastScrolledClusterId = useRef<string | null>(null);

  const activeClusterId = activeCluster?.anchor.id ?? '';

  // Reset multi-select and reference frame when switching to a different active subject
  useEffect(() => {
    setPickedIds(new Set());
    setLastClickedIdx(null);
    setReferencePhotoId(null);
  }, [activeClusterId]);

  const timeline = useMemo<readonly ShootTimelinePhoto[]>(() => {
    if (!session || !activeCluster) return [];
    const reviewPhotos = ambiguousReviewQueue.map((item) => item.photo);
    return buildUnorganizedShootTimeline(
      session.clusters,
      activeCluster.anchor.id,
      sortMode,
      reviewPhotos,
    );
  }, [session, activeCluster, ambiguousReviewQueue, sortMode]);

  const visibleTimeline = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return timeline.filter((item) => {
      if (folderFilterId && item.photo.directoryId !== folderFilterId) {
        return false;
      }
      if (onlySingletons && !item.isInActiveCluster && item.clusterPhotoCount > 1) {
        return false;
      }
      if (q.length > 0) {
        const hay = `${item.photo.fileName} ${item.clusterName} ${item.photo.capturedAtLabel ?? ''}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [timeline, folderFilterId, onlySingletons, searchQuery]);

  const anchorPhotoId = activeCluster?.anchor.anchorPhotoId ?? activeCluster?.photos[0]?.id ?? null;

  const scrollToAnchor = (smooth: boolean) => {
    if (!anchorPhotoId) return;
    const idx = visibleTimeline.findIndex((item) => item.photo.id === anchorPhotoId);
    if (idx >= 0) {
      gridRef.current?.scrollToIndex(idx, smooth);
    }
  };

  // Auto-scroll right timeline directly to the active subject's chronological spot
  useEffect(() => {
    if (!activeClusterId || lastScrolledClusterId.current === activeClusterId) return;
    const frame = window.requestAnimationFrame(() => {
      scrollToAnchor(false);
      lastScrolledClusterId.current = activeClusterId;
    });
    return () => window.cancelAnimationFrame(frame);
  });

  // Pressing Enter confirms the current multi-selection match
  useEffect(() => {
    if (!activeCluster || pickedIds.size === 0) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      ) {
        return;
      }
      if (event.key === 'Enter') {
        event.preventDefault();
        const ids = Array.from(pickedIds);
        setPickedIds(new Set());
        void movePhotosToCluster(ids, activeCluster.anchor.id);
      } else if (event.key === 'Escape') {
        event.preventDefault();
        setPickedIds(new Set());
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeCluster, pickedIds, movePhotosToCluster]);

  if (!session || !activeCluster) return null;

  const refPhoto =
    activeCluster.photos.find((p) => p.id === referencePhotoId) ?? activeCluster.photos[0] ?? null;
  const singletonCount = session.clusters.filter(
    (c) => !c.anchor.id.startsWith('_') && c.photos.length === 1,
  ).length;
  const dirNameById = new Map(session.directories.map((d) => [d.id, d.name]));

  const togglePick = (item: ShootTimelinePhoto, index: number, shiftKey: boolean) => {
    if (item.isInActiveCluster) return;

    setPickedIds((prev) => {
      const next = new Set(prev);
      if (shiftKey && lastClickedIdx !== null) {
        const start = Math.min(lastClickedIdx, index);
        const end = Math.max(lastClickedIdx, index);
        for (let i = start; i <= end; i++) {
          const entry = visibleTimeline[i];
          if (entry && !entry.isInActiveCluster) {
            next.add(entry.photo.id);
          }
        }
      } else if (next.has(item.photo.id)) {
        next.delete(item.photo.id);
      } else {
        next.add(item.photo.id);
      }
      return next;
    });
    setLastClickedIdx(index);
  };

  const commitManualMatches = async () => {
    if (pickedIds.size === 0) return;
    const ids = Array.from(pickedIds);
    setPickedIds(new Set());
    setLastClickedIdx(null);
    await movePhotosToCluster(ids, activeCluster.anchor.id);
  };

  return (
    <div className="flex-1 min-h-0 flex overflow-hidden bg-canvas">
      {/* LEFT PANE: Full Unobstructed Active Subject Reference */}
      <aside
        aria-label="Active Subject Reference"
        className="w-[310px] xl:w-[370px] shrink-0 bg-surface-low border-r border-hairline flex flex-col overflow-hidden"
      >
        <div className="px-4 py-2.5 bg-surface border-b border-hairline flex items-center justify-between gap-2 shrink-0">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-display text-[14px] font-bold text-ink-primary truncate">
                {activeCluster.anchor.subjectName}
              </span>
              <TelemetryPill tone="emerald">{activeCluster.photos.length} matched</TelemetryPill>
            </div>
            {refPhoto && (
              <span className="tabular-nums-mono text-[10px] text-ink-telemetry block truncate">
                {refPhoto.fileName}
                {refPhoto.capturedAtLabel ? ` • ${refPhoto.capturedAtLabel}` : ''}
              </span>
            )}
          </div>

          <DarkroomButton
            variant="subtle"
            size="xs"
            icon="close"
            onClick={() => setManualMatchSplitOpen(false)}
            title="Exit match split"
          >
            Done
          </DarkroomButton>
        </div>

        {/* Full Uncropped Reference Portrait */}
        <div className="flex-1 min-h-0 p-3 flex flex-col gap-2.5 overflow-hidden">
          <div className="flex-1 min-h-0 rounded-xl bg-surface-lowest border border-amber-core/40 overflow-hidden flex items-center justify-center relative">
            {refPhoto ? (
              <img
                src={refPhoto.imageUrl}
                alt={refPhoto.fileName}
                className="w-full h-full object-contain"
              />
            ) : (
              <span className="text-[12px] text-ink-telemetry">No photo selected</span>
            )}

            {refPhoto && (
              <button
                type="button"
                onClick={() => openPhotoInDevelop(refPhoto.id, activeCluster.anchor.id)}
                className="absolute bottom-2.5 right-2.5 px-2 py-1 rounded-md bg-surface-lowest/85 backdrop-blur border border-hairline hover:border-amber-core text-[10px] tabular-nums-mono text-ink-secondary hover:text-amber-soft cursor-pointer transition-colors"
              >
                Open in Develop
              </button>
            )}
          </div>

          {/* Existing Photos Already in This Subject */}
          {activeCluster.photos.length > 1 && (
            <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5 shrink-0">
              {activeCluster.photos.map((photo) => {
                const isActiveRef = photo.id === refPhoto?.id;
                return (
                  <button
                    key={photo.id}
                    type="button"
                    onClick={() => setReferencePhotoId(photo.id)}
                    title={photo.fileName}
                    className={`w-12 h-15 rounded overflow-hidden shrink-0 border transition-colors cursor-pointer bg-surface-lowest ${
                      isActiveRef
                        ? 'border-amber-core ring-1 ring-amber-core/60'
                        : 'border-hairline hover:border-hairline-strong opacity-80 hover:opacity-100'
                    }`}
                  >
                    <img
                      src={photo.imageUrl}
                      alt={photo.fileName}
                      className="w-full h-full object-cover"
                    />
                  </button>
                );
              })}
            </div>
          )}

          {/* Quick Singleton Sweep Footer */}
          <div className="pt-1 border-t border-hairline flex items-center justify-between gap-2 shrink-0">
            <span className="tabular-nums-mono text-[10px] text-ink-telemetry">
              {singletonCount} single-photo left
            </span>
            <DarkroomButton
              variant="secondary"
              size="xs"
              icon="skip_next"
              disabled={singletonCount === 0}
              onClick={selectNextSingletonCluster}
              title="Next 1-photo subject"
            >
              Next 1-Photo
            </DarkroomButton>
          </div>
        </div>
      </aside>

      {/* RIGHT PANE: Unorganized Shoot Browser (Sorted by Filename / Timestamp) */}
      <div className="flex-1 min-w-0 flex flex-col overflow-hidden bg-canvas">
        {/* Timeline Controls Toolbar */}
        <div className="px-4 py-2.5 bg-surface-low border-b border-hairline flex flex-wrap items-center justify-between gap-2 shrink-0">
          <div className="flex items-center flex-wrap gap-1.5">
            <span className="font-display text-[12px] font-semibold text-ink-primary mr-1">
              All Project Photos
            </span>

            {/* Subfolder Filter Pills */}
            {session.directories.length > 1 && (
              <div className="flex items-center gap-1 bg-canvas p-0.5 rounded-lg border border-hairline">
                <button
                  type="button"
                  onClick={() => setFolderFilterId(null)}
                  className={`px-2 py-0.5 rounded text-[10px] tabular-nums-mono cursor-pointer transition-colors ${
                    folderFilterId === null
                      ? 'bg-surface-active text-amber-soft font-semibold'
                      : 'text-ink-telemetry hover:text-ink-primary'
                  }`}
                >
                  All
                </button>
                {session.directories.map((dir) => (
                  <button
                    key={dir.id}
                    type="button"
                    onClick={() => setFolderFilterId(folderFilterId === dir.id ? null : dir.id)}
                    className={`px-2 py-0.5 rounded text-[10px] tabular-nums-mono cursor-pointer transition-colors ${
                      folderFilterId === dir.id
                        ? 'bg-surface-active text-amber-soft font-semibold'
                        : 'text-ink-telemetry hover:text-ink-primary'
                    }`}
                  >
                    {dir.name}
                  </button>
                ))}
              </div>
            )}

            {/* Sort Order Selector */}
            <div className="flex items-center gap-0.5 bg-canvas p-0.5 rounded-lg border border-hairline">
              {SORT_MODES.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setSortMode(m.id)}
                  className={`px-2 py-0.5 rounded text-[10px] tabular-nums-mono cursor-pointer transition-colors ${
                    sortMode === m.id
                      ? 'bg-surface-active text-ink-primary font-semibold'
                      : 'text-ink-telemetry hover:text-ink-primary'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <DarkroomButton
              variant={onlySingletons ? 'emerald' : 'subtle'}
              size="xs"
              icon="filter_1"
              onClick={() => setOnlySingletons((prev) => !prev)}
              title="Show 1-photo subjects only"
            >
              1-Photo Only
            </DarkroomButton>

            <DarkroomButton
              variant="subtle"
              size="xs"
              icon="my_location"
              onClick={() => scrollToAnchor(true)}
              title="Jump to anchor frame"
            >
              Jump to Frame
            </DarkroomButton>
          </div>

          <div className="relative w-44 xl:w-56">
            <span className="material-symbols-outlined text-[14px] text-ink-telemetry absolute left-2 top-1/2 -translate-y-1/2 pointer-events-none">
              search
            </span>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter filename..."
              className="w-full pl-7 pr-2 py-1 rounded-md bg-canvas border border-hairline focus:border-amber-core text-[11px] tabular-nums-mono text-ink-primary placeholder:text-ink-telemetry outline-none"
            />
          </div>
        </div>

        {/* Virtualized Multi-Select Chronological Photo Grid */}
        {visibleTimeline.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-[12px] text-ink-telemetry">
            No photos match this filter.
          </div>
        ) : (
          <VirtualGrid
            ref={gridRef}
            items={visibleTimeline}
            getKey={(item) => item.photo.id}
            minItemWidth={170}
            footerHeightPx={46}
            gapPx={12}
            paddingPx={16}
            renderItem={(item, idx) => {
              const isPicked = pickedIds.has(item.photo.id);
              const folderName = dirNameById.get(item.photo.directoryId);

              return (
                <div
                  onClick={(e) => togglePick(item, idx, e.shiftKey)}
                  className={`group rounded-lg overflow-hidden border transition-colors flex flex-col select-none ${
                    item.isInActiveCluster
                      ? 'bg-surface border-emerald-focus/75 ring-1 ring-emerald-focus/35 cursor-default'
                      : isPicked
                        ? 'bg-surface-active border-amber-core ring-2 ring-amber-core/50 cursor-pointer'
                        : 'bg-surface border-hairline hover:border-hairline-strong cursor-pointer'
                  }`}
                >
                  <div className="relative aspect-[3/4] bg-surface-lowest overflow-hidden">
                    <img
                      src={item.photo.imageUrl}
                      alt={item.photo.fileName}
                      loading="lazy"
                      decoding="async"
                      className="w-full h-full object-cover transition-transform duration-150 group-hover:scale-[1.02]"
                    />

                    {/* Selection Checkbox / Matched Indicator */}
                    <div className="absolute top-2 left-2 z-10">
                      {item.isInActiveCluster ? (
                        <span className="px-1.5 py-0.5 rounded bg-emerald-focus text-canvas font-display text-[9px] font-bold uppercase tracking-wider">
                          In Subject
                        </span>
                      ) : (
                        <span
                          className={`w-5 h-5 rounded flex items-center justify-center border transition-colors ${
                            isPicked
                              ? 'bg-amber-core border-amber-core text-amber-ink'
                              : 'bg-surface-lowest/80 border-hairline-strong group-hover:border-amber-core'
                          }`}
                        >
                          {isPicked && (
                            <span className="material-symbols-outlined text-[14px] font-bold">
                              check
                            </span>
                          )}
                        </span>
                      )}
                    </div>

                    {/* Proximity / Current Group Badge */}
                    <div className="absolute top-2 right-2 z-10 flex flex-col items-end gap-1">
                      {!item.isInActiveCluster && item.proximityLabel && (
                        <span className="px-1.5 py-0.5 rounded bg-emerald-focus/90 text-canvas tabular-nums-mono text-[9px] font-semibold">
                          {item.proximityLabel}
                        </span>
                      )}
                      {!item.isInActiveCluster && (
                        <span className="px-1.5 py-0.5 rounded bg-surface-lowest/85 backdrop-blur border border-hairline tabular-nums-mono text-[9px] text-ink-secondary max-w-[115px] truncate">
                          {item.clusterName}
                          {item.clusterPhotoCount === 1 ? ' • 1p' : ` • ${item.clusterPhotoCount}p`}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="px-2 py-1.5 bg-surface border-t border-hairline flex flex-col">
                    <span className="tabular-nums-mono text-[11px] font-medium text-ink-primary truncate">
                      {item.photo.fileName}
                    </span>
                    <span className="tabular-nums-mono text-[9px] text-ink-telemetry truncate">
                      {[folderName, item.photo.capturedAtLabel].filter(Boolean).join(' • ') ||
                        item.photo.exif.shutterSpeed}
                    </span>
                  </div>
                </div>
              );
            }}
          />
        )}

        {/* Sticky Multi-Select Action Bar */}
        <div className="px-4 py-2.5 bg-surface border-t border-hairline flex items-center justify-between gap-3 shrink-0">
          <span className="tabular-nums-mono text-[11px] text-ink-secondary">
            {pickedIds.size === 0 ? (
              <>
                Click photos (or <kbd className="px-1 py-0.5 rounded bg-canvas border border-hairline text-[10px]">Shift+Click</kbd> a range) to match into{' '}
                <strong className="text-ink-primary">{activeCluster.anchor.subjectName}</strong>
              </>
            ) : (
              <>
                <strong className="text-amber-soft">{pickedIds.size}</strong> photo
                {pickedIds.size === 1 ? '' : 's'} selected to match into{' '}
                <strong className="text-ink-primary">{activeCluster.anchor.subjectName}</strong>
              </>
            )}
          </span>

          <div className="flex items-center gap-2 shrink-0">
            {pickedIds.size > 0 && (
              <DarkroomButton variant="subtle" size="sm" onClick={() => setPickedIds(new Set())}>
                Clear
              </DarkroomButton>
            )}
            <DarkroomButton
              variant="primary"
              size="sm"
              icon="person_add"
              disabled={pickedIds.size === 0}
              onClick={() => void commitManualMatches()}
            >
              <span>
                Add {pickedIds.size > 0 ? `${pickedIds.size} ` : ''}to{' '}
                {activeCluster.anchor.subjectName}
              </span>
            </DarkroomButton>
          </div>
        </div>
      </div>
    </div>
  );
}
