import { useEffect, useMemo, useRef, useState } from 'react';
import { useStudio, type AnchorSortMode } from '../../app/StudioContext';
import {
  filterSubjectClusters,
  type SubjectFilterMode,
} from '../../application/useCases/filterSubjectClusters';
import { isFaceOnlySlateCode } from '../../domain/identity';
import {
  formatVirtualFolderBreadcrumb,
  matchesVirtualFolderFilter,
} from '../../domain/virtualFolder';
import {
  DarkroomButton,
  DarkroomTooltip,
  PanelResizeHandle,
  TelemetryPill,
  useResizableWidth,
} from '../darkroom-kit';
import { GradedThumbnailImg } from '../develop/GradedThumbnailImg';
import { VirtualFolderTreePanel } from '../folders/VirtualFolderTreePanel';
import {
  beginKromaDrag,
  clearActiveKromaDrag,
  getActiveKromaDrag,
  readKromaDragFromDrop,
} from '../folders/kromaDragState';
import {
  SubjectHoverPreviewCard,
  useSubjectHoverPreview,
} from './SubjectHoverPreviewCard';

const SORT_LABELS: Record<AnchorSortMode, string> = {
  capture_time: 'Order',
  confidence: 'Confidence',
  exposures: 'Count',
};

const SORT_ORDER: readonly AnchorSortMode[] = ['capture_time', 'exposures', 'confidence'];

export function SlateAnchorColumn() {
  const {
    session,
    activeWorkspace,
    activeCluster,
    activeDirectoryId,
    activeVirtualFolderId,
    activeVirtualFolderAllowedIds,
    toggleDirectoryFilter,
    selectCluster,
    anchorSortMode,
    setAnchorSortMode,
    renameActiveSlate,
    movePhotosToCluster,
    mergeClustersById,
  } = useStudio();

  const [subjectQuery, setSubjectQuery] = useState('');
  const [subjectFilter, setSubjectFilter] = useState<SubjectFilterMode>('all');
  const [editingClusterId, setEditingClusterId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState('');
  const [dragOverClusterId, setDragOverClusterId] = useState<string | null>(null);
  const [selectedClusterIds, setSelectedClusterIds] = useState<ReadonlySet<string>>(new Set());
  const searchInputRef = useRef<HTMLInputElement>(null);
  const subjectsPanel = useResizableWidth({
    storageKey: 'culling-subjects',
    defaultWidth: 286,
    minWidth: 224,
    maxWidth: 440,
  });
  const {
    preview: hoverPreview,
    schedulePreview: scheduleHoverPreview,
    clearPreview: clearHoverPreview,
  } = useSubjectHoverPreview();

  // Reset subject search/filter/selection when switching projects
  useEffect(() => {
    setSubjectQuery('');
    setSubjectFilter('all');
    setSelectedClusterIds(new Set());
  }, [session?.folderPath]);

  const folderFilteredClusters = useMemo(() => {
    const clusters = session?.clusters ?? [];
    if (activeVirtualFolderId === 'all') return clusters;
    return clusters.filter((cluster) =>
      cluster.photos.some((p) =>
        matchesVirtualFolderFilter(
          p.virtualFolderId,
          activeVirtualFolderId,
          activeVirtualFolderAllowedIds,
        ),
      ),
    );
  }, [session?.clusters, activeVirtualFolderId, activeVirtualFolderAllowedIds]);

  const virtualFolderById = useMemo(
    () => new Map((session?.virtualFolders ?? []).map((f) => [f.id, f])),
    [session?.virtualFolders],
  );

  // Ctrl+K focuses Subjects search when Face Organizer is active
  useEffect(() => {
    if (activeWorkspace !== 'culling') return;
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeWorkspace]);

  const filterCounts = useMemo(() => {
    const clusters = folderFilteredClusters;
    let review = 0;
    let single = 0;
    let multi = 0;
    for (const c of clusters) {
      if (c.anchor.ambiguousCount > 0) review += 1;
      if (c.photos.length === 1) single += 1;
      else if (c.photos.length >= 2) multi += 1;
    }
    return { total: clusters.length, review, single, multi };
  }, [folderFilteredClusters]);

  const filteredClusters = useMemo(
    () =>
      filterSubjectClusters(folderFilteredClusters, {
        query: subjectQuery,
        filter: subjectFilter,
      }),
    [folderFilteredClusters, subjectQuery, subjectFilter],
  );

  if (!session) return null;

  const isFiltered =
    subjectQuery.trim().length > 0 ||
    subjectFilter !== 'all' ||
    activeVirtualFolderId !== 'all';

  const cycleSort = () => {
    const nextIdx = (SORT_ORDER.indexOf(anchorSortMode) + 1) % SORT_ORDER.length;
    setAnchorSortMode(SORT_ORDER[nextIdx]);
  };

  const beginEdit = (clusterId: string, currentName: string) => {
    clearHoverPreview();
    setEditingClusterId(clusterId);
    setDraftName(currentName);
  };

  const commitEdit = async (clusterId: string) => {
    if (draftName.trim().length > 0) {
      await renameActiveSlate(clusterId, draftName);
    }
    setEditingClusterId(null);
  };

  return (
    <>
      <section
        aria-label="Detected Subjects"
        style={{ width: `${subjectsPanel.width}px` }}
        className="shrink-0 bg-surface-low flex flex-col overflow-hidden"
      >
      {/* Virtual Department Folder Tree */}
      <VirtualFolderTreePanel />

      {/* Clean Column Header */}
      <div className="h-10 px-3 bg-surface border-b border-hairline flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-display text-[13px] font-semibold text-ink-primary truncate">
            Subjects
          </span>
          <TelemetryPill tone="amber">
            {isFiltered
              ? `${filteredClusters.length}/${session.clusters.length}`
              : session.clusters.length}
          </TelemetryPill>
          {selectedClusterIds.size > 1 && (
            <button
              type="button"
              onClick={() => setSelectedClusterIds(new Set())}
              title="Drag selected subjects onto a folder, or click to clear selection"
              className="px-1.5 py-0.5 rounded bg-emerald-focus/15 border border-emerald-focus/40 tabular-nums-mono text-[9.5px] text-emerald-bright hover:brightness-110 cursor-pointer shrink-0"
            >
              {selectedClusterIds.size} picked ×
            </button>
          )}
        </div>

        <DarkroomButton
          variant="subtle"
          size="xs"
          onClick={cycleSort}
          title="Change sort order"
          className="h-6 px-2 tabular-nums-mono text-[10px]"
        >
          <span>{SORT_LABELS[anchorSortMode]}</span>
          <span className="material-symbols-outlined text-[13px]">unfold_more</span>
        </DarkroomButton>
      </div>

      {/* Subject Search & Quick Filters */}
      {session.clusters.length > 1 && (
        <div className="px-2.5 py-2 border-b border-hairline bg-surface-low flex flex-col gap-1.5 shrink-0">
          <div className="h-7 px-2 rounded-md bg-canvas border border-hairline focus-within:border-amber-core transition-colors flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[14px] text-ink-telemetry shrink-0">
              search
            </span>
            <input
              ref={searchInputRef}
              type="text"
              value={subjectQuery}
              onChange={(e) => setSubjectQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape' && subjectQuery) {
                  e.stopPropagation();
                  setSubjectQuery('');
                }
              }}
              placeholder="Search subjects or file..."
              aria-label="Search subjects or filename"
              className="bg-transparent text-ink-primary placeholder:text-ink-telemetry text-[11px] flex-1 min-w-0 outline-none border-none"
            />
            {subjectQuery ? (
              <button
                type="button"
                onClick={() => setSubjectQuery('')}
                aria-label="Clear subject search"
                className="text-ink-telemetry hover:text-ink-primary cursor-pointer flex items-center shrink-0"
              >
                <span className="material-symbols-outlined text-[13px]">close</span>
              </button>
            ) : (
              <kbd className="tabular-nums-mono text-[9px] text-ink-telemetry/80 shrink-0">
                Ctrl+K
              </kbd>
            )}
          </div>

          <div className="flex items-center gap-1 overflow-x-auto">
            {(
              [
                { id: 'all', label: 'All', count: filterCounts.total },
                ...(filterCounts.review > 0
                  ? [{ id: 'review' as const, label: 'Review', count: filterCounts.review }]
                  : []),
                { id: 'single', label: '1 Photo', count: filterCounts.single },
                { id: 'multi', label: '2+', count: filterCounts.multi },
              ] as const
            ).map((chip) => {
              const active = subjectFilter === chip.id;
              return (
                <button
                  key={chip.id}
                  type="button"
                  onClick={() => setSubjectFilter(chip.id)}
                  className={`px-2 py-0.5 rounded-md tabular-nums-mono text-[10px] border transition-colors cursor-pointer shrink-0 flex items-center gap-1 ${
                    active
                      ? chip.id === 'review'
                        ? 'bg-tangerine-alert/15 border-tangerine-alert/50 text-tangerine-alert'
                        : 'bg-surface-active border-amber-core/50 text-amber-soft'
                      : 'bg-canvas border-hairline text-ink-telemetry hover:text-ink-primary'
                  }`}
                >
                  <span>{chip.label}</span>
                  {chip.id !== 'all' && <span className="opacity-75">{chip.count}</span>}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Compact Subfolder Filter Strip (only rendered if shoot has 2+ subfolders) */}
      {session.directories.length > 1 && (
        <div className="px-2.5 py-2 border-b border-hairline bg-surface-low flex items-center gap-1.5 overflow-x-auto shrink-0">
          <button
            type="button"
            onClick={() => {
              if (activeDirectoryId) toggleDirectoryFilter(activeDirectoryId);
            }}
            className={`px-2 py-0.5 rounded-md tabular-nums-mono text-[10px] border transition-colors cursor-pointer shrink-0 ${activeDirectoryId === null
                ? 'bg-surface-active border-amber-core/50 text-amber-soft'
                : 'bg-canvas border-hairline text-ink-telemetry hover:text-ink-primary'
              }`}
          >
            All
          </button>
          {session.directories.map((dir) => {
            const selected = activeDirectoryId === dir.id;
            return (
              <button
                key={dir.id}
                type="button"
                onClick={() => toggleDirectoryFilter(dir.id)}
                className={`px-2 py-0.5 rounded-md tabular-nums-mono text-[10px] border transition-colors cursor-pointer shrink-0 ${selected
                    ? 'bg-surface-active border-amber-core/50 text-amber-soft'
                    : 'bg-canvas border-hairline text-ink-telemetry hover:text-ink-primary'
                  }`}
              >
                {dir.name}
              </button>
            );
          })}
        </div>
      )}

      {/* Scrollable Subject List — shrink-0 on every card prevents vertical squashing */}
      <div
        onMouseLeave={clearHoverPreview}
        onDragEnd={clearHoverPreview}
        onScroll={clearHoverPreview}
        className="flex-1 min-h-0 overflow-y-auto p-2.5 flex flex-col gap-2"
      >
        {filteredClusters.length === 0 ? (
          <div className="py-10 px-3 flex flex-col items-center justify-center text-center gap-2 text-ink-telemetry">
            <span className="material-symbols-outlined text-[22px]">person_search</span>
            <span className="text-[11.5px] text-ink-secondary">
              No subjects match the current filter
            </span>
            <DarkroomButton
              variant="subtle"
              size="xs"
              onClick={() => {
                setSubjectQuery('');
                setSubjectFilter('all');
              }}
            >
              Reset Filter
            </DarkroomButton>
          </div>
        ) : (
          filteredClusters.map((cluster) => {
          const { anchor } = cluster;
          const isSelected = activeCluster?.anchor.id === anchor.id;
          const isMultiSelected = selectedClusterIds.has(anchor.id);
          const isEditing = editingClusterId === anchor.id;
          const hasAmbiguous = anchor.ambiguousCount > 0;
          const faceOnly = isFaceOnlySlateCode(anchor.slateCode);
          const isSpecialBucket = anchor.id.startsWith('_');
          const isDropTarget = dragOverClusterId === anchor.id;

          // Summarize assigned virtual folders for this subject
          const assignedFolderIds = new Set<string>();
          let matchedInActiveFolder = 0;
          for (const p of cluster.photos) {
            if (p.virtualFolderId && virtualFolderById.has(p.virtualFolderId)) {
              assignedFolderIds.add(p.virtualFolderId);
            }
            if (
              activeVirtualFolderId !== 'all' &&
              matchesVirtualFolderFilter(
                p.virtualFolderId,
                activeVirtualFolderId,
                activeVirtualFolderAllowedIds,
              )
            ) {
              matchedInActiveFolder += 1;
            }
          }
          const folderSummary =
            assignedFolderIds.size === 1
              ? (virtualFolderById.get(Array.from(assignedFolderIds)[0])?.name ?? null)
              : assignedFolderIds.size > 1
                ? `${assignedFolderIds.size} folders`
                : null;
          const folderTooltip =
            assignedFolderIds.size === 1
              ? formatVirtualFolderBreadcrumb(
                  session.virtualFolders,
                  Array.from(assignedFolderIds)[0],
                )
              : undefined;

          return (
            <article
              key={anchor.id}
              onClick={(e) => {
                if (e.ctrlKey || e.metaKey) {
                  setSelectedClusterIds((prev) => {
                    const next = new Set(prev);
                    if (next.has(anchor.id)) next.delete(anchor.id);
                    else next.add(anchor.id);
                    return next;
                  });
                  return;
                }
                setSelectedClusterIds(new Set([anchor.id]));
                selectCluster(anchor.id);
              }}
              onMouseEnter={(e) => {
                if (!isEditing) scheduleHoverPreview(cluster, e.currentTarget);
              }}
              className={`shrink-0 p-2 rounded-lg border relative overflow-hidden cursor-pointer transition-colors ${isDropTarget
                  ? 'bg-emerald-focus/15 border-emerald-bright ring-1 ring-emerald-bright/60'
                  : isMultiSelected && selectedClusterIds.size > 1
                    ? 'bg-emerald-focus/10 border-emerald-focus/60'
                    : isSelected
                      ? 'bg-surface-active border-amber-core/70'
                      : 'bg-surface border-hairline hover:bg-surface-elevated hover:border-hairline-strong'
                }`}
            >
              {isSelected && (
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-amber-core" />
              )}

              <div
                draggable={!isEditing}
                onDragStart={(e) => {
                  clearHoverPreview();
                  const dragIds =
                    selectedClusterIds.has(anchor.id) && selectedClusterIds.size > 1
                      ? Array.from(selectedClusterIds)
                      : [anchor.id];
                  beginKromaDrag(e.dataTransfer, { kind: 'clusters', clusterIds: dragIds });
                }}
                onDragEnd={() => {
                  clearActiveKromaDrag();
                  setDragOverClusterId(null);
                }}
                onDragEnter={(e) => {
                  if (!session.isClustered || isSpecialBucket) return;
                  const active = getActiveKromaDrag();
                  if (active && active.kind === 'vfolder') return;
                  e.preventDefault();
                }}
                onDragOver={(e) => {
                  if (!session.isClustered || isSpecialBucket) return;
                  const active = getActiveKromaDrag();
                  if (active && active.kind === 'vfolder') return;
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'move';
                  if (dragOverClusterId !== anchor.id) {
                    setDragOverClusterId(anchor.id);
                    scheduleHoverPreview(cluster, e.currentTarget.closest('article') ?? e.currentTarget);
                  }
                }}
                onDragLeave={() => {
                  if (dragOverClusterId === anchor.id) {
                    setDragOverClusterId(null);
                  }
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOverClusterId(null);
                  clearHoverPreview();
                  if (!session.isClustered || isSpecialBucket) {
                    clearActiveKromaDrag();
                    return;
                  }
                  const dropped = readKromaDragFromDrop(e.dataTransfer);
                  clearActiveKromaDrag();
                  if (!dropped) return;
                  if (dropped.kind === 'photos' && dropped.photoIds.length > 0) {
                    void movePhotosToCluster(dropped.photoIds, anchor.id);
                    return;
                  }
                  if (dropped.kind === 'clusters' && dropped.clusterIds.length === 1) {
                    const sourceClusterId = dropped.clusterIds[0];
                    if (sourceClusterId && sourceClusterId !== anchor.id) {
                      void mergeClustersById(sourceClusterId, anchor.id);
                    }
                  }
                }}
                className="flex items-center gap-2.5 pl-1"
              >
                {/* Clean Portrait Thumbnail */}
                <div className="w-12 h-14 rounded overflow-hidden relative shrink-0 bg-surface-lowest border border-hairline">
                  {(() => {
                    const repPhoto =
                      cluster.photos.find((p) => p.id === anchor.anchorPhotoId) ?? cluster.photos[0];
                    return repPhoto ? (
                      <GradedThumbnailImg
                        src={anchor.thumbnailUrl}
                        alt={anchor.subjectName}
                        recipe={repPhoto.developRecipe}
                        draggable={false}
                        loading="lazy"
                        decoding="async"
                        className="w-full h-full object-cover pointer-events-none"
                      />
                    ) : (
                      <img
                        src={anchor.thumbnailUrl}
                        alt={anchor.subjectName}
                        draggable={false}
                        loading="lazy"
                        decoding="async"
                        className="w-full h-full object-cover pointer-events-none"
                      />
                    );
                  })()}
                  {hasAmbiguous && (
                    <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-tangerine-alert" />
                  )}
                </div>

                {/* Essential Subject Info Only */}
                <div className="flex-1 min-w-0 flex flex-col gap-1">
                  <div className="flex items-center justify-between gap-1">
                    {isEditing ? (
                      <input
                        type="text"
                        autoFocus
                        value={draftName}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => setDraftName(e.target.value)}
                        onBlur={() => void commitEdit(anchor.id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') void commitEdit(anchor.id);
                          if (e.key === 'Escape') setEditingClusterId(null);
                        }}
                        className="w-full px-1.5 py-0.5 rounded bg-canvas border border-amber-core text-[13px] font-display font-bold text-ink-primary outline-none"
                      />
                    ) : (
                      <>
                        <span className="font-display text-[13px] font-semibold text-ink-primary truncate">
                          {anchor.subjectName}
                        </span>
                        {isSelected && !isSpecialBucket && (
                          <DarkroomTooltip content="Rename" side="top">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                beginEdit(anchor.id, anchor.subjectName);
                              }}
                              aria-label="Rename subject"
                              className="text-ink-telemetry hover:text-amber-soft shrink-0 cursor-pointer flex items-center"
                            >
                              <span className="material-symbols-outlined text-[14px]">edit</span>
                            </button>
                          </DarkroomTooltip>
                        )}
                      </>
                    )}
                  </div>

                  <div className="flex items-center justify-between gap-1 tabular-nums-mono text-[10px]">
                    <span className="text-ink-secondary">
                      {activeVirtualFolderId !== 'all' &&
                      matchedInActiveFolder !== cluster.photos.length
                        ? `${matchedInActiveFolder}/${cluster.photos.length} photos`
                        : `${cluster.photos.length} ${cluster.photos.length === 1 ? 'photo' : 'photos'}`}
                    </span>

                    {folderSummary ? (
                      <span
                        title={folderTooltip}
                        className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-canvas border border-hairline text-ink-secondary max-w-[100px] truncate"
                      >
                        <span className="material-symbols-outlined text-[10px] text-amber-soft shrink-0">
                          folder
                        </span>
                        <span className="truncate">{folderSummary}</span>
                      </span>
                    ) : hasAmbiguous ? (
                      <span className="text-tangerine-alert font-medium">
                        {anchor.ambiguousCount} review
                      </span>
                    ) : isSpecialBucket ? (
                      <span className="text-ink-telemetry">Bucket</span>
                    ) : faceOnly ? (
                      <span className="text-ink-telemetry">Face-Only</span>
                    ) : (
                      <span className="text-amber-soft">Slate {anchor.slateCode}</span>
                    )}
                  </div>
                </div>
              </div>
            </article>
          );
        })
        )}
      </div>

      <SubjectHoverPreviewCard
        cluster={hoverPreview?.cluster ?? null}
        anchorRect={hoverPreview?.rect ?? null}
      />
      </section>

      <PanelResizeHandle
        panelSide="left"
        width={subjectsPanel.width}
        minWidth={210}
        maxWidth={440}
        onChangeWidth={subjectsPanel.setWidth}
        onResetWidth={subjectsPanel.resetWidth}
        label="Resize Subjects sidebar"
      />
    </>
  );
}
