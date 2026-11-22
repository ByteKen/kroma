import { AnimatePresence, motion } from 'framer-motion';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStudio, type GridDisplayMode } from '../../app/StudioContext';
import { isFaceOnlySlateCode } from '../../domain/identity';
import {
  buildVirtualFolderTree,
  formatVirtualFolderBreadcrumb,
  type VirtualFolderTreeNode,
} from '../../domain/virtualFolder';
import {
  DarkroomButton,
  FAST_FADE,
  SegmentedControl,
  TelemetryPill,
  VirtualGrid,
  type SegmentedOption,
  type VirtualGridHandle,
} from '../darkroom-kit';
import type { RectLike } from './lightboxGeometry';
import { FloatingCullingDock } from './FloatingCullingDock';
import { ManualMatchSplitView } from './ManualMatchSplitView';
import { PhotoCarouselLightbox } from './PhotoCarouselLightbox';
import { PhotoClusterCard } from './PhotoClusterCard';

const DISPLAY_MODES: readonly SegmentedOption<GridDisplayMode>[] = [
  { id: 'grid', icon: 'grid_view', title: 'Photo grid' },
  { id: 'split', icon: 'view_column', title: '2-up compare' },
  { id: 'loupe', icon: 'search_check', title: 'Single loupe' },
];

export function ClusterGridWorkspace() {
  const {
    session,
    preferences,
    activeCluster,
    activeVirtualFolderId,
    setActiveVirtualFolderId,
    requestNewVirtualFolder,
    assignPhotosToVirtualFolder,
    visibleClusterPhotos,
    selectedPhotoIds,
    reviewWarningThreshold,
    approvedReviewPhotoIds,
    manualMatchSplitOpen,
    setManualMatchSplitOpen,
    displayMode,
    setDisplayMode,
    showLandmarksOverlay,
    toggleLandmarksOverlay,
    togglePhotoSelection,
    selectAllVisiblePhotos,
    openPhotoInDevelop,
    selectNextCluster,
  } = useStudio();

  const gridRef = useRef<VirtualGridHandle | null>(null);
  const moveMenuRef = useRef<HTMLDivElement | null>(null);
  const [moveMenuOpen, setMoveMenuOpen] = useState(false);
  const [lightboxState, setLightboxState] = useState<{
    readonly photoId: string;
    readonly originRect: RectLike | null;
  } | null>(null);

  const virtualFolders = session?.virtualFolders ?? [];
  const folderById = useMemo(
    () => new Map(virtualFolders.map((f) => [f.id, f])),
    [virtualFolders],
  );
  const folderTree = useMemo(() => buildVirtualFolderTree(virtualFolders), [virtualFolders]);

  // Flatten tree in visual reading order for the toolbar "Move to Folder" picker
  const flatFolders = useMemo(() => {
    const out: { readonly id: string; readonly name: string; readonly depth: number }[] = [];
    const walk = (nodes: readonly VirtualFolderTreeNode[]) => {
      for (const n of nodes) {
        out.push({ id: n.folder.id, name: n.folder.name, depth: n.depth });
        walk(n.children);
      }
    };
    walk(folderTree);
    return out;
  }, [folderTree]);

  useEffect(() => {
    if (!moveMenuOpen) return;
    const onPointerDown = (e: MouseEvent) => {
      if (moveMenuRef.current && !moveMenuRef.current.contains(e.target as Node)) {
        setMoveMenuOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMoveMenuOpen(false);
    };
    window.addEventListener('mousedown', onPointerDown);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('mousedown', onPointerDown);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [moveMenuOpen]);

  useEffect(() => {
    if (selectedPhotoIds.size === 0) setMoveMenuOpen(false);
  }, [selectedPhotoIds.size]);

  const activeClusterId = activeCluster?.anchor.id ?? '';
  useEffect(() => {
    setLightboxState(null);
  }, [activeClusterId, manualMatchSplitOpen]);

  const lightboxIndex = lightboxState
    ? visibleClusterPhotos.findIndex((p) => p.id === lightboxState.photoId)
    : -1;
  const isLightboxOpen = lightboxIndex >= 0;

  const handleOpenLightbox = useCallback((photoId: string, originRect: RectLike | null) => {
    setLightboxState({ photoId, originRect });
  }, []);

  const handleLightboxChangeIndex = useCallback(
    (nextIndex: number) => {
      const target = visibleClusterPhotos[nextIndex];
      if (!target) return;
      setLightboxState((prev) => (prev ? { ...prev, photoId: target.id } : null));
      gridRef.current?.scrollToIndex(nextIndex, false);
    },
    [visibleClusterPhotos],
  );

  const handleCloseLightbox = useCallback(() => {
    setLightboxState(null);
  }, []);

  if (!session || !activeCluster) return null;

  const { anchor } = activeCluster;
  const faceOnly = isFaceOnlySlateCode(anchor.slateCode);
  const isSpecialBucket = anchor.id.startsWith('_');
  const activeFolderBreadcrumb =
    activeVirtualFolderId === 'all'
      ? null
      : activeVirtualFolderId === 'unassigned'
        ? 'Unassigned'
        : formatVirtualFolderBreadcrumb(virtualFolders, activeVirtualFolderId) || 'Folder';

  return (
    <section
      aria-label="Subject Photo Grid"
      className="flex-1 min-w-0 bg-canvas flex flex-col overflow-hidden relative @container"
    >
      {/* Single Clean Cluster & Folder Toolbar */}
      <div className="px-5 py-3 bg-surface border-b border-hairline flex flex-wrap items-center justify-between gap-3 shrink-0">
        {/* Subject Identity + Active Virtual Folder Breadcrumb */}
        <div className="flex items-center gap-2.5 min-w-0 flex-wrap">
          <h1 className="font-display text-[17px] font-bold text-ink-primary tracking-tight truncate">
            {anchor.subjectName}
          </h1>

          {!isSpecialBucket && (
            <TelemetryPill tone={faceOnly ? 'neutral' : 'amber'} className="shrink-0">
              {faceOnly ? 'Face-Only' : `Slate ${anchor.slateCode}`}
            </TelemetryPill>
          )}

          {activeFolderBreadcrumb && (
            <button
              type="button"
              onClick={() => setActiveVirtualFolderId('all')}
              title="Clear folder filter and show all photos"
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-core/12 border border-amber-core/45 tabular-nums-mono text-[10.5px] text-amber-soft hover:bg-amber-core/20 cursor-pointer shrink-0"
            >
              <span className="material-symbols-outlined text-[13px]">folder</span>
              <span className="max-w-[180px] truncate">{activeFolderBreadcrumb}</span>
              <span className="material-symbols-outlined text-[12px]">close</span>
            </button>
          )}

          <span className="tabular-nums-mono text-[11px] text-ink-telemetry shrink-0">
            {visibleClusterPhotos.length} shown
          </span>
        </div>

        {/* Focused View, Virtual Folder & Selection Controls */}
        <div className="flex items-center flex-wrap gap-2">
          <DarkroomButton
            variant="subtle"
            size="sm"
            icon="create_new_folder"
            onClick={requestNewVirtualFolder}
            title="Create a new virtual folder"
          >
            <span className="hidden @2xl:inline">New Folder</span>
          </DarkroomButton>

          <div ref={moveMenuRef} className="relative">
            <DarkroomButton
              variant={selectedPhotoIds.size > 0 ? 'emerald' : 'subtle'}
              size="sm"
              icon="drive_file_move"
              active={moveMenuOpen}
              disabled={selectedPhotoIds.size === 0}
              onClick={() => setMoveMenuOpen((prev) => !prev)}
              title={
                selectedPhotoIds.size > 0
                  ? `Move ${selectedPhotoIds.size} selected photo(s) to a virtual folder (or drag directly onto the folder tree)`
                  : 'Select photos first (Shift+Click for range) to move into a virtual folder'
              }
            >
              <span>
                {selectedPhotoIds.size > 0
                  ? `Move (${selectedPhotoIds.size})`
                  : 'Move to Folder'}
              </span>
              <span className="material-symbols-outlined text-[14px] -mr-0.5">
                {moveMenuOpen ? 'expand_less' : 'expand_more'}
              </span>
            </DarkroomButton>

            <AnimatePresence>
              {moveMenuOpen && (
                <motion.div
                  role="menu"
                  aria-label="Move selected photos to virtual folder"
                  initial={{ opacity: 0, y: -4, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -4, scale: 0.98 }}
                  transition={FAST_FADE}
                  className="absolute right-0 mt-1.5 w-64 rounded-xl bg-surface-elevated border border-hairline-strong shadow-[0_18px_48px_rgba(0,0,0,0.6)] p-1.5 z-40"
                >
                  <div className="px-2.5 pt-1 pb-1.5 tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                    Move {selectedPhotoIds.size} Selected Photo
                    {selectedPhotoIds.size === 1 ? '' : 's'}
                  </div>

                  <div className="max-h-56 overflow-y-auto flex flex-col gap-0.5">
                    {flatFolders.length === 0 ? (
                      <div className="px-2.5 py-2 text-[11px] text-ink-telemetry">
                        No virtual folders yet — create one below.
                      </div>
                    ) : (
                      flatFolders.map((folder) => (
                        <button
                          key={folder.id}
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            setMoveMenuOpen(false);
                            void assignPhotosToVirtualFolder(
                              Array.from(selectedPhotoIds),
                              folder.id,
                            );
                          }}
                          style={{ paddingLeft: `${10 + folder.depth * 14}px` }}
                          className="w-full py-1.5 pr-2.5 rounded-lg text-left text-[12px] text-ink-primary hover:bg-surface-active flex items-center gap-2 transition-colors cursor-pointer"
                        >
                          <span className="material-symbols-outlined text-[15px] text-amber-soft shrink-0">
                            folder
                          </span>
                          <span className="truncate">{folder.name}</span>
                        </button>
                      ))
                    )}

                    {flatFolders.length > 0 && (
                      <button
                        type="button"
                        role="menuitem"
                        onClick={() => {
                          setMoveMenuOpen(false);
                          void assignPhotosToVirtualFolder(Array.from(selectedPhotoIds), null);
                        }}
                        className="w-full px-2.5 py-1.5 rounded-lg text-left text-[12px] text-ink-secondary hover:bg-surface-active hover:text-ink-primary flex items-center gap-2 transition-colors cursor-pointer"
                      >
                        <span className="material-symbols-outlined text-[15px] text-ink-telemetry shrink-0">
                          folder_off
                        </span>
                        <span>Unassigned (Remove from folder)</span>
                      </button>
                    )}
                  </div>

                  <div className="my-1 border-t border-hairline" />

                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setMoveMenuOpen(false);
                      requestNewVirtualFolder();
                    }}
                    className="w-full px-2.5 py-1.5 rounded-lg text-left text-[11.5px] font-medium text-amber-soft hover:bg-surface-active flex items-center gap-2 transition-colors cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-[15px]">
                      create_new_folder
                    </span>
                    <span>Create New Virtual Folder…</span>
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <DarkroomButton
            variant={showLandmarksOverlay ? 'emerald' : 'subtle'}
            size="sm"
            icon="face"
            onClick={toggleLandmarksOverlay}
            title="Toggle face boxes"
          >
            <span className="hidden @3xl:inline">Face Boxes</span>
          </DarkroomButton>

          <SegmentedControl
            layoutId="culling-display-mode"
            options={DISPLAY_MODES}
            value={displayMode}
            onChange={setDisplayMode}
          />

          <DarkroomButton
            variant={selectedPhotoIds.size > 0 ? 'secondary' : 'subtle'}
            size="sm"
            icon="checklist"
            onClick={selectAllVisiblePhotos}
            title="Toggle select all (tip: Shift+Click selects a range)"
          >
            <span>
              {selectedPhotoIds.size > 0 ? `Selected (${selectedPhotoIds.size})` : 'Select All'}
            </span>
          </DarkroomButton>

          {session.clusters.length > 1 && (
            <DarkroomButton
              variant="secondary"
              size="sm"
              onClick={selectNextCluster}
              title="Next subject"
            >
              <span>Next →</span>
            </DarkroomButton>
          )}
        </div>
      </div>

      {/* Split-View Manual Matcher OR Virtualized Photo Grid Canvas (handles 2,000+ photos smoothly) */}
      {manualMatchSplitOpen && session.isClustered && !isSpecialBucket ? (
        <ManualMatchSplitView />
      ) : visibleClusterPhotos.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 text-ink-secondary p-5">
          <span className="material-symbols-outlined text-[34px] text-ink-telemetry">
            filter_alt_off
          </span>
          <p className="font-display text-[14px] text-ink-primary">
            No photos match the current search or filter.
          </p>
          <p className="text-[12px] text-ink-telemetry">
            Clear your search query or switch quick filters to see all frames.
          </p>
        </div>
      ) : (
        <VirtualGrid
          ref={gridRef}
          items={visibleClusterPhotos}
          getKey={(photo) => photo.id}
          fixedColumns={displayMode === 'loupe' ? 1 : displayMode === 'split' ? 2 : undefined}
          maxContentWidthPx={displayMode === 'loupe' ? 580 : undefined}
          minItemWidth={205}
          footerHeightPx={preferences.showThumbnailMetadata ? 56 : 0}
          gapPx={16}
          paddingPx={20}
          bottomPaddingPx={88}
          renderItem={(photo) => {
            const isSelected = selectedPhotoIds.has(photo.id);
            const virtualFolderLabel = photo.virtualFolderId
              ? (folderById.get(photo.virtualFolderId)?.name ?? null)
              : null;
            return (
              <PhotoClusterCard
                photo={photo}
                isSelected={isSelected}
                virtualFolderLabel={virtualFolderLabel}
                dragPhotoIds={
                  isSelected && selectedPhotoIds.size > 1
                    ? Array.from(selectedPhotoIds)
                    : [photo.id]
                }
                isClustered={session.isClustered}
                reviewWarningThreshold={reviewWarningThreshold}
                isReviewApproved={approvedReviewPhotoIds.has(photo.id)}
                showLandmarksOverlay={showLandmarksOverlay}
                showMetadata={preferences.showThumbnailMetadata}
                onToggleSelect={togglePhotoSelection}
                onOpenLightbox={handleOpenLightbox}
                onOpenDevelop={(photoId) => openPhotoInDevelop(photoId)}
                onManualMatch={() => setManualMatchSplitOpen(true)}
              />
            );
          }}
        />
      )}

      <FloatingCullingDock
        visible={!(manualMatchSplitOpen && session.isClustered && !isSpecialBucket)}
      />

      <PhotoCarouselLightbox
        open={isLightboxOpen}
        photos={visibleClusterPhotos}
        activeIndex={lightboxIndex}
        originRect={lightboxState?.originRect ?? null}
        subjectName={anchor.subjectName}
        isClustered={session.isClustered}
        isSpecialBucket={isSpecialBucket}
        reviewWarningThreshold={reviewWarningThreshold}
        approvedReviewPhotoIds={approvedReviewPhotoIds}
        showLandmarksOverlay={showLandmarksOverlay}
        selectedPhotoIds={selectedPhotoIds}
        onChangeIndex={handleLightboxChangeIndex}
        onClose={handleCloseLightbox}
        onToggleSelect={togglePhotoSelection}
        onToggleLandmarks={toggleLandmarksOverlay}
        onOpenDevelop={(photoId) => openPhotoInDevelop(photoId)}
        onOpenManualMatch={() => setManualMatchSplitOpen(true)}
      />
    </section>
  );
}
