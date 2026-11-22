import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import {
  buildVirtualFolderTree,
  computeFolderPhotoCounts,
  isDescendantFolder,
  type VirtualFolderTreeNode,
} from '../../domain/virtualFolder';
import { DarkroomTooltip, FAST_FADE, TelemetryPill } from '../darkroom-kit';
import {
  beginKromaDrag,
  clearActiveKromaDrag,
  getActiveKromaDrag,
  readKromaDragFromDrop,
} from './kromaDragState';

const DRAG_AUTO_EXPAND_MS = 450;

export function VirtualFolderTreePanel() {
  const {
    session,
    activeVirtualFolderId,
    setActiveVirtualFolderId,
    folderCreateNonce,
    createVirtualFolder,
    renameVirtualFolder,
    moveVirtualFolder,
    deleteVirtualFolder,
    assignPhotosToVirtualFolder,
    assignClustersToVirtualFolder,
  } = useStudio();

  const [collapsedPanel, setCollapsedPanel] = useState(false);
  const [expandedIds, setExpandedIds] = useState<ReadonlySet<string>>(new Set());
  const [creatingParentId, setCreatingParentId] = useState<string | null | undefined>(undefined);
  const [draftCreateName, setDraftCreateName] = useState('');
  const [renamingFolderId, setRenamingFolderId] = useState<string | null>(null);
  const [draftRenameName, setDraftRenameName] = useState('');
  const [dragOverTarget, setDragOverTarget] = useState<string | null>(null);
  const [draggingFolderId, setDraggingFolderId] = useState<string | null>(null);

  const hoverExpandTimerRef = useRef<number | null>(null);
  const createInputRef = useRef<HTMLInputElement>(null);

  const virtualFolders = session?.virtualFolders ?? [];

  const allPhotos = useMemo(
    () => (session ? session.clusters.flatMap((c) => c.photos) : []),
    [session],
  );

  const tree = useMemo(() => buildVirtualFolderTree(virtualFolders), [virtualFolders]);

  const counts = useMemo(
    () => computeFolderPhotoCounts(virtualFolders, allPhotos),
    [virtualFolders, allPhotos],
  );

  // Open inline creation input when triggered from the main Photo Grid toolbar
  useEffect(() => {
    if (folderCreateNonce === 0) return;
    setCollapsedPanel(false);
    const parent =
      activeVirtualFolderId !== 'all' && activeVirtualFolderId !== 'unassigned'
        ? activeVirtualFolderId
        : null;
    if (parent) {
      setExpandedIds((prev) => new Set(prev).add(parent));
    }
    setCreatingParentId(parent);
    setDraftCreateName('');
  }, [folderCreateNonce]);

  useEffect(() => {
    if (creatingParentId !== undefined) {
      window.setTimeout(() => createInputRef.current?.focus(), 0);
    }
  }, [creatingParentId]);

  useEffect(() => {
    return () => {
      if (hoverExpandTimerRef.current !== null) {
        window.clearTimeout(hoverExpandTimerRef.current);
      }
    };
  }, []);

  if (!session) return null;

  const clearHoverExpand = () => {
    if (hoverExpandTimerRef.current !== null) {
      window.clearTimeout(hoverExpandTimerRef.current);
      hoverExpandTimerRef.current = null;
    }
  };

  const scheduleAutoExpand = (folderId: string) => {
    clearHoverExpand();
    hoverExpandTimerRef.current = window.setTimeout(() => {
      setExpandedIds((prev) => (prev.has(folderId) ? prev : new Set(prev).add(folderId)));
      hoverExpandTimerRef.current = null;
    }, DRAG_AUTO_EXPAND_MS);
  };

  const toggleExpand = (folderId: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) next.delete(folderId);
      else next.add(folderId);
      return next;
    });
  };

  const beginCreateFolder = (parentId: string | null) => {
    setCollapsedPanel(false);
    if (parentId) {
      setExpandedIds((prev) => new Set(prev).add(parentId));
    }
    setRenamingFolderId(null);
    setCreatingParentId(parentId);
    setDraftCreateName('');
  };

  const commitCreateFolder = async () => {
    if (creatingParentId === undefined) return;
    const trimmed = draftCreateName.trim();
    if (!trimmed) {
      setCreatingParentId(undefined);
      return;
    }
    const ok = await createVirtualFolder(trimmed, creatingParentId);
    if (ok) {
      setDraftCreateName('');
      setCreatingParentId(undefined);
    }
  };

  const commitRenameFolder = async (folderId: string) => {
    const trimmed = draftRenameName.trim();
    setRenamingFolderId(null);
    if (!trimmed) return;
    await renameVirtualFolder(folderId, trimmed);
  };

  const handleDropIntoFolder = (e: React.DragEvent, targetFolderId: string | null) => {
    e.preventDefault();
    e.stopPropagation();
    clearHoverExpand();
    setDragOverTarget(null);

    const payload = readKromaDragFromDrop(e.dataTransfer);
    clearActiveKromaDrag();
    if (!payload) return;

    if (payload.kind === 'photos' && payload.photoIds.length > 0) {
      void assignPhotosToVirtualFolder(payload.photoIds, targetFolderId);
      return;
    }

    if (payload.kind === 'clusters' && payload.clusterIds.length > 0) {
      void assignClustersToVirtualFolder(payload.clusterIds, targetFolderId);
      return;
    }

    if (payload.kind === 'vfolder') {
      if (targetFolderId && isDescendantFolder(virtualFolders, targetFolderId, payload.folderId)) {
        return;
      }
      void moveVirtualFolder(payload.folderId, targetFolderId);
    }
  };

  const canAcceptDrop = (e: React.DragEvent, targetFolderId: string | null): boolean => {
    const active = getActiveKromaDrag();
    if (active) {
      if (
        active.kind === 'vfolder' &&
        targetFolderId &&
        isDescendantFolder(virtualFolders, targetFolderId, active.folderId)
      ) {
        return false;
      }
      return true;
    }

    const types = Array.from(e.dataTransfer.types ?? []);
    const hasKromaPayload =
      types.includes('application/x-kroma-photos') ||
      types.includes('application/x-kroma-clusters') ||
      types.includes('application/x-kroma-cluster') ||
      types.includes('application/x-kroma-vfolder') ||
      types.includes('text/plain');
    if (!hasKromaPayload) return false;
    if (
      draggingFolderId &&
      targetFolderId &&
      isDescendantFolder(virtualFolders, targetFolderId, draggingFolderId)
    ) {
      return false;
    }
    return true;
  };

  const creatingParentName =
    creatingParentId !== undefined && creatingParentId !== null
      ? (virtualFolders.find((f) => f.id === creatingParentId)?.name ?? 'Folder')
      : null;

  const renderNode = (node: VirtualFolderTreeNode) => {
    const { folder, depth, children } = node;
    const hasChildren = children.length > 0;
    const isExpanded = expandedIds.has(folder.id);
    const isActive = activeVirtualFolderId === folder.id;
    const isDropTarget = dragOverTarget === folder.id;
    const isRenaming = renamingFolderId === folder.id;
    const rolledCount = counts.rolledUpByFolderId.get(folder.id) ?? 0;

    return (
      <div key={folder.id} className="flex flex-col">
        <div
          role="treeitem"
          aria-selected={isActive}
          aria-expanded={hasChildren ? isExpanded : undefined}
          draggable={!isRenaming}
          onDragStart={(e) => {
            e.stopPropagation();
            setDraggingFolderId(folder.id);
            beginKromaDrag(e.dataTransfer, { kind: 'vfolder', folderId: folder.id });
          }}
          onDragEnd={() => {
            clearActiveKromaDrag();
            setDraggingFolderId(null);
            setDragOverTarget(null);
            clearHoverExpand();
          }}
          onDragEnter={(e) => {
            if (!canAcceptDrop(e, folder.id)) return;
            e.preventDefault();
            e.stopPropagation();
          }}
          onDragOver={(e) => {
            if (!canAcceptDrop(e, folder.id)) return;
            e.preventDefault();
            e.stopPropagation();
            e.dataTransfer.dropEffect = 'move';
            if (dragOverTarget !== folder.id) {
              setDragOverTarget(folder.id);
              if (hasChildren && !isExpanded) {
                scheduleAutoExpand(folder.id);
              }
            }
          }}
          onDragLeave={(e) => {
            e.stopPropagation();
            if (dragOverTarget === folder.id) {
              setDragOverTarget(null);
              clearHoverExpand();
            }
          }}
          onDrop={(e) => handleDropIntoFolder(e, folder.id)}
          onClick={() => {
            if (!isRenaming) setActiveVirtualFolderId(folder.id);
          }}
          style={{ paddingLeft: `${8 + depth * 14}px` }}
          className={`group h-7 pr-1.5 rounded-md border flex items-center gap-1 transition-colors cursor-pointer select-none ${
            isDropTarget
              ? 'bg-emerald-focus/18 border-emerald-bright text-emerald-bright ring-1 ring-emerald-bright/50'
              : isActive
                ? 'bg-surface-active border-amber-core/60 text-amber-soft font-medium'
                : 'border-transparent text-ink-secondary hover:bg-surface-elevated hover:text-ink-primary'
          }`}
        >
          <button
            type="button"
            tabIndex={-1}
            onClick={(e) => {
              e.stopPropagation();
              if (hasChildren) toggleExpand(folder.id);
            }}
            aria-label={hasChildren ? (isExpanded ? 'Collapse folder' : 'Expand folder') : undefined}
            className={`w-4 h-4 flex items-center justify-center shrink-0 rounded ${
              hasChildren
                ? 'text-ink-telemetry hover:text-ink-primary cursor-pointer'
                : 'opacity-0 pointer-events-none'
            }`}
          >
            <span className="material-symbols-outlined text-[14px]">
              {isExpanded ? 'expand_more' : 'chevron_right'}
            </span>
          </button>

          <span
            className={`material-symbols-outlined text-[15px] shrink-0 ${
              isDropTarget
                ? 'text-emerald-bright'
                : isActive
                  ? 'text-amber-core'
                  : 'text-amber-soft/75'
            }`}
          >
            {isActive || isExpanded ? 'folder_open' : 'folder'}
          </span>

          {isRenaming ? (
            <input
              type="text"
              autoFocus
              value={draftRenameName}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => setDraftRenameName(e.target.value)}
              onBlur={() => void commitRenameFolder(folder.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  void commitRenameFolder(folder.id);
                } else if (e.key === 'Escape') {
                  e.stopPropagation();
                  setRenamingFolderId(null);
                }
              }}
              className="flex-1 min-w-0 h-5 px-1.5 rounded bg-canvas border border-amber-core text-[11.5px] text-ink-primary outline-none"
            />
          ) : (
            <>
              <span className="flex-1 min-w-0 truncate text-[11.5px]">{folder.name}</span>

              <div className="hidden group-hover:flex items-center gap-0.5 shrink-0">
                <DarkroomTooltip content={`New subfolder in ${folder.name}`} side="top">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      beginCreateFolder(folder.id);
                    }}
                    aria-label={`Create subfolder in ${folder.name}`}
                    className="w-5 h-5 rounded flex items-center justify-center text-ink-telemetry hover:text-amber-soft hover:bg-canvas/70 cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-[13px]">create_new_folder</span>
                  </button>
                </DarkroomTooltip>
                <DarkroomTooltip content="Rename folder" side="top">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setCreatingParentId(undefined);
                      setRenamingFolderId(folder.id);
                      setDraftRenameName(folder.name);
                    }}
                    aria-label={`Rename ${folder.name}`}
                    className="w-5 h-5 rounded flex items-center justify-center text-ink-telemetry hover:text-ink-primary hover:bg-canvas/70 cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-[13px]">edit</span>
                  </button>
                </DarkroomTooltip>
                <DarkroomTooltip content="Delete folder (photos stay safe in Unassigned)" side="top">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      void deleteVirtualFolder(folder.id);
                    }}
                    aria-label={`Delete ${folder.name}`}
                    className="w-5 h-5 rounded flex items-center justify-center text-ink-telemetry hover:text-tangerine-alert hover:bg-canvas/70 cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-[13px]">delete</span>
                  </button>
                </DarkroomTooltip>
              </div>

              <span className="tabular-nums-mono text-[10px] text-ink-telemetry group-hover:hidden shrink-0 pl-1">
                {rolledCount}
              </span>
            </>
          )}
        </div>

        {hasChildren && isExpanded && (
          <div className="flex flex-col">{children.map((child) => renderNode(child))}</div>
        )}
      </div>
    );
  };

  const isAllActive = activeVirtualFolderId === 'all';
  const isUnassignedActive = activeVirtualFolderId === 'unassigned';

  return (
    <div className="border-b border-hairline bg-surface-low flex flex-col shrink-0">
      {/* Virtual Folders Section Header */}
      <div className="h-9 px-3 bg-surface border-b border-hairline flex items-center justify-between gap-2 shrink-0">
        <button
          type="button"
          onClick={() => setCollapsedPanel((prev) => !prev)}
          className="flex items-center gap-1.5 min-w-0 text-left cursor-pointer group"
        >
          <span className="material-symbols-outlined text-[15px] text-ink-telemetry group-hover:text-ink-primary">
            {collapsedPanel ? 'chevron_right' : 'expand_more'}
          </span>
          <span className="font-display text-[12px] font-semibold text-ink-primary truncate">
            Folders
          </span>
          {virtualFolders.length > 0 && (
            <TelemetryPill tone="neutral">{virtualFolders.length}</TelemetryPill>
          )}
        </button>

        <DarkroomTooltip content="Create a new virtual folder" side="bottom">
          <button
            type="button"
            onClick={() => beginCreateFolder(null)}
            aria-label="Create a new virtual folder"
            className="h-6 px-1.5 rounded-md bg-canvas border border-hairline hover:border-amber-core/60 text-ink-secondary hover:text-amber-soft flex items-center gap-1 text-[10px] font-medium transition-colors cursor-pointer shrink-0"
          >
            <span className="material-symbols-outlined text-[13px]">create_new_folder</span>
            <span>New</span>
          </button>
        </DarkroomTooltip>
      </div>

      <AnimatePresence initial={false}>
        {!collapsedPanel && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={FAST_FADE}
            className="overflow-hidden"
          >
            <div
              role="tree"
              aria-label="Virtual Folders"
              className="p-2 flex flex-col gap-0.5 max-h-[210px] overflow-y-auto"
            >
              {/* All Photos (drop a folder here to move it back to root level) */}
              <div
                role="treeitem"
                aria-selected={isAllActive}
                onClick={() => setActiveVirtualFolderId('all')}
                onDragEnter={(e) => {
                  const active = getActiveKromaDrag();
                  if (active ? active.kind !== 'vfolder' : !e.dataTransfer.types.includes('application/x-kroma-vfolder')) {
                    return;
                  }
                  e.preventDefault();
                }}
                onDragOver={(e) => {
                  const active = getActiveKromaDrag();
                  if (active ? active.kind !== 'vfolder' : !e.dataTransfer.types.includes('application/x-kroma-vfolder')) {
                    return;
                  }
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'move';
                  setDragOverTarget('__root__');
                }}
                onDragLeave={() => {
                  if (dragOverTarget === '__root__') setDragOverTarget(null);
                }}
                onDrop={(e) => {
                  setDragOverTarget(null);
                  const payload = readKromaDragFromDrop(e.dataTransfer);
                  clearActiveKromaDrag();
                  if (payload?.kind === 'vfolder') {
                    e.preventDefault();
                    void moveVirtualFolder(payload.folderId, null);
                  }
                }}
                className={`h-7 px-2 rounded-md border flex items-center gap-1.5 transition-colors cursor-pointer select-none ${
                  dragOverTarget === '__root__'
                    ? 'bg-emerald-focus/18 border-emerald-bright text-emerald-bright'
                    : isAllActive
                      ? 'bg-surface-active border-amber-core/60 text-amber-soft font-medium'
                      : 'border-transparent text-ink-secondary hover:bg-surface-elevated hover:text-ink-primary'
                }`}
              >
                <span className="material-symbols-outlined text-[15px] shrink-0">
                  photo_library
                </span>
                <span className="flex-1 min-w-0 truncate text-[11.5px]">All Photos</span>
                <span className="tabular-nums-mono text-[10px] text-ink-telemetry shrink-0">
                  {counts.all}
                </span>
              </div>

              {/* Unassigned (drop photos/subjects here to clear their virtual folder) */}
              {virtualFolders.length > 0 && (
                <div
                  role="treeitem"
                  aria-selected={isUnassignedActive}
                  onClick={() => setActiveVirtualFolderId('unassigned')}
                  onDragEnter={(e) => {
                    const active = getActiveKromaDrag();
                    if (active && active.kind === 'vfolder') return;
                    if (!canAcceptDrop(e, null)) return;
                    e.preventDefault();
                  }}
                  onDragOver={(e) => {
                    const active = getActiveKromaDrag();
                    if (active && active.kind === 'vfolder') return;
                    if (!canAcceptDrop(e, null)) return;
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    setDragOverTarget('__unassigned__');
                  }}
                  onDragLeave={() => {
                    if (dragOverTarget === '__unassigned__') setDragOverTarget(null);
                  }}
                  onDrop={(e) => handleDropIntoFolder(e, null)}
                  className={`h-7 px-2 rounded-md border flex items-center gap-1.5 transition-colors cursor-pointer select-none ${
                    dragOverTarget === '__unassigned__'
                      ? 'bg-amber-core/18 border-amber-core text-amber-soft'
                      : isUnassignedActive
                        ? 'bg-surface-active border-amber-core/60 text-amber-soft font-medium'
                        : 'border-transparent text-ink-secondary hover:bg-surface-elevated hover:text-ink-primary'
                  }`}
                >
                  <span className="material-symbols-outlined text-[15px] shrink-0">folder_off</span>
                  <span className="flex-1 min-w-0 truncate text-[11.5px]">Unassigned</span>
                  <span className="tabular-nums-mono text-[10px] text-ink-telemetry shrink-0">
                    {counts.unassigned}
                  </span>
                </div>
              )}

              {/* User-Created Virtual Folder Tree */}
              {tree.map((node) => renderNode(node))}

              {/* Inline New Folder Input */}
              {creatingParentId !== undefined && (
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void commitCreateFolder();
                  }}
                  className="mt-1 p-1.5 rounded-lg bg-canvas border border-amber-core/60 flex flex-col gap-1"
                >
                  {creatingParentName && (
                    <span className="tabular-nums-mono text-[9.5px] text-ink-telemetry truncate">
                      Inside {creatingParentName}
                    </span>
                  )}
                  <div className="flex items-center gap-1">
                    <input
                      ref={createInputRef}
                      type="text"
                      value={draftCreateName}
                      onChange={(e) => setDraftCreateName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Escape') {
                          e.stopPropagation();
                          setCreatingParentId(undefined);
                        }
                      }}
                      placeholder="Folder name..."
                      className="flex-1 min-w-0 h-6 px-1.5 rounded bg-surface border border-hairline focus:border-amber-core text-[11.5px] text-ink-primary placeholder:text-ink-telemetry outline-none"
                    />
                    <button
                      type="submit"
                      className="h-6 px-2 rounded bg-amber-core text-amber-ink font-display text-[10.5px] font-bold hover:brightness-110 cursor-pointer shrink-0"
                    >
                      Add
                    </button>
                    <button
                      type="button"
                      onClick={() => setCreatingParentId(undefined)}
                      aria-label="Cancel new folder"
                      className="w-6 h-6 rounded flex items-center justify-center text-ink-telemetry hover:text-ink-primary cursor-pointer shrink-0"
                    >
                      <span className="material-symbols-outlined text-[14px]">close</span>
                    </button>
                  </div>
                </form>
              )}

              {/* Helpful Empty Hint */}
              {virtualFolders.length === 0 && creatingParentId === undefined && (
                <div className="px-2 py-1.5 text-[10.5px] text-ink-telemetry leading-snug">
                  Create virtual folders to group your photos, then drag selected photos or
                  subjects here.
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
