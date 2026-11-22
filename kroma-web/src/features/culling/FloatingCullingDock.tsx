import { AnimatePresence, motion } from 'framer-motion';
import { useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import { DarkroomButton, SOFT_SPRING } from '../darkroom-kit';
import { MergeClusterModal } from './MergeClusterModal';

interface FloatingCullingDockProps {
  readonly visible: boolean;
}

/**
 * Floating frosted-glass action dock centered at the bottom of the Face Organizer grid.
 * Auto-hides when Manual Match Split View or other workspaces with bottom chrome are active.
 */
export function FloatingCullingDock({ visible }: FloatingCullingDockProps) {
  const {
    session,
    isScanningFolder,
    activeCluster,
    selectedPhotoIds,
    sfaceThreshold,
    lastClusteredThreshold,
    useTemporalAssist,
    lastClusteredTemporalAssist,
    setActiveWorkspace,
    setManualMatchSplitOpen,
    mergeActiveClusterInto,
    splitSelectedPhotos,
    runFaceOrganizer,
    canUndo,
    canRedo,
    undoLabel,
    redoLabel,
    undoOrganizationAction,
    redoOrganizationAction,
  } = useStudio();

  const [mergeModalOpen, setMergeModalOpen] = useState(false);

  if (!session) return null;

  const isSpecialBucket = activeCluster?.anchor.id.startsWith('_') ?? false;
  const mergeCandidates = activeCluster
    ? session.clusters.filter(
        (c) => c.anchor.id !== activeCluster.anchor.id && !c.anchor.id.startsWith('_'),
      )
    : [];
  const needsRecluster =
    session.isClustered &&
    (Math.abs(sfaceThreshold - lastClusteredThreshold) >= 0.005 ||
      useTemporalAssist !== lastClusteredTemporalAssist);

  const isEntireClusterSelected = Boolean(
    activeCluster &&
      activeCluster.photos.length > 0 &&
      activeCluster.photos.every((p) => selectedPhotoIds.has(p.id)),
  );
  const splitDisabled =
    !session.isClustered ||
    isSpecialBucket ||
    selectedPhotoIds.size === 0 ||
    isEntireClusterSelected;
  const splitTooltip = isEntireClusterSelected
    ? 'Keep at least 1 photo here'
    : selectedPhotoIds.size === 0
      ? 'Select photos to split'
      : `Split ${selectedPhotoIds.size} into new subject`;

  return (
    <>
      <AnimatePresence>
        {visible && (
          <div className="pointer-events-none absolute inset-x-0 bottom-4 z-30 flex justify-center px-4">
            <motion.nav
              aria-label="Subject organization actions"
              initial={{ opacity: 0, y: 16, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 12, scale: 0.96 }}
              transition={SOFT_SPRING}
              className="pointer-events-auto flex items-center gap-1 px-2 py-1.5 rounded-2xl bg-surface-elevated/92 backdrop-blur-xl border border-hairline-strong shadow-[0_14px_38px_rgba(0,0,0,0.78),0_0_0_1px_rgba(255,255,255,0.04)_inset] max-w-full overflow-x-auto"
            >
              <DarkroomButton
                variant="ghost"
                size="sm"
                icon="undo"
                tooltipSide="top"
                disabled={!canUndo || isScanningFolder}
                onClick={() => void undoOrganizationAction()}
                title={undoLabel ? `Undo ${undoLabel} (Ctrl+Z)` : 'Undo (Ctrl+Z)'}
                aria-label="Undo organization change"
              />

              <DarkroomButton
                variant="ghost"
                size="sm"
                icon="redo"
                tooltipSide="top"
                disabled={!canRedo || isScanningFolder}
                onClick={() => void redoOrganizationAction()}
                title={redoLabel ? `Redo ${redoLabel} (Ctrl+Shift+Z)` : 'Redo (Ctrl+Shift+Z)'}
                aria-label="Redo organization change"
              />

              <div className="h-4 w-px bg-hairline mx-0.5 shrink-0" />

              <DarkroomButton
                variant="ghost"
                size="sm"
                icon="tune"
                tooltipSide="top"
                onClick={() => setActiveWorkspace('develop')}
                title="Open Studio"
              >
                <span>
                  {selectedPhotoIds.size > 0 ? `Studio (${selectedPhotoIds.size})` : 'Studio'}
                </span>
              </DarkroomButton>

              <div className="h-4 w-px bg-hairline mx-0.5 shrink-0" />

              <DarkroomButton
                variant={session.isClustered && !isSpecialBucket ? 'emerald' : 'ghost'}
                size="sm"
                icon="compare"
                tooltipSide="top"
                disabled={!session.isClustered || isSpecialBucket}
                onClick={() => setManualMatchSplitOpen(true)}
                title="Side-by-side manual match"
              >
                <span>Manual Match</span>
              </DarkroomButton>

              <DarkroomButton
                variant="ghost"
                size="sm"
                icon="merge"
                tooltipSide="top"
                disabled={!session.isClustered || isSpecialBucket || mergeCandidates.length === 0}
                onClick={() => setMergeModalOpen(true)}
                title="Merge into another subject"
              >
                <span>Merge</span>
              </DarkroomButton>

              <DarkroomButton
                variant={splitDisabled ? 'ghost' : 'secondary'}
                size="sm"
                icon="call_split"
                tooltipSide="top"
                disabled={splitDisabled}
                className={!splitDisabled ? 'border-amber-core/45 text-amber-soft' : undefined}
                onClick={() => void splitSelectedPhotos()}
                title={splitTooltip}
              >
                <span>Split</span>
              </DarkroomButton>

              <div className="h-4 w-px bg-hairline mx-0.5 shrink-0" />

              <DarkroomButton
                variant={
                  !session.isClustered ? 'emerald' : needsRecluster ? 'primary' : 'secondary'
                }
                size="sm"
                icon={session.isClustered ? 'refresh' : 'face'}
                tooltipSide="top"
                disabled={isScanningFolder}
                className={
                  session.isClustered && !needsRecluster
                    ? 'border-amber-core/35 text-amber-soft hover:border-amber-core/60'
                    : undefined
                }
                onClick={() => void runFaceOrganizer()}
                title={
                  !session.isClustered
                    ? 'Group photos by face'
                    : needsRecluster
                      ? 'Apply match settings'
                      : 'Re-run face grouping'
                }
              >
                <span>
                  {!session.isClustered
                    ? 'Organize by Face'
                    : needsRecluster
                      ? 'Apply Re-cluster'
                      : 'Re-cluster'}
                </span>
              </DarkroomButton>
            </motion.nav>
          </div>
        )}
      </AnimatePresence>

      {activeCluster && (
        <MergeClusterModal
          open={mergeModalOpen}
          onClose={() => setMergeModalOpen(false)}
          sourceCluster={activeCluster}
          candidates={mergeCandidates}
          onConfirmMerge={(targetId) => void mergeActiveClusterInto(targetId)}
        />
      )}
    </>
  );
}
