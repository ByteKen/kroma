import { useMemo, useState } from 'react';
import {
  compareSubjectClustersAlphabetically,
  type SubjectCluster,
} from '../../domain/identity';
import { DarkroomButton, DarkroomModal, TelemetryPill } from '../darkroom-kit';

interface MergeClusterModalProps {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly sourceCluster: SubjectCluster;
  readonly candidates: readonly SubjectCluster[];
  readonly onConfirmMerge: (targetClusterId: string) => void;
}

export function MergeClusterModal({
  open,
  onClose,
  sourceCluster,
  candidates,
  onConfirmMerge,
}: MergeClusterModalProps) {
  const sortedCandidates = useMemo(
    () => [...candidates].sort(compareSubjectClustersAlphabetically),
    [candidates],
  );
  const [selectedTargetId, setSelectedTargetId] = useState<string>(
    sortedCandidates[0]?.anchor.id ?? '',
  );

  const effectiveTargetId =
    sortedCandidates.some((c) => c.anchor.id === selectedTargetId)
      ? selectedTargetId
      : (sortedCandidates[0]?.anchor.id ?? '');

  return (
    <DarkroomModal
      open={open}
      onClose={onClose}
      icon="merge"
      title={`Merge "${sourceCluster.anchor.subjectName}" Into...`}
      subtitle={`Moves all ${sourceCluster.photos.length} photo(s) into the target subject cluster`}
      footer={
        <>
          <DarkroomButton variant="secondary" size="md" onClick={onClose}>
            Cancel
          </DarkroomButton>
          <DarkroomButton
            variant="primary"
            size="md"
            icon="merge"
            disabled={!effectiveTargetId}
            onClick={() => {
              if (effectiveTargetId) {
                onConfirmMerge(effectiveTargetId);
                onClose();
              }
            }}
          >
            Merge Subjects
          </DarkroomButton>
        </>
      }
    >
      {sortedCandidates.length === 0 ? (
        <p className="text-[12px] text-ink-secondary">
          No other subject clusters are available to merge into.
        </p>
      ) : (
        <div className="flex flex-col gap-1.5 max-h-[300px] overflow-y-auto">
          {sortedCandidates.map((cluster) => {
            const isPicked = cluster.anchor.id === effectiveTargetId;
            return (
              <button
                key={cluster.anchor.id}
                type="button"
                onClick={() => setSelectedTargetId(cluster.anchor.id)}
                className={`p-2 rounded-lg border flex items-center gap-3 text-left transition-colors cursor-pointer ${
                  isPicked
                    ? 'bg-surface-active border-amber-core/65'
                    : 'bg-canvas border-hairline hover:bg-surface-elevated'
                }`}
              >
                <img
                  src={cluster.anchor.thumbnailUrl}
                  alt={cluster.anchor.subjectName}
                  className="w-10 h-12 rounded object-cover shrink-0 bg-surface-lowest"
                />
                <div className="flex-1 min-w-0">
                  <div className="font-display text-[13px] font-semibold text-ink-primary truncate">
                    {cluster.anchor.subjectName}
                  </div>
                  <div className="tabular-nums-mono text-[10px] text-ink-telemetry">
                    {cluster.photos.length} photo(s)
                  </div>
                </div>
                {isPicked && <TelemetryPill tone="amber">Target</TelemetryPill>}
              </button>
            );
          })}
        </div>
      )}
    </DarkroomModal>
  );
}
