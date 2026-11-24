import { AnimatePresence, motion } from 'framer-motion';
import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import { formatSimilarityPercent, isFaceOnlySlateCode } from '../../domain/identity';
import {
  DarkroomButton,
  FAST_FADE,
  SOFT_SPRING,
  TelemetryPill,
  VirtualVerticalList,
} from '../darkroom-kit';

export const AmbiguousReviewWorkspace = memo(function AmbiguousReviewWorkspace() {
  const {
    ambiguousReviewQueue,
    sfaceThreshold,
    reviewWarningThreshold,
    resolveAmbiguousReview,
    batchResolveAmbiguousReview,
    selectCluster,
    setActiveWorkspace,
  } = useStudio();

  const [selectedPhotoId, setSelectedPhotoId] = useState<string | null>(null);
  const [checkedPhotoIds, setCheckedPhotoIds] = useState<ReadonlySet<string>>(new Set());
  const lastCheckedIdRef = useRef<string | null>(null);

  // Drop stale checked IDs once items leave the queue
  useEffect(() => {
    const validIds = new Set(ambiguousReviewQueue.map((item) => item.photo.id));
    setCheckedPhotoIds((prev) => {
      if (prev.size === 0) return prev;
      let changed = false;
      const next = new Set<string>();
      for (const id of prev) {
        if (validIds.has(id)) next.add(id);
        else changed = true;
      }
      return changed ? next : prev;
    });
  }, [ambiguousReviewQueue]);

  const activeItem =
    ambiguousReviewQueue.find((item) => item.photo.id === selectedPhotoId) ??
    ambiguousReviewQueue[0] ??
    null;

  const sameSubjectItemIds = useMemo(() => {
    if (!activeItem) return [];
    const targetClusterId = activeItem.candidateCluster.anchor.id;
    return ambiguousReviewQueue
      .filter((item) => item.candidateCluster.anchor.id === targetClusterId)
      .map((item) => item.photo.id);
  }, [ambiguousReviewQueue, activeItem]);

  const toggleChecked = (photoId: string, shiftKey: boolean) => {
    const anchorId = lastCheckedIdRef.current;
    if (shiftKey && anchorId) {
      const startIdx = ambiguousReviewQueue.findIndex((item) => item.photo.id === anchorId);
      const endIdx = ambiguousReviewQueue.findIndex((item) => item.photo.id === photoId);
      if (startIdx !== -1 && endIdx !== -1) {
        const lo = Math.min(startIdx, endIdx);
        const hi = Math.max(startIdx, endIdx);
        setCheckedPhotoIds((prev) => {
          const next = new Set(prev);
          for (let i = lo; i <= hi; i += 1) {
            const entry = ambiguousReviewQueue[i];
            if (entry) next.add(entry.photo.id);
          }
          return next;
        });
        lastCheckedIdRef.current = photoId;
        return;
      }
    }

    lastCheckedIdRef.current = photoId;
    setCheckedPhotoIds((prev) => {
      const next = new Set(prev);
      if (next.has(photoId)) next.delete(photoId);
      else next.add(photoId);
      return next;
    });
  };

  const handleBatchApprove = async () => {
    const ids = Array.from(checkedPhotoIds);
    if (ids.length === 0) return;
    setCheckedPhotoIds(new Set());
    await batchResolveAmbiguousReview(ids, 'approve');
  };

  const handleBatchSplitTogether = async () => {
    const ids = Array.from(checkedPhotoIds);
    if (ids.length === 0) return;
    setCheckedPhotoIds(new Set());
    await batchResolveAmbiguousReview(ids, 'split_together');
  };

  // Rapid triage hotkeys: [A] Approve, [S] Split (respects multi-select when active)
  useEffect(() => {
    if (!activeItem) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      ) {
        return;
      }
      if (event.ctrlKey || event.metaKey || event.altKey) return;

      const key = event.key.toLowerCase();
      if (key === 'a') {
        event.preventDefault();
        if (checkedPhotoIds.size > 0) {
          void handleBatchApprove();
        } else {
          resolveAmbiguousReview(activeItem.photo.id, 'approve');
        }
      } else if (key === 's') {
        event.preventDefault();
        if (checkedPhotoIds.size > 0) {
          void handleBatchSplitTogether();
        } else {
          resolveAmbiguousReview(activeItem.photo.id, 'split');
        }
      } else if (key === 'escape' && checkedPhotoIds.size > 0) {
        setCheckedPhotoIds(new Set());
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeItem, checkedPhotoIds, resolveAmbiguousReview, batchResolveAmbiguousReview]);

  if (!activeItem) {
    return (
      <section
        aria-label="Ambiguous Match Review Queue"
        className="flex-1 flex flex-col items-center justify-center p-8 bg-canvas text-center gap-4"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={SOFT_SPRING}
          className="max-w-[480px] p-7 rounded-2xl bg-surface border border-emerald-focus/40 flex flex-col items-center gap-3.5 shadow-[0_12px_40px_rgba(0,0,0,0.45)]"
        >
          <div className="w-14 h-14 rounded-2xl bg-emerald-focus/15 border border-emerald-focus/40 text-emerald-bright flex items-center justify-center">
            <span className="material-symbols-outlined text-[30px]">verified</span>
          </div>
          <h1 className="font-display text-[20px] font-bold text-ink-primary">
            Review Queue Clear
          </h1>
          <p className="text-[13px] text-ink-secondary leading-relaxed">
            Every borderline face match has been verified. All subject clusters are locked and
            ready for editing or export.
          </p>
          <div className="flex items-center gap-2.5 pt-1">
            <DarkroomButton
              variant="secondary"
              size="md"
              icon="face"
              onClick={() => setActiveWorkspace('culling')}
            >
              Back to Face Organizer
            </DarkroomButton>
            <DarkroomButton
              variant="primary"
              size="md"
              icon="file_upload"
              onClick={() => setActiveWorkspace('export')}
            >
              Proceed to Batch Export
            </DarkroomButton>
          </div>
        </motion.div>
      </section>
    );
  }

  const { photo, candidateCluster, reason, source } = activeItem;
  const { anchor } = candidateCluster;
  const faceOnly = isFaceOnlySlateCode(anchor.slateCode);
  const scorePct = Math.min(100, Math.max(0, photo.sfaceCosineSimilarity * 100));
  const activeCutoff =
    source === 'cluster_low_confidence' ? reviewWarningThreshold : sfaceThreshold;
  const thresholdPct = Math.min(100, Math.max(0, activeCutoff * 100));
  const batchCount = checkedPhotoIds.size;

  const handleInspectInOrganizer = () => {
    selectCluster(candidateCluster.anchor.id);
    setActiveWorkspace('culling');
  };

  return (
    <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden bg-canvas">
      {/* Left Queue Sidebar with Multi-Select Checkboxes */}
      <aside
        aria-label="Pending Review Frames"
        className="w-[300px] shrink-0 bg-surface-low border-r border-hairline flex flex-col min-h-0 overflow-hidden"
      >
        <div className="h-11 px-3.5 bg-surface border-b border-hairline flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <span className="font-display text-[13px] font-semibold text-ink-primary">
              Verification Queue
            </span>
            <TelemetryPill tone="tangerine">{ambiguousReviewQueue.length}</TelemetryPill>
          </div>
          <span className="tabular-nums-mono text-[10px] text-ink-telemetry">A / S keys</span>
        </div>

        {/* Multi-Select Helper Strip */}
        <div className="px-3 py-1.5 bg-surface-low border-b border-hairline flex items-center justify-between gap-2 shrink-0">
          <span className="tabular-nums-mono text-[10px] text-ink-telemetry truncate">
            {batchCount > 0 ? `${batchCount} selected` : 'Check or Shift+Click to batch'}
          </span>

          <div className="flex items-center gap-2 shrink-0">
            {sameSubjectItemIds.length > 1 && (
              <button
                type="button"
                onClick={() => setCheckedPhotoIds(new Set(sameSubjectItemIds))}
                title={`Select all ${sameSubjectItemIds.length} review frames under ${anchor.subjectName}`}
                className="tabular-nums-mono text-[10px] text-amber-soft hover:underline cursor-pointer"
              >
                Same Subject ({sameSubjectItemIds.length})
              </button>
            )}
            {batchCount > 0 && (
              <button
                type="button"
                onClick={() => setCheckedPhotoIds(new Set())}
                className="tabular-nums-mono text-[10px] text-ink-secondary hover:text-ink-primary cursor-pointer"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        <div className="flex-1 min-h-0 flex flex-col p-2.5 overflow-hidden">
          <VirtualVerticalList
            items={ambiguousReviewQueue}
            rowHeightPx={68}
            gapPx={6}
            overscan={6}
            getKey={(item) => item.photo.id}
            className="flex-1 min-h-0 pr-0.5"
            renderItem={(item) => {
              const isSelected = item.photo.id === activeItem.photo.id;
              const isChecked = checkedPhotoIds.has(item.photo.id);
              return (
                <div
                  onClick={(e) => {
                    if (e.shiftKey || e.ctrlKey || e.metaKey) {
                      toggleChecked(item.photo.id, e.shiftKey);
                    }
                    setSelectedPhotoId(item.photo.id);
                  }}
                  className={`relative h-full p-2 rounded-lg border text-left flex items-center gap-2 transition-colors cursor-pointer select-none ${isSelected
                      ? 'bg-surface-active border-tangerine-alert/70'
                      : isChecked
                        ? 'bg-amber-core/10 border-amber-core/50'
                        : 'bg-surface border-hairline hover:bg-surface-active'
                    }`}
                >
                  {isSelected && (
                    <span
                      aria-hidden="true"
                      className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r bg-tangerine-alert"
                    />
                  )}

                  <input
                    type="checkbox"
                    checked={isChecked}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleChecked(item.photo.id, e.shiftKey);
                    }}
                    onChange={() => { }}
                    aria-label={`Select ${item.photo.fileName} for batch action`}
                    className="accent-amber-core shrink-0 cursor-pointer ml-0.5"
                  />

                  <img
                    src={item.photo.imageUrl}
                    alt={item.photo.fileName}
                    loading="lazy"
                    decoding="async"
                    draggable={false}
                    className="w-11 h-13 rounded object-cover object-top shrink-0 bg-surface-lowest"
                  />

                  <div className="flex-1 min-w-0 flex flex-col justify-between gap-0.5">
                    <div className="flex items-center justify-between gap-1">
                      <span className="tabular-nums-mono text-[11px] font-semibold text-ink-primary truncate">
                        {item.photo.fileName}
                      </span>
                      <span className="tabular-nums-mono text-[10px] text-tangerine-alert font-semibold shrink-0">
                        {item.photo.sfaceCosineSimilarity.toFixed(2)}
                      </span>
                    </div>
                    <span className="text-[11px] text-ink-secondary truncate">
                      → {item.candidateCluster.anchor.subjectName}
                    </span>
                    <span className="tabular-nums-mono text-[9px] text-ink-telemetry truncate">
                      {item.reason}
                    </span>
                  </div>
                </div>
              );
            }}
          />
        </div>
      </aside>

      {/* Main Viewport-Fit Side-by-Side Face Verification Stage (never overflows or clips) */}
      <section className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
        {/* Compact Top Telemetry Header */}
        <div className="px-5 py-2.5 bg-surface border-b border-hairline flex flex-wrap items-center justify-between gap-3 shrink-0">
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2 min-w-0">
              <h1 className="font-display text-[15.5px] font-bold text-ink-primary truncate">
                Side-by-Side Identity Review
              </h1>
              <TelemetryPill tone="tangerine" dot className="shrink-0">
                {reason}
              </TelemetryPill>
            </div>
            <span className="tabular-nums-mono text-[10px] text-ink-telemetry truncate">
              Candidate {photo.fileName} vs Anchor {anchor.subjectName} (
              {faceOnly ? 'Face-Only Cluster' : `Slate ${anchor.slateCode}`})
            </span>
          </div>

          <TelemetryPill tone="tangerine" className="shrink-0">
            Review &lt; {Math.round(reviewWarningThreshold * 100)}%
          </TelemetryPill>
        </div>

        {/* Flexible Center Portrait Comparison Stage — automatically scales to fit any viewport */}
        <div className="flex-1 min-h-0 p-4 flex items-center justify-center overflow-hidden">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={photo.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={FAST_FADE}
              className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-[960px] h-full min-h-0"
            >
              {/* Reference Anchor Card */}
              <div className="rounded-xl bg-surface border border-emerald-focus/40 overflow-hidden flex flex-col min-h-0 h-full">
                <div className="px-3 py-1.5 bg-surface-low border-b border-hairline flex items-center justify-between gap-2 shrink-0">
                  <span className="font-display text-[12.5px] font-bold text-ink-primary truncate">
                    Cluster Anchor: {anchor.subjectName}
                  </span>
                  <TelemetryPill tone={faceOnly ? 'amber' : 'emerald'} className="shrink-0">
                    {faceOnly ? 'FACE-ONLY' : `SLATE ${anchor.slateCode}`}
                  </TelemetryPill>
                </div>
                <div className="flex-1 min-h-0 p-2 bg-surface-lowest overflow-hidden">
                  <div className="w-full h-full flex items-center justify-center [container-type:size]">
                    <img
                      src={anchor.thumbnailUrl}
                      alt={anchor.subjectName}
                      decoding="async"
                      draggable={false}
                      className="max-w-[100cqw] max-h-[100cqh] w-auto h-auto rounded block"
                    />
                  </div>
                </div>
                <div className="px-3 py-1.5 border-t border-hairline tabular-nums-mono text-[10px] text-ink-telemetry flex items-center justify-between shrink-0">
                  <span className="truncate">{anchor.cameraLabel}</span>
                  <span className="shrink-0">{anchor.reportedTotalExposures} matched frames</span>
                </div>
              </div>

              {/* Candidate Ambiguous Frame Card */}
              <div className="rounded-xl bg-surface border border-tangerine-alert/55 overflow-hidden flex flex-col min-h-0 h-full">
                <div className="px-3 py-1.5 bg-surface-low border-b border-hairline flex items-center justify-between gap-2 shrink-0">
                  <span className="font-display text-[12.5px] font-bold text-ink-primary truncate">
                    Candidate: {photo.fileName}
                  </span>
                  <TelemetryPill tone="tangerine" dot className="shrink-0">
                    {formatSimilarityPercent(photo.sfaceCosineSimilarity)}
                  </TelemetryPill>
                </div>
                <div className="flex-1 min-h-0 p-2 bg-surface-lowest overflow-hidden">
                  <div className="w-full h-full flex items-center justify-center [container-type:size]">
                    <div className="relative inline-block">
                      <img
                        src={photo.imageUrl}
                        alt={photo.fileName}
                        decoding="async"
                        draggable={false}
                        className="max-w-[100cqw] max-h-[100cqh] w-auto h-auto rounded block"
                      />
                      {photo.faceOverlay && (
                        <div
                          className="absolute border-2 border-tangerine-alert rounded-sm pointer-events-none shadow-[0_0_16px_rgba(249,115,22,0.4)]"
                          style={{
                            top: `${photo.faceOverlay.topPct}%`,
                            left: `${photo.faceOverlay.leftPct}%`,
                            width: `${photo.faceOverlay.widthPct}%`,
                            height: `${photo.faceOverlay.heightPct}%`,
                          }}
                        />
                      )}
                    </div>
                  </div>
                </div>
                <div className="px-3 py-1.5 border-t border-hairline tabular-nums-mono text-[10px] text-ink-telemetry flex items-center justify-between shrink-0">
                  <span className="truncate">
                    {photo.exif.shutterSpeed} • {photo.exif.aperture} • ISO {photo.exif.iso}
                  </span>
                  <span
                    className={`shrink-0 ${photo.sharpnessPct < 65 ? 'text-tangerine-alert font-semibold' : ''
                      }`}
                  >
                    Sharpness {photo.sharpnessPct}%
                  </span>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Pinned Bottom Similarity Gauge + Single / Batch Action Command Deck */}
        <div className="px-5 py-3 bg-surface border-t border-hairline flex flex-col gap-2.5 shrink-0">
          {/* Compact Cosine Similarity Meter */}
          <div className="w-full max-w-[960px] mx-auto flex items-center gap-3">
            <span className="tabular-nums-mono text-[10.5px] text-ink-secondary shrink-0">
              Similarity vs Threshold ({Math.round(activeCutoff * 100)}%)
            </span>
            <div className="relative flex-1 h-2.5 rounded-full bg-canvas border border-hairline overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${scorePct}%` }}
                transition={SOFT_SPRING}
                className={`h-full ${photo.sfaceCosineSimilarity >= activeCutoff
                    ? 'bg-emerald-focus'
                    : 'bg-tangerine-alert'
                  }`}
              />
              <div
                style={{ left: `${thresholdPct}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-amber-core shadow-[0_0_8px_#F59E0B]"
              />
            </div>
            <span
              className={`tabular-nums-mono text-[11px] font-semibold shrink-0 ${photo.sfaceCosineSimilarity >= activeCutoff
                  ? 'text-emerald-bright'
                  : 'text-tangerine-alert'
                }`}
            >
              {formatSimilarityPercent(photo.sfaceCosineSimilarity)}
            </span>
          </div>

          {/* Triage Action Command Deck */}
          <div className="flex flex-wrap items-center justify-center gap-2.5">
            {batchCount > 0 ? (
              <>
                <DarkroomButton
                  variant="emerald"
                  size="md"
                  icon="check_circle"
                  onClick={() => void handleBatchApprove()}
                  className="px-4 h-9 text-[12.5px] font-display font-semibold"
                >
                  <span>Approve Selected ({batchCount})</span>
                  <kbd className="ml-1.5 tabular-nums-mono text-[10px] px-1.5 py-0.5 rounded bg-canvas border border-emerald-focus/40">
                    A
                  </kbd>
                </DarkroomButton>

                <DarkroomButton
                  variant="primary"
                  size="md"
                  icon="call_split"
                  onClick={() => void handleBatchSplitTogether()}
                  title={`Split all ${batchCount} selected photos together into one new subject`}
                  className="px-4 h-9 text-[12.5px] font-display font-semibold"
                >
                  <span>
                    Split Selected ({batchCount}) into 1 New Person
                  </span>
                  <kbd className="ml-1.5 tabular-nums-mono text-[10px] px-1.5 py-0.5 rounded bg-amber-ink/15 border border-amber-ink/30">
                    S
                  </kbd>
                </DarkroomButton>

                <DarkroomButton
                  variant="subtle"
                  size="md"
                  icon="face"
                  onClick={handleInspectInOrganizer}
                  className="px-3.5 h-9 text-[12px]"
                >
                  Inspect in Organizer
                </DarkroomButton>
              </>
            ) : (
              <>
                <DarkroomButton
                  variant="emerald"
                  size="md"
                  icon="check_circle"
                  onClick={() => resolveAmbiguousReview(photo.id, 'approve')}
                  className="px-4 h-9 text-[12.5px] font-display font-semibold"
                >
                  <span>Approve &amp; Keep in {anchor.subjectName}</span>
                  <kbd className="ml-1.5 tabular-nums-mono text-[10px] px-1.5 py-0.5 rounded bg-canvas border border-emerald-focus/40">
                    A
                  </kbd>
                </DarkroomButton>

                <DarkroomButton
                  variant="secondary"
                  size="md"
                  icon="call_split"
                  onClick={() => resolveAmbiguousReview(photo.id, 'split')}
                  className="px-4 h-9 text-[12.5px] font-display font-semibold"
                >
                  <span>Split into New Person</span>
                  <kbd className="ml-1.5 tabular-nums-mono text-[10px] px-1.5 py-0.5 rounded bg-canvas border border-hairline-strong">
                    S
                  </kbd>
                </DarkroomButton>

                <DarkroomButton
                  variant="subtle"
                  size="md"
                  icon="face"
                  onClick={handleInspectInOrganizer}
                  className="px-3.5 h-9 text-[12px]"
                >
                  Inspect in Organizer
                </DarkroomButton>
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
});
