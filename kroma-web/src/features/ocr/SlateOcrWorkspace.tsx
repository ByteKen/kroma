import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import { createDefaultDevelopRecipe } from '../../domain/develop';
import type { PhotoAsset } from '../../domain/photo';
import {
  classifySlateOcrConfidence,
  filterSlateOcrRows,
  isSlateOcrAlreadyApplied,
  type ClusterSlateOcrResult,
  type OcrTextLine,
  type SlateOcrFilter,
  type SlateRenameApplyItem,
} from '../../domain/slateOcr';
import {
  DarkroomButton,
  DarkroomToggle,
  FAST_FADE,
  PanelResizeHandle,
  SOFT_SPRING,
  TelemetryPill,
  useResizableWidth,
  VirtualVerticalList,
} from '../darkroom-kit';

interface DraftSlateEdit {
  readonly name: string;
  readonly code?: string;
  readonly slatePhotoId: string | null;
  readonly markAsSlatePhoto: boolean;
  readonly lines: readonly OcrTextLine[];
  readonly confidence: number;
  readonly proposedName: string | null;
  readonly scannedManually?: boolean;
}

const FILTER_OPTIONS: readonly { readonly id: SlateOcrFilter; readonly label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'ready', label: 'Ready' },
  { id: 'check', label: 'Needs Check' },
  { id: 'none', label: 'No Slate' },
  { id: 'applied', label: 'Applied' },
];

export function SlateOcrWorkspace() {
  const {
    session,
    runtimeInfo,
    scanProgress,
    isScanningFolder,
    isScanningOcr,
    slateOcrResults,
    scanProjectSlatesOcr,
    cancelSlateOcrScan,
    scanSinglePhotoOcr,
    applySlateOcrRenames,
    setActiveWorkspace,
    notifyAction,
  } = useStudio();

  const [filter, setFilter] = useState<SlateOcrFilter>('all');
  const [query, setQuery] = useState('');
  const [scanAllFrames, setScanAllFrames] = useState(false);
  const [scanMenuOpen, setScanMenuOpen] = useState(false);
  const [showOcrBoxes, setShowOcrBoxes] = useState(true);
  const [zoomSlateRegion, setZoomSlateRegion] = useState(false);
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [inspectedPhotoId, setInspectedPhotoId] = useState<string | null>(null);
  const [checkedIds, setCheckedIds] = useState<ReadonlySet<string>>(new Set());
  const [draftsByCluster, setDraftsByCluster] = useState<Record<string, DraftSlateEdit>>({});
  const [isRescanningSingle, setIsRescanningSingle] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const scanMenuRef = useRef<HTMLDivElement>(null);
  const lastSyncedOcrRef = useRef<Map<string, ClusterSlateOcrResult>>(new Map());

  const listPanel = useResizableWidth({
    storageKey: 'slate-ocr-list',
    defaultWidth: 420,
    minWidth: 330,
    maxWidth: 560,
  });

  // Close the scan scope popover on outside click or Escape
  useEffect(() => {
    if (!scanMenuOpen) return;
    const handlePointerDown = (event: MouseEvent) => {
      if (scanMenuRef.current && !scanMenuRef.current.contains(event.target as Node)) {
        setScanMenuOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setScanMenuOpen(false);
    };
    window.addEventListener('mousedown', handlePointerDown);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('mousedown', handlePointerDown);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [scanMenuOpen]);

  // Reset local drafts and selection when switching projects
  const activeProjectKey = session?.folderPath ?? session?.telemetry.projectName ?? null;
  useEffect(() => {
    setDraftsByCluster({});
    setCheckedIds(new Set());
    setScanMenuOpen(false);
    lastSyncedOcrRef.current.clear();
  }, [activeProjectKey]);

  // Only show real subject clusters when organized; if unclustered, allow inspecting all photos.
  const subjectClusters = useMemo(() => {
    if (!session) return [];
    const primary = session.clusters.filter((c) => !c.anchor.id.startsWith('_'));
    return primary.length > 0 ? primary : session.clusters;
  }, [session]);

  const ocrResultMap = useMemo(() => {
    const map = new Map<string, ClusterSlateOcrResult>();
    for (const r of slateOcrResults) {
      map.set(r.clusterId, r);
    }
    return map;
  }, [slateOcrResults]);

  // Incrementally sync newly scanned PP-OCRv5 subjects without wiping prior results or edits
  useEffect(() => {
    if (slateOcrResults.length === 0) {
      lastSyncedOcrRef.current.clear();
      return;
    }

    const updatedResults: ClusterSlateOcrResult[] = [];
    for (const r of slateOcrResults) {
      if (lastSyncedOcrRef.current.get(r.clusterId) !== r) {
        updatedResults.push(r);
        lastSyncedOcrRef.current.set(r.clusterId, r);
      }
    }
    if (updatedResults.length === 0) return;

    setDraftsByCluster((prev) => {
      const next = { ...prev };
      for (const r of updatedResults) {
        next[r.clusterId] = {
          name: r.proposedName ?? '',
          code: prev[r.clusterId]?.code,
          slatePhotoId: r.slatePhotoId,
          markAsSlatePhoto: Boolean(r.proposedName),
          lines: r.lines,
          confidence: r.confidence,
          proposedName: r.proposedName,
        };
      }
      return next;
    });

    setCheckedIds((prev) => {
      const next = new Set(prev);
      for (const r of updatedResults) {
        const tier = classifySlateOcrConfidence(r.proposedName, r.confidence);
        if (
          tier === 'ready' &&
          r.proposedName &&
          !isSlateOcrAlreadyApplied(r.currentName, r.proposedName)
        ) {
          next.add(r.clusterId);
        } else {
          next.delete(r.clusterId);
        }
      }
      return next;
    });
  }, [slateOcrResults]);

  const rows = useMemo(() => {
    return subjectClusters.map((cluster) => {
      const cid = cluster.anchor.id;
      const scanned = ocrResultMap.get(cid);
      const draft = draftsByCluster[cid];
      const existingSlatePhoto =
        cluster.photos.find((p) => p.isSlateAnchor) ?? cluster.photos[0] ?? null;
      const draftName = draft?.name ?? scanned?.proposedName ?? '';
      const generatedCode =
        cluster.anchor.subjectCode ??
        (cluster.anchor.slateCode !== 'NO SLATE' ? cluster.anchor.slateCode : '');
      const draftCode = draft?.code !== undefined ? draft.code : generatedCode;
      const confidence = draft?.confidence ?? scanned?.confidence ?? 0;
      const proposedName = draft?.proposedName ?? scanned?.proposedName ?? null;
      const slatePhotoId = draft?.slatePhotoId ?? scanned?.slatePhotoId ?? existingSlatePhoto?.id ?? null;
      const slatePhoto =
        cluster.photos.find((p) => p.id === slatePhotoId) ?? existingSlatePhoto;
      const slatePreviewUrl =
        slatePhoto?.imageUrl ?? scanned?.slatePreviewUrl ?? cluster.anchor.thumbnailUrl;
      const lines = draft?.lines ?? scanned?.lines ?? [];
      const markAsSlatePhoto = draft?.markAsSlatePhoto ?? Boolean(proposedName);
      const isApplied = isSlateOcrAlreadyApplied(cluster.anchor.subjectName, draftName);

      return {
        cluster,
        clusterId: cid,
        currentName: cluster.anchor.subjectName,
        draftName,
        draftCode,
        confidence,
        proposedName,
        slatePhotoId,
        slatePreviewUrl,
        lines,
        markAsSlatePhoto,
        isApplied,
        scannedFrameCount: scanned?.scannedFrameCount ?? 0,
        hasBeenScanned: Boolean(scanned || draft?.scannedManually),
      };
    });
  }, [subjectClusters, ocrResultMap, draftsByCluster]);

  const counts = useMemo(() => {
    let ready = 0;
    let check = 0;
    let none = 0;
    let applied = 0;
    let scanned = 0;
    for (const r of rows) {
      if (r.hasBeenScanned) scanned += 1;
      if (r.isApplied) applied += 1;
      const tier = classifySlateOcrConfidence(r.proposedName, r.confidence);
      if (tier === 'ready') ready += 1;
      else if (tier === 'check') check += 1;
      else none += 1;
    }
    return { ready, check, none, applied, scanned };
  }, [rows]);

  const unscannedIds = useMemo(
    () => rows.filter((r) => !r.hasBeenScanned).map((r) => r.clusterId),
    [rows],
  );

  const filteredRows = useMemo(
    () => filterSlateOcrRows(rows, filter, query),
    [rows, filter, query],
  );

  const activeRow = useMemo(() => {
    return (
      filteredRows.find((r) => r.clusterId === selectedClusterId) ??
      filteredRows[0] ??
      rows[0] ??
      null
    );
  }, [filteredRows, selectedClusterId, rows]);

  // Keep inspected photo synced with the selected subject's detected slate photo
  useEffect(() => {
    if (!activeRow) {
      setInspectedPhotoId(null);
      return;
    }
    setInspectedPhotoId(activeRow.slatePhotoId ?? activeRow.cluster.photos[0]?.id ?? null);
  }, [activeRow?.clusterId, activeRow?.slatePhotoId]);

  const inspectedPhoto: PhotoAsset | null = useMemo(() => {
    if (!activeRow) return null;
    if (inspectedPhotoId) {
      const found = activeRow.cluster.photos.find((p) => p.id === inspectedPhotoId);
      if (found) return found;
      // Standalone no-face whiteboard frame linked by timestamp (not in cluster.photos)
      if (inspectedPhotoId === activeRow.slatePhotoId && activeRow.slatePreviewUrl) {
        return {
          ...(activeRow.cluster.photos[0] ?? {
            id: inspectedPhotoId,
            fileName: 'Whiteboard_Slate',
            directoryId: 'dir-root',
            imageUrl: activeRow.slatePreviewUrl,
            isSlateAnchor: true,
            clusterId: activeRow.clusterId,
            sfaceCosineSimilarity: 1,
            sharpnessPct: 95,
            eyeAfLocked: false,
            heroSelect: false,
            exif: {
              shutterSpeed: 'SLATE',
              aperture: 'OCR',
              iso: 100,
              focalLengthMm: 'Native',
              cameraModel: 'Whiteboard',
              lensModel: 'PP-OCRv5',
            },
            faceOverlay: null,
            developRecipe: createDefaultDevelopRecipe(),
          }),
          id: inspectedPhotoId,
          imageUrl: activeRow.slatePreviewUrl,
          isSlateAnchor: true,
        };
      }
    }
    return activeRow.cluster.photos[0] ?? null;
  }, [activeRow, inspectedPhotoId]);

  const updateDraft = (clusterId: string, patch: Partial<DraftSlateEdit>) => {
    const existingRow = rows.find((r) => r.clusterId === clusterId);
    setDraftsByCluster((prev) => {
      const base: DraftSlateEdit = prev[clusterId] ?? {
        name: existingRow?.draftName ?? '',
        code: existingRow?.draftCode ?? '',
        slatePhotoId: existingRow?.slatePhotoId ?? null,
        markAsSlatePhoto: existingRow?.markAsSlatePhoto ?? true,
        lines: existingRow?.lines ?? [],
        confidence: existingRow?.confidence ?? 0,
        proposedName: existingRow?.proposedName ?? null,
      };
      return {
        ...prev,
        [clusterId]: { ...base, ...patch },
      };
    });
  };

  const toggleChecked = (clusterId: string) => {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(clusterId)) next.delete(clusterId);
      else next.add(clusterId);
      return next;
    });
  };

  const selectAllReady = () => {
    const next = new Set<string>();
    for (const r of rows) {
      if (
        r.draftName.trim() &&
        !r.isApplied &&
        classifySlateOcrConfidence(r.proposedName, r.confidence) === 'ready'
      ) {
        next.add(r.clusterId);
      }
    }
    setCheckedIds(next);
  };

  const selectVisibleBatch = () => {
    const limit = Math.min(10, filteredRows.length);
    const next = new Set<string>();
    for (let i = 0; i < limit; i += 1) {
      const row = filteredRows[i];
      if (row) next.add(row.clusterId);
    }
    setCheckedIds(next);
  };

  const handleStartScan = (clusterIds?: readonly string[]) => {
    setScanMenuOpen(false);
    void scanProjectSlatesOcr({
      scanAllFrames,
      clusterIds,
    });
  };

  const checkedApplyItems = useMemo<SlateRenameApplyItem[]>(() => {
    const items: SlateRenameApplyItem[] = [];
    for (const r of rows) {
      if (!checkedIds.has(r.clusterId)) continue;
      const cleanName = r.draftName.trim();
      if (!cleanName || r.isApplied) continue;
      items.push({
        clusterId: r.clusterId,
        nextName: cleanName,
        subjectCode: r.draftCode.trim() || null,
        slatePhotoId: r.slatePhotoId,
        ocrRawText: r.lines.map((l) => l.text).join(' | ') || cleanName,
        ocrConfidence: r.confidence > 0 ? r.confidence : 1.0,
        markAsSlatePhoto: r.markAsSlatePhoto,
      });
    }
    return items;
  }, [rows, checkedIds]);

  const handleApplyBatch = async () => {
    if (!session?.isClustered) {
      notifyAction('Run Face Organizer first so OCR names can rename each person’s folder');
      return;
    }
    if (checkedApplyItems.length === 0) return;
    setIsApplying(true);
    try {
      const ok = await applySlateOcrRenames(checkedApplyItems);
      if (ok) {
        setCheckedIds(new Set());
      }
    } finally {
      setIsApplying(false);
    }
  };

  const advanceToNextRow = (currentClusterId: string) => {
    const idx = filteredRows.findIndex((r) => r.clusterId === currentClusterId);
    const next = filteredRows[idx + 1] ?? filteredRows[0];
    if (next && next.clusterId !== currentClusterId) {
      setSelectedClusterId(next.clusterId);
    }
  };

  const handleApplySingleAndNext = async () => {
    if (!activeRow) return;
    if (!session?.isClustered) {
      notifyAction('Run Face Organizer first so OCR names can rename each person’s folder');
      return;
    }
    const cleanName = activeRow.draftName.trim();
    if (!cleanName) {
      notifyAction('Enter or click a whiteboard text line before applying');
      return;
    }
    setIsApplying(true);
    try {
      const ok = await applySlateOcrRenames([
        {
          clusterId: activeRow.clusterId,
          nextName: cleanName,
          subjectCode: activeRow.draftCode.trim() || null,
          slatePhotoId: inspectedPhoto?.id ?? activeRow.slatePhotoId,
          ocrRawText: activeRow.lines.map((l) => l.text).join(' | ') || cleanName,
          ocrConfidence: activeRow.confidence > 0 ? activeRow.confidence : 1.0,
          markAsSlatePhoto: activeRow.markAsSlatePhoto,
        },
      ]);
      if (ok) {
        setCheckedIds((prev) => {
          const next = new Set(prev);
          next.delete(activeRow.clusterId);
          return next;
        });
        advanceToNextRow(activeRow.clusterId);
      }
    } finally {
      setIsApplying(false);
    }
  };

  const handleRescanInspectedPhoto = async () => {
    if (!activeRow || !inspectedPhoto) return;
    setIsRescanningSingle(true);
    try {
      const res = await scanSinglePhotoOcr(inspectedPhoto.id);
      if (!res) return;
      updateDraft(activeRow.clusterId, {
        name: res.proposedName ?? activeRow.draftName,
        code: res.proposedCode ?? activeRow.draftCode,
        slatePhotoId: res.photoId,
        lines: res.lines,
        confidence: res.confidence,
        proposedName: res.proposedName,
        markAsSlatePhoto: Boolean(res.proposedName),
        scannedManually: true,
      });
      if (res.proposedName) {
        setCheckedIds((prev) => new Set(prev).add(activeRow.clusterId));
      }
    } finally {
      setIsRescanningSingle(false);
    }
  };

  // Compute zoom transform centered around detected whiteboard lines when "Zoom Whiteboard" is active
  const whiteboardZoomStyle = useMemo(() => {
    if (!zoomSlateRegion || !activeRow || activeRow.lines.length === 0) return undefined;
    let minL = 1;
    let minT = 1;
    let maxR = 0;
    let maxB = 0;
    for (const l of activeRow.lines) {
      minL = Math.min(minL, l.left);
      minT = Math.min(minT, l.top);
      maxR = Math.max(maxR, l.left + l.width);
      maxB = Math.max(maxB, l.top + l.height);
    }
    const cx = ((minL + maxR) * 0.5) * 100;
    const cy = ((minT + maxB) * 0.5) * 100;
    return {
      transformOrigin: `${cx.toFixed(1)}% ${cy.toFixed(1)}%`,
      transform: 'scale(2.15)',
    };
  }, [zoomSlateRegion, activeRow]);

  if (!session) return null;

  const ocrReady = runtimeInfo?.ocrModelsReady ?? false;
  const ocrProgressPct =
    scanProgress && scanProgress.phase === 'ocr' && scanProgress.total > 0
      ? Math.round((scanProgress.completed / scanProgress.total) * 100)
      : null;

  return (
    <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden bg-canvas">
      {/* Top Slate OCR Command Toolbar */}
      <header className="h-12 px-5 bg-surface border-b border-hairline flex items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-2.5 min-w-0">
          <span className="material-symbols-outlined text-[19px] text-amber-core shrink-0">
            document_scanner
          </span>
          <h1 className="font-display text-[14.5px] font-bold text-ink-primary shrink-0">
            Slate OCR
          </h1>
          <TelemetryPill tone={ocrReady ? 'emerald' : 'amber'} className="shrink-0">
            {ocrReady ? 'AI Ready' : 'AI Unavailable'}
          </TelemetryPill>
          {counts.scanned > 0 && (
            <div className="hidden lg:flex items-center gap-2 tabular-nums-mono text-[10.5px] text-ink-telemetry pl-2 border-l border-hairline">
              <span>
                {counts.scanned}/{rows.length} scanned
              </span>
              <span>•</span>
              <span className="text-emerald-bright">{counts.ready} ready</span>
              <span>•</span>
              <span className="text-amber-soft">{counts.check} check</span>
              <span>•</span>
              <span>{counts.applied} applied</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          {isScanningOcr ? (
            <DarkroomButton
              variant="secondary"
              size="sm"
              icon="stop_circle"
              onClick={() => void cancelSlateOcrScan()}
            >
              Stop Scan
            </DarkroomButton>
          ) : (
            <div ref={scanMenuRef} className="relative">
              <DarkroomButton
                variant="secondary"
                size="sm"
                icon="document_scanner"
                active={scanMenuOpen}
                aria-haspopup="menu"
                aria-expanded={scanMenuOpen}
                disabled={!ocrReady || isScanningFolder || rows.length === 0}
                onClick={() => setScanMenuOpen((prev) => !prev)}
              >
                <span>Scan Slates</span>
                <span className="material-symbols-outlined text-[15px] -mr-0.5 text-ink-telemetry">
                  {scanMenuOpen ? 'expand_less' : 'expand_more'}
                </span>
              </DarkroomButton>

              <AnimatePresence>
                {scanMenuOpen && (
                  <motion.div
                    role="menu"
                    aria-label="Choose slate scan scope"
                    initial={{ opacity: 0, y: -4, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -4, scale: 0.98 }}
                    transition={FAST_FADE}
                    className="absolute right-0 mt-1.5 w-80 rounded-xl bg-surface-elevated border border-hairline-strong shadow-[0_18px_48px_rgba(0,0,0,0.55)] p-1.5 z-40"
                  >
                    <div className="px-2.5 pt-1 pb-1.5 tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                      Choose What to Scan
                    </div>

                    <button
                      type="button"
                      role="menuitem"
                      disabled={!activeRow || !inspectedPhoto || isRescanningSingle}
                      onClick={() => {
                        setScanMenuOpen(false);
                        void handleRescanInspectedPhoto();
                      }}
                      className="w-full px-2.5 py-2 rounded-lg text-left flex items-start gap-2.5 hover:bg-surface-active transition-colors cursor-pointer disabled:opacity-40 disabled:pointer-events-none"
                    >
                      <span className="material-symbols-outlined text-[17px] text-amber-soft mt-0.5 shrink-0">
                        photo
                      </span>
                      <span className="flex-1 min-w-0">
                        <span className="block text-[12px] font-medium text-ink-primary truncate">
                          Scan Current Frame
                        </span>
                        <span className="block tabular-nums-mono text-[10px] text-ink-telemetry truncate">
                          {inspectedPhoto
                            ? `${inspectedPhoto.fileName} • 1 photo`
                            : 'Select a photo first'}
                        </span>
                      </span>
                    </button>

                    <button
                      type="button"
                      role="menuitem"
                      disabled={!activeRow}
                      onClick={() => activeRow && handleStartScan([activeRow.clusterId])}
                      className="w-full px-2.5 py-2 rounded-lg text-left flex items-start gap-2.5 hover:bg-surface-active transition-colors cursor-pointer disabled:opacity-40 disabled:pointer-events-none"
                    >
                      <span className="material-symbols-outlined text-[17px] text-amber-core mt-0.5 shrink-0">
                        person_search
                      </span>
                      <span className="flex-1 min-w-0">
                        <span className="block text-[12px] font-medium text-ink-primary truncate">
                          Scan Current Subject
                        </span>
                        <span className="block tabular-nums-mono text-[10px] text-ink-telemetry truncate">
                          {activeRow
                            ? `${activeRow.currentName} • ${activeRow.cluster.photos.length} frame${activeRow.cluster.photos.length === 1 ? '' : 's'}`
                            : 'Select a subject first'}
                        </span>
                      </span>
                    </button>

                    <button
                      type="button"
                      role="menuitem"
                      disabled={checkedIds.size === 0}
                      onClick={() => handleStartScan(Array.from(checkedIds))}
                      className="w-full px-2.5 py-2 rounded-lg text-left flex items-start gap-2.5 hover:bg-surface-active transition-colors cursor-pointer disabled:opacity-45 disabled:pointer-events-none"
                    >
                      <span className="material-symbols-outlined text-[17px] text-emerald-bright mt-0.5 shrink-0">
                        checklist
                      </span>
                      <span className="flex-1 min-w-0">
                        <span className="block text-[12px] font-medium text-ink-primary">
                          Scan Checked Subjects ({checkedIds.size})
                        </span>
                        <span className="block text-[10px] text-ink-telemetry">
                          {checkedIds.size > 0
                            ? `Scan ${checkedIds.size} checked subject${checkedIds.size === 1 ? '' : 's'}`
                            : 'Check subjects on the left first'}
                        </span>
                      </span>
                    </button>

                    {unscannedIds.length > 0 && unscannedIds.length < rows.length && (
                      <button
                        type="button"
                        role="menuitem"
                        onClick={() => handleStartScan(unscannedIds)}
                        className="w-full px-2.5 py-2 rounded-lg text-left flex items-start gap-2.5 hover:bg-surface-active transition-colors cursor-pointer"
                      >
                        <span className="material-symbols-outlined text-[17px] text-amber-soft mt-0.5 shrink-0">
                          filter_none
                        </span>
                        <span className="flex-1 min-w-0">
                          <span className="block text-[12px] font-medium text-ink-primary">
                            Scan Unscanned Only ({unscannedIds.length})
                          </span>
                          <span className="block tabular-nums-mono text-[10px] text-ink-telemetry">
                            Skips {rows.length - unscannedIds.length} already scanned
                          </span>
                        </span>
                      </button>
                    )}

                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => handleStartScan(undefined)}
                      className="w-full px-2.5 py-2 rounded-lg text-left flex items-start gap-2.5 hover:bg-surface-active transition-colors cursor-pointer"
                    >
                      <span className="material-symbols-outlined text-[17px] text-ink-secondary mt-0.5 shrink-0">
                        select_all
                      </span>
                      <span className="flex-1 min-w-0">
                        <span className="block text-[12px] font-medium text-ink-primary">
                          Scan All Subjects ({rows.length})
                        </span>
                        <span className="block text-[10px] text-ink-telemetry">
                          Scan whiteboards across the entire project
                        </span>
                      </span>
                    </button>

                    <div className="my-1.5 border-t border-hairline" />

                    <label className="px-2.5 py-1.5 rounded-lg flex items-start gap-2 hover:bg-surface-active/60 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={scanAllFrames}
                        onChange={(e) => setScanAllFrames(e.target.checked)}
                        className="accent-amber-core rounded mt-0.5 cursor-pointer shrink-0"
                      />
                      <span className="flex-1 min-w-0">
                        <span className="block text-[11px] font-medium text-ink-primary">
                          Check every frame per subject
                        </span>
                        <span className="block text-[10px] text-ink-telemetry leading-snug">
                          Off stops on the first whiteboard read (fastest).
                        </span>
                      </span>
                    </label>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          <DarkroomButton
            variant="primary"
            size="sm"
            icon="check_circle"
            disabled={
              isApplying ||
              isScanningOcr ||
              !session.isClustered ||
              checkedApplyItems.length === 0
            }
            onClick={() => void handleApplyBatch()}
          >
            Apply Selected ({checkedApplyItems.length})
          </DarkroomButton>
        </div>
      </header>

      {/* Live PP-OCRv5 Scan Progress Strip */}
      <AnimatePresence initial={false}>
        {isScanningOcr && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={FAST_FADE}
            className="overflow-hidden shrink-0 bg-surface-low border-b border-hairline"
          >
            <div className="px-5 py-2 flex flex-col gap-1.5">
              <div className="flex items-center justify-between gap-3 tabular-nums-mono text-[10.5px]">
                <span className="text-amber-soft truncate">
                  {scanProgress?.phase === 'ocr'
                    ? scanProgress.message
                    : 'Reading whiteboard handwriting…'}
                </span>
                <span className="text-ink-primary font-semibold shrink-0">
                  {ocrProgressPct !== null ? `${ocrProgressPct}%` : 'Warming up…'}
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-canvas overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${ocrProgressPct ?? 12}%` }}
                  transition={SOFT_SPRING}
                  className="h-full bg-amber-core"
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Prompt when Face Organizer hasn't grouped photos into people yet */}
      {!session.isClustered && (
        <div className="px-5 py-2.5 bg-amber-core/10 border-b border-amber-core/35 flex flex-wrap items-center justify-between gap-3 shrink-0">
          <div className="text-[11.5px] text-ink-primary">
            <strong>Separate Workflow:</strong> Run <strong>Face Organizer</strong> first so each
            whiteboard name renames that person&apos;s entire portrait folder — or scan slates now
            to inspect whiteboard readings per photo.
          </div>
          <DarkroomButton
            variant="emerald"
            size="xs"
            icon="face"
            onClick={() => setActiveWorkspace('culling')}
          >
            Go to Face Organizer
          </DarkroomButton>
        </div>
      )}

      {/* Main Split View: Left Virtualized Subject List + Right Interactive Whiteboard Loupe */}
      <div className="flex-1 min-h-0 flex overflow-hidden">
        {/* Left Master Column */}
        <aside
          aria-label="Subject Whiteboard Results"
          style={{ width: `${listPanel.width}px` }}
          className="shrink-0 bg-surface-low flex flex-col min-h-0 overflow-hidden"
        >
          {/* Search + Filter Chips */}
          <div className="p-3 bg-surface border-b border-hairline flex flex-col gap-2 shrink-0">
            <div className="flex items-center gap-2 h-8 px-2.5 rounded-md bg-canvas border border-hairline focus-within:border-amber-core">
              <span className="material-symbols-outlined text-[15px] text-ink-telemetry shrink-0">
                search
              </span>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={`Filter ${rows.length} subjects or OCR names…`}
                className="bg-transparent text-[11.5px] text-ink-primary placeholder:text-ink-telemetry flex-1 min-w-0 outline-none"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery('')}
                  className="text-ink-telemetry hover:text-ink-primary cursor-pointer flex items-center"
                >
                  <span className="material-symbols-outlined text-[14px]">close</span>
                </button>
              )}
            </div>

            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1 flex-wrap">
                {FILTER_OPTIONS.map((opt) => {
                  const active = filter === opt.id;
                  return (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => setFilter(opt.id)}
                      className={`px-2 py-0.5 rounded-md tabular-nums-mono text-[10px] border transition-colors cursor-pointer ${
                        active
                          ? 'bg-amber-core/15 border-amber-core/50 text-amber-soft font-medium'
                          : 'bg-canvas border-hairline text-ink-telemetry hover:text-ink-primary'
                      }`}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>

              {checkedIds.size > 0 ? (
                <button
                  type="button"
                  onClick={() => setCheckedIds(new Set())}
                  className="tabular-nums-mono text-[10px] text-amber-soft hover:underline cursor-pointer shrink-0"
                >
                  Clear ({checkedIds.size})
                </button>
              ) : counts.ready > 0 ? (
                <button
                  type="button"
                  onClick={selectAllReady}
                  className="tabular-nums-mono text-[10px] text-amber-soft hover:underline cursor-pointer shrink-0"
                >
                  Select Ready
                </button>
              ) : filteredRows.length > 0 ? (
                <button
                  type="button"
                  onClick={selectVisibleBatch}
                  className="tabular-nums-mono text-[10px] text-amber-soft hover:underline cursor-pointer shrink-0"
                >
                  {filteredRows.length <= 10
                    ? `Select Visible (${filteredRows.length})`
                    : 'Select First 10'}
                </button>
              ) : null}
            </div>
          </div>

          {/* Virtualized Rows */}
          <div className="flex-1 min-h-0 p-2.5 flex flex-col overflow-hidden">
            {filteredRows.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 px-6 text-ink-telemetry">
                <span className="material-symbols-outlined text-[26px]">document_scanner</span>
                <span className="text-[12px] text-ink-secondary">
                  {rows.length === 0
                    ? 'No subjects in this project.'
                    : 'No subjects match this OCR filter.'}
                </span>
              </div>
            ) : (
              <VirtualVerticalList
                items={filteredRows}
                rowHeightPx={74}
                gapPx={6}
                overscan={8}
                getKey={(item) => item.clusterId}
                className="flex-1 min-h-0 pr-1"
                renderItem={(row) => {
                  const isSelected = activeRow?.clusterId === row.clusterId;
                  const isChecked = checkedIds.has(row.clusterId);
                  const tier = classifySlateOcrConfidence(row.proposedName, row.confidence);
                  const confPct = Math.round(row.confidence * 100);

                  return (
                    <div
                      onClick={() => setSelectedClusterId(row.clusterId)}
                      className={`w-full h-full px-2.5 rounded-lg border flex items-center gap-2.5 transition-colors cursor-pointer ${
                        isSelected
                          ? 'bg-surface-active border-amber-core/60'
                          : 'bg-surface border-hairline hover:bg-surface-elevated hover:border-hairline-strong'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onClick={(e) => e.stopPropagation()}
                        onChange={() => toggleChecked(row.clusterId)}
                        aria-label={`Select ${row.currentName} for scanning or batch rename`}
                        className="accent-amber-core shrink-0 cursor-pointer"
                      />

                      <div className="w-10 h-12 rounded overflow-hidden bg-surface-lowest border border-hairline shrink-0">
                        {row.slatePreviewUrl ? (
                          <img
                            src={row.slatePreviewUrl}
                            alt={row.currentName}
                            loading="lazy"
                            decoding="async"
                            className="w-full h-full object-cover object-top"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-ink-telemetry">
                            <span className="material-symbols-outlined text-[15px]">person</span>
                          </div>
                        )}
                      </div>

                      <div className="flex-1 min-w-0 flex flex-col gap-1">
                        <div className="flex items-center justify-between gap-1.5">
                          <span className="tabular-nums-mono text-[10.5px] text-ink-telemetry truncate">
                            {row.currentName}
                            {row.draftCode ? ` • ${row.draftCode}` : ''}
                          </span>
                          <div className="flex items-center gap-1 shrink-0">
                            {row.isApplied && (
                              <span className="px-1.5 py-0.5 rounded bg-emerald-focus/15 border border-emerald-focus/35 tabular-nums-mono text-[9px] text-emerald-bright">
                                Applied
                              </span>
                            )}
                            {row.hasBeenScanned && (
                              <span
                                className={`px-1.5 py-0.5 rounded tabular-nums-mono text-[9px] border ${
                                  tier === 'ready'
                                    ? 'bg-emerald-focus/12 border-emerald-focus/35 text-emerald-bright'
                                    : tier === 'check'
                                      ? 'bg-amber-core/15 border-amber-core/40 text-amber-soft'
                                      : 'bg-canvas border-hairline text-ink-telemetry'
                                }`}
                              >
                                {tier === 'none' ? 'No slate' : `${confPct}%`}
                              </span>
                            )}
                          </div>
                        </div>

                        <input
                          type="text"
                          value={row.draftName}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedClusterId(row.clusterId);
                          }}
                          onChange={(e) => {
                            const val = e.target.value;
                            updateDraft(row.clusterId, { name: val });
                            if (val.trim()) {
                              setCheckedIds((prev) => new Set(prev).add(row.clusterId));
                            }
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              void handleApplySingleAndNext();
                            }
                          }}
                          placeholder={
                            row.hasBeenScanned
                              ? 'Type subject name or click photo text…'
                              : 'Check to batch-scan, or click Scan…'
                          }
                          aria-label={`Proposed OCR name for ${row.currentName}`}
                          className="h-6 px-2 rounded bg-canvas/90 border border-hairline/80 focus:border-amber-core font-display text-[12px] font-medium text-ink-primary placeholder:text-ink-telemetry/70 outline-none"
                        />
                      </div>
                    </div>
                  );
                }}
              />
            )}
          </div>
        </aside>

        <PanelResizeHandle
          panelSide="left"
          width={listPanel.width}
          minWidth={330}
          maxWidth={560}
          onChangeWidth={listPanel.setWidth}
          onResetWidth={listPanel.resetWidth}
          label="Resize Slate OCR subject list"
        />

        {/* Right Visual Whiteboard Inspector & Interactive Box Picker */}
        <section
          aria-label="Whiteboard OCR Inspector"
          className="flex-1 min-w-0 flex flex-col overflow-hidden bg-canvas"
        >
          {activeRow ? (
            <>
              {/* Inspector Sub-Header */}
              <div className="h-11 px-4 bg-surface border-b border-hairline flex items-center justify-between gap-3 shrink-0">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-display text-[13px] font-bold text-ink-primary truncate">
                    {activeRow.currentName}
                  </span>
                  <span className="tabular-nums-mono text-[10.5px] text-ink-telemetry shrink-0">
                    {activeRow.cluster.photos.length} photo
                    {activeRow.cluster.photos.length === 1 ? '' : 's'}
                  </span>
                  {inspectedPhoto && (
                    <span className="tabular-nums-mono text-[10.5px] text-ink-telemetry truncate hidden sm:inline">
                      • {inspectedPhoto.fileName}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <DarkroomButton
                    variant="ghost"
                    size="xs"
                    icon="crop_free"
                    active={showOcrBoxes}
                    aria-pressed={showOcrBoxes}
                    onClick={() => setShowOcrBoxes((prev) => !prev)}
                    title={showOcrBoxes ? 'Hide OCR boxes' : 'Show OCR boxes'}
                  >
                    OCR Boxes
                  </DarkroomButton>
                  <DarkroomButton
                    variant="ghost"
                    size="xs"
                    icon={zoomSlateRegion ? 'zoom_out' : 'zoom_in'}
                    active={zoomSlateRegion}
                    aria-pressed={zoomSlateRegion}
                    disabled={activeRow.lines.length === 0}
                    onClick={() => setZoomSlateRegion((prev) => !prev)}
                    title="Zoom into the detected whiteboard area"
                  >
                    {zoomSlateRegion ? '1× Fit' : 'Zoom Slate'}
                  </DarkroomButton>
                  <DarkroomButton
                    variant="secondary"
                    size="xs"
                    icon={
                      isRescanningSingle
                        ? 'progress_activity'
                        : activeRow.hasBeenScanned
                          ? 'refresh'
                          : 'document_scanner'
                    }
                    disabled={!ocrReady || isRescanningSingle || !inspectedPhoto}
                    onClick={() => void handleRescanInspectedPhoto()}
                    title="Scan only the single frame currently shown"
                  >
                    {isRescanningSingle
                      ? 'Reading…'
                      : activeRow.hasBeenScanned
                        ? 'Re-scan Frame'
                        : 'Scan This Frame'}
                  </DarkroomButton>
                </div>
              </div>

              {/* Large Interactive Whiteboard Loupe with Exact Normalized OCR Bounding Boxes */}
              <div className="flex-1 min-h-0 p-4 flex items-center justify-center overflow-hidden bg-surface-lowest/50 relative">
                {inspectedPhoto?.imageUrl ? (
                  <div className="relative max-w-full max-h-full flex items-center justify-center overflow-hidden rounded-xl border border-hairline bg-canvas shadow-[0_18px_48px_rgba(0,0,0,0.45)]">
                    <div
                      className="relative inline-block transition-transform duration-200 ease-out"
                      style={whiteboardZoomStyle}
                    >
                      <img
                        src={inspectedPhoto.imageUrl}
                        alt={inspectedPhoto.fileName}
                        className="block max-h-[calc(100vh-370px)] max-w-full w-auto h-auto select-none"
                      />
                      {showOcrBoxes &&
                        activeRow.lines.map((line, idx) => {
                          const isHigh = line.confidence >= 0.85;
                          return (
                            <button
                              key={`${line.text}-${idx}`}
                              type="button"
                              onClick={() => {
                                updateDraft(activeRow.clusterId, { name: line.text });
                                setCheckedIds((prev) => new Set(prev).add(activeRow.clusterId));
                                nameInputRef.current?.focus();
                              }}
                              title={`Click to use "${line.text}" (${Math.round(line.confidence * 100)}%)`}
                              style={{
                                left: `${line.left * 100}%`,
                                top: `${line.top * 100}%`,
                                width: `${line.width * 100}%`,
                                height: `${line.height * 100}%`,
                              }}
                              className={`absolute border-2 rounded-sm transition-colors cursor-pointer group ${
                                isHigh
                                  ? 'border-emerald-focus bg-emerald-focus/12 hover:bg-emerald-focus/28'
                                  : 'border-amber-core bg-amber-core/15 hover:bg-amber-core/30'
                              }`}
                            >
                              <span
                                className={`pointer-events-none absolute -top-4 left-0 px-1 py-0.5 rounded text-[8.5px] leading-none font-semibold tabular-nums-mono whitespace-nowrap shadow group-hover:z-20 ${
                                  isHigh
                                    ? 'bg-emerald-focus/95 text-amber-ink'
                                    : 'bg-amber-core/95 text-amber-ink'
                                }`}
                              >
                                {line.text} ({Math.round(line.confidence * 100)}%)
                              </span>
                            </button>
                          );
                        })}
                    </div>
                  </div>
                ) : (
                  <div className="text-[12px] text-ink-telemetry">
                    Select a subject to inspect their whiteboard frame.
                  </div>
                )}
              </div>

              {/* Cluster Frame Selector Strip (in case whiteboard isn't frame #1) */}
              {activeRow.cluster.photos.length > 1 && (
                <div className="px-4 py-2 bg-surface-low border-t border-hairline flex items-center gap-2 overflow-x-auto shrink-0">
                  <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry shrink-0 mr-1">
                    Frames
                  </span>
                  {activeRow.cluster.photos.map((photo) => {
                    const isViewing = inspectedPhoto?.id === photo.id;
                    const isDetectedSlate = activeRow.slatePhotoId === photo.id || photo.isSlateAnchor;
                    return (
                      <button
                        key={photo.id}
                        type="button"
                        onClick={() => setInspectedPhotoId(photo.id)}
                        className={`relative w-12 h-14 rounded-md overflow-hidden border-2 shrink-0 transition-all cursor-pointer ${
                          isViewing
                            ? 'border-amber-core ring-2 ring-amber-core/30'
                            : 'border-hairline hover:border-hairline-strong opacity-80 hover:opacity-100'
                        }`}
                        title={photo.fileName}
                      >
                        <img
                          src={photo.imageUrl}
                          alt={photo.fileName}
                          loading="lazy"
                          className="w-full h-full object-cover object-top"
                        />
                        {isDetectedSlate && (
                          <span className="absolute bottom-0 inset-x-0 bg-amber-core/90 text-amber-ink text-[8px] font-bold uppercase text-center leading-tight py-0.5">
                            Slate
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Bottom Rename & Detected Lines Action Bar */}
              <div className="p-4 bg-surface border-t border-hairline flex flex-col gap-3 shrink-0">
                {/* Clickable Detected OCR Lines */}
                {activeRow.lines.length > 0 && (
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry mr-1">
                      Detected Text:
                    </span>
                    {activeRow.lines.map((line, i) => (
                      <div
                        key={`${line.text}-${i}`}
                        className="inline-flex items-center rounded-md bg-canvas border border-hairline overflow-hidden"
                      >
                        <button
                          type="button"
                          onClick={() => {
                            updateDraft(activeRow.clusterId, { name: line.text });
                            setCheckedIds((prev) => new Set(prev).add(activeRow.clusterId));
                          }}
                          title="Set as subject name"
                          className="px-2 py-0.5 text-[11px] text-ink-primary hover:bg-surface-elevated transition-colors cursor-pointer"
                        >
                          {line.text}{' '}
                          <span className="tabular-nums-mono text-[9.5px] text-ink-telemetry">
                            {Math.round(line.confidence * 100)}%
                          </span>
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            const merged = [activeRow.draftName.trim(), line.text]
                              .filter(Boolean)
                              .join(' ');
                            updateDraft(activeRow.clusterId, { name: merged });
                            setCheckedIds((prev) => new Set(prev).add(activeRow.clusterId));
                          }}
                          title="Append to subject name"
                          className="px-1.5 py-0.5 border-l border-hairline text-[10px] text-amber-soft hover:bg-amber-core/15 cursor-pointer"
                        >
                          +
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-[1fr_140px_auto] gap-2.5 items-end">
                  <label className="flex flex-col gap-1">
                    <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                      Subject Folder Name
                    </span>
                    <input
                      ref={nameInputRef}
                      type="text"
                      value={activeRow.draftName}
                      onChange={(e) => {
                        const val = e.target.value;
                        updateDraft(activeRow.clusterId, { name: val });
                        if (val.trim()) {
                          setCheckedIds((prev) => new Set(prev).add(activeRow.clusterId));
                        }
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          void handleApplySingleAndNext();
                        }
                      }}
                      placeholder="e.g. Dela Cruz, Juan"
                      className="h-9 px-3 rounded-lg bg-canvas border border-hairline focus:border-amber-core font-display text-[13px] font-semibold text-ink-primary outline-none"
                    />
                  </label>

                  <label className="flex flex-col gap-1">
                    <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                      Slate # (Auto)
                    </span>
                    <input
                      type="text"
                      value={activeRow.draftCode}
                      onChange={(e) => updateDraft(activeRow.clusterId, { code: e.target.value })}
                      placeholder="Auto"
                      title="Auto-generated sequential subject number — edit only if you need a custom code"
                      className="h-9 px-3 rounded-lg bg-canvas border border-hairline focus:border-amber-core tabular-nums-mono text-[12px] text-amber-soft outline-none"
                    />
                  </label>

                  <div className="flex items-center gap-2">
                    <DarkroomButton
                      variant="ghost"
                      size="md"
                      onClick={() => advanceToNextRow(activeRow.clusterId)}
                    >
                      Skip
                    </DarkroomButton>
                    <DarkroomButton
                      variant="primary"
                      size="md"
                      icon="check"
                      disabled={
                        isApplying || !session.isClustered || !activeRow.draftName.trim()
                      }
                      onClick={() => void handleApplySingleAndNext()}
                    >
                      Apply &amp; Next
                    </DarkroomButton>
                  </div>
                </div>

                <DarkroomToggle
                  checked={activeRow.markAsSlatePhoto}
                  onChange={(val) =>
                    updateDraft(activeRow.clusterId, { markAsSlatePhoto: val })
                  }
                  label="Mark this frame as Whiteboard / Slate photo (auto-excluded from client packs on Export)"
                />
              </div>
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 p-8 text-ink-telemetry">
              <span className="material-symbols-outlined text-[32px]">document_scanner</span>
              <div className="font-display text-[14px] font-semibold text-ink-primary">
                Ready for Whiteboard Scanning
              </div>
              <p className="text-[12px] text-ink-secondary max-w-md">
                Click <strong>Scan Slates</strong> above to scan the current subject, a
                checked batch, or the entire project locally with zero cloud uploads.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
