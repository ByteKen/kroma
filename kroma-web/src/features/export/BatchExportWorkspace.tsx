import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import { isRecipeModified } from '../../domain/develop';
import { GradedThumbnailImg } from '../develop/GradedThumbnailImg';
import {
  buildDefaultExportLocation,
  estimateExportSize as estimateExportSizeLocal,
  fromMeasuredExportSize,
  joinExportPath,
  type ExportDeliveryMode,
  type ExportRenderFormat,
  type MeasuredExportSizePayload,
} from '../../domain/exportEstimate';
import { isFaceOnlySlateCode } from '../../domain/identity';
import {
  allocateUniqueExportPath,
  buildSimpleExportPattern,
  formatExportPreviewPath,
  inferSimpleExportNaming,
  resolveUniqueExportSubjectNames,
  type ExportFileNamingMode,
} from '../../domain/organization';
import type { PhotoAsset } from '../../domain/photo';
import { resolveVirtualFolderPath } from '../../domain/virtualFolder';
import {
  DarkroomButton,
  DarkroomScrubber,
  DarkroomToggle,
  FAST_FADE,
  PanelResizeHandle,
  SOFT_SPRING,
  SegmentedControl,
  TelemetryPill,
  useResizableWidth,
  VirtualVerticalList,
  type SegmentedOption,
} from '../darkroom-kit';

type DeliveryMode = ExportDeliveryMode;
type RenderFormat = ExportRenderFormat;

const DELIVERY_MODES: readonly SegmentedOption<DeliveryMode>[] = [
  {
    id: 'develop',
    label: 'Edited',
    title: 'Export edited photos only — original files stay untouched',
  },
  {
    id: 'develop_and_raw',
    label: 'Edited + RAW',
    title: 'Export both edited photos (Edited/) and untouched originals (RAW/)',
  },
  {
    id: 'copy',
    label: 'Copy',
    title: 'Copy original files into organized folders without edits',
  },
  {
    id: 'move',
    label: 'Move',
    title: 'Move original files out of the source folder into organized folders',
  },
];

const RENDER_FORMATS: readonly SegmentedOption<RenderFormat>[] = [
  { id: 'jpeg', label: 'JPEG', title: 'Standard web & client delivery (smallest files)' },
  { id: 'png', label: 'PNG', title: 'Lossless 8-bit PNG' },
  { id: 'tiff16', label: '16-bit TIFF', title: 'Uncompressed archival print TIFF' },
];

const FILE_NAMING_MODES: readonly SegmentedOption<ExportFileNamingMode>[] = [
  {
    id: 'original',
    label: 'Keep Original',
    title: 'Keep each photo’s original camera filename (e.g. DSC04123.jpg)',
  },
  {
    id: 'sequential',
    label: 'Sequential (001)',
    title: 'Number photos sequentially inside each folder (IMG_001.jpg, IMG_002.jpg)',
  },
];

const NAMING_TOKENS = [
  { token: '[Folder]', label: 'Virtual Folder' },
  { token: '[Slate_Name]', label: 'Subject Name' },
  { token: '[Original_Name]', label: 'Original File' },
  { token: '[ID]', label: 'Subject #' },
  { token: '[Seq]', label: 'Number (001)' },
] as const;

function formatExtension(mode: DeliveryMode, format: RenderFormat, sourceFileName: string): string {
  if (mode === 'copy' || mode === 'move') {
    const ext = sourceFileName.split('.').pop();
    return ext ? `.${ext}` : '.ARW';
  }
  if (format === 'png') return '.png';
  if (format === 'tiff16') return '.tiff';
  return '.jpg';
}

function splitPreviewIntoEditedAndRaw(
  renderedPath: string,
  originalFileName: string,
): { editedPath: string; rawPath: string } {
  const slashIdx = renderedPath.lastIndexOf('/');
  if (slashIdx <= 0) {
    return {
      editedPath: `/Edited/${renderedPath.replace(/^\//, '')}`,
      rawPath: `/RAW/${originalFileName}`,
    };
  }
  const dir = renderedPath.slice(0, slashIdx);
  const file = renderedPath.slice(slashIdx + 1);
  return {
    editedPath: `${dir}/Edited/${file}`,
    rawPath: `${dir}/RAW/${originalFileName}`,
  };
}

export function BatchExportWorkspace() {
  const {
    isDesktopRuntime,
    session,
    preferences,
    outputRule,
    ambiguousReviewQueue,
    updatePreferences,
    setExcludeSlateFromClientPack,
    setOutputPattern,
    pickExportDirectory,
    estimateExportSize: measureExportSizeOnBackend,
    runBatchExport,
    exportJob,
    runFaceOrganizer,
    setActiveWorkspace,
    isScanningFolder,
  } = useStudio();

  const [deliveryMode, setDeliveryModeState] = useState<DeliveryMode>(
    preferences.defaultDeliveryMode,
  );
  const [renderFormat, setRenderFormatState] = useState<RenderFormat>(
    preferences.defaultRenderFormat,
  );
  const [jpegQuality, setJpegQualityState] = useState(preferences.defaultJpegQuality);
  const [customExportBaseDir, setCustomExportBaseDirState] = useState<string | null>(
    preferences.defaultExportBaseDir,
  );
  const [createProjectSubfolder, setCreateProjectSubfolderState] = useState(
    preferences.createExportProjectSubfolder,
  );
  const [archiveSlatesSeparately, setArchiveSlatesSeparatelyState] = useState(
    preferences.archiveSlatesSeparately,
  );

  const setDeliveryMode = (mode: DeliveryMode) => {
    setDeliveryModeState(mode);
    updatePreferences({ defaultDeliveryMode: mode });
  };
  const setRenderFormat = (format: RenderFormat) => {
    setRenderFormatState(format);
    updatePreferences({ defaultRenderFormat: format });
  };
  const setJpegQuality = (quality: number) => {
    setJpegQualityState(quality);
    updatePreferences({ defaultJpegQuality: Math.round(quality) });
  };
  const setCustomExportBaseDir = (dir: string | null) => {
    setCustomExportBaseDirState(dir);
    updatePreferences({ defaultExportBaseDir: dir });
  };
  const setCreateProjectSubfolder = (enabled: boolean) => {
    setCreateProjectSubfolderState(enabled);
    updatePreferences({ createExportProjectSubfolder: enabled });
  };
  const setArchiveSlatesSeparately = (enabled: boolean) => {
    setArchiveSlatesSeparatelyState(enabled);
    updatePreferences({ archiveSlatesSeparately: enabled });
  };
  const [showMoreOptions, setShowMoreOptions] = useState(false);
  const [showCustomPattern, setShowCustomPattern] = useState(false);
  const [folderQuery, setFolderQuery] = useState('');
  const folderSearchInputRef = useRef<HTMLInputElement>(null);
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(
    session?.clusters[0]?.anchor.id ?? null,
  );
  const [selectedExportPhotoIds, setSelectedExportPhotoIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [measuredExportSize, setMeasuredExportSize] = useState<MeasuredExportSizePayload | null>(
    null,
  );
  const [isMeasuringSize, setIsMeasuringSize] = useState(false);

  const exportRootFolderName = `${(session?.telemetry.projectName ?? 'Project').replace(/\.[a-z0-9]{2,5}$/i, '')}_Export`;
  const defaultDestinationPath = useMemo(
    () =>
      buildDefaultExportLocation(
        session?.folderPath,
        session?.telemetry.projectName ?? 'Kroma_Project',
        session?.sessionMode ?? 'folder',
      ),
    [session?.folderPath, session?.telemetry.projectName, session?.sessionMode],
  );
  const resolvedDestinationPath = useMemo(() => {
    if (!customExportBaseDir) return defaultDestinationPath;
    return createProjectSubfolder
      ? joinExportPath(customExportBaseDir, exportRootFolderName)
      : customExportBaseDir;
  }, [customExportBaseDir, createProjectSubfolder, exportRootFolderName, defaultDestinationPath]);

  // Reset export destination and selective picks when switching projects
  useEffect(() => {
    setCustomExportBaseDirState(preferences.defaultExportBaseDir);
    setSelectedExportPhotoIds(new Set());
  }, [session?.folderPath, preferences.defaultExportBaseDir]);

  // Ctrl+K focuses Export folder filter while Export workspace is mounted
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        folderSearchInputRef.current?.focus();
        folderSearchInputRef.current?.select();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const settingsPanel = useResizableWidth({
    storageKey: 'export-settings',
    defaultWidth: 400,
    minWidth: 340,
    maxWidth: 490,
  });

  const rendersEdited = deliveryMode === 'develop_and_raw' || deliveryMode === 'develop';

  const exportPlan = useMemo(() => {
    const clusters = session?.clusters ?? [];
    const uniqueExportNames = resolveUniqueExportSubjectNames(clusters, outputRule.pattern);
    const usedFinalPaths = new Set<string>();
    const usedRawCompanionPaths = new Set<string>();
    let totalPhotos = 0;
    let totalOutputFiles = 0;
    let totalExcludedSlates = 0;
    let totalGradedRecipes = 0;

    const folderEntries = clusters.map((cluster) => {
      const exportSubjectName =
        uniqueExportNames.get(cluster.anchor.id) ?? cluster.anchor.subjectName;
      const exportCode = cluster.anchor.subjectCode ?? cluster.anchor.slateCode;
      const slates = cluster.photos.filter((p) => p.isSlateAnchor);
      const clientPhotos = cluster.photos.filter((p) => {
        if (outputRule.excludeSlateFromClientPack && p.isSlateAnchor) return false;
        return true;
      });

      if (outputRule.excludeSlateFromClientPack) {
        totalExcludedSlates += slates.length;
      }
      totalPhotos += clientPhotos.length;
      totalOutputFiles +=
        deliveryMode === 'develop_and_raw' ? clientPhotos.length * 2 : clientPhotos.length;

      let gradedInFolder = 0;
      const seqByFolderKey = new Map<string, number>();
      const virtualFolders = session?.virtualFolders ?? [];
      const mappedFiles = clientPhotos.map((photo) => {
        const graded = isRecipeModified(photo.developRecipe);
        if (graded) {
          totalGradedRecipes += 1;
          gradedInFolder += 1;
        }
        const folderSegments = resolveVirtualFolderPath(virtualFolders, photo.virtualFolderId);
        const folderKey = folderSegments.join('/');
        const nextSeq = (seqByFolderKey.get(folderKey) ?? 0) + 1;
        seqByFolderKey.set(folderKey, nextSeq);

        const rawPath = formatExportPreviewPath(
          outputRule,
          exportSubjectName,
          exportCode,
          nextSeq,
          folderSegments,
          photo.fileName,
        );
        const targetExt = formatExtension(deliveryMode, renderFormat, photo.fileName);
        const baseFinalPath = rawPath.replace(/\.[^./]+$/, targetExt);

        if (deliveryMode === 'develop_and_raw') {
          const dual = splitPreviewIntoEditedAndRaw(baseFinalPath, photo.fileName);
          return {
            photo,
            finalPath: allocateUniqueExportPath(dual.editedPath, usedFinalPaths),
            rawCompanionPath: allocateUniqueExportPath(dual.rawPath, usedRawCompanionPaths),
            graded,
          };
        }

        return {
          photo,
          finalPath: allocateUniqueExportPath(baseFinalPath, usedFinalPaths),
          rawCompanionPath: null as string | null,
          graded,
        };
      });

      const firstPhotoFolderSegments = resolveVirtualFolderPath(
        virtualFolders,
        clientPhotos[0]?.virtualFolderId,
      );
      const sampleBase = formatExportPreviewPath(
        outputRule,
        exportSubjectName,
        exportCode,
        1,
        firstPhotoFolderSegments,
        clientPhotos[0]?.fileName,
      );
      const folderLabel = sampleBase.includes('/')
        ? sampleBase.slice(0, sampleBase.lastIndexOf('/')) || '/'
        : '/';

      return {
        cluster,
        folderLabel,
        mappedFiles,
        gradedInFolder,
        slateCount: slates.length,
      };
    });

    return {
      folderEntries,
      totalPhotos,
      totalOutputFiles,
      totalExcludedSlates,
      totalGradedRecipes,
    };
  }, [session, outputRule, deliveryMode, renderFormat]);

  const activeExportScope = useMemo(() => {
    const validIds = new Set<string>();
    for (const entry of exportPlan.folderEntries) {
      for (const file of entry.mappedFiles) {
        validIds.add(file.photo.id);
      }
    }

    const effectiveSelectedIds = new Set<string>();
    for (const id of selectedExportPhotoIds) {
      if (validIds.has(id)) {
        effectiveSelectedIds.add(id);
      }
    }

    const isSelective = effectiveSelectedIds.size > 0;
    const selectedPhotoIdList = isSelective ? Array.from(effectiveSelectedIds) : undefined;
    const selectedPhotoIdsKey = selectedPhotoIdList ? selectedPhotoIdList.join('\0') : '';

    const activePhotos: PhotoAsset[] = [];
    let activeFolderCount = 0;
    let activeGradedCount = 0;

    for (const entry of exportPlan.folderEntries) {
      let folderHasActive = false;
      for (const file of entry.mappedFiles) {
        if (isSelective && !effectiveSelectedIds.has(file.photo.id)) continue;
        activePhotos.push(file.photo);
        folderHasActive = true;
        if (file.graded) activeGradedCount += 1;
      }
      if (folderHasActive) {
        activeFolderCount += 1;
      }
    }

    const activeOutputFiles =
      deliveryMode === 'develop_and_raw' ? activePhotos.length * 2 : activePhotos.length;

    return {
      effectiveSelectedIds,
      isSelective,
      selectedPhotoIdList,
      selectedPhotoIdsKey,
      activePhotos,
      activePhotoCount: activePhotos.length,
      activeOutputFiles,
      activeFolderCount,
      activeGradedCount,
    };
  }, [exportPlan.folderEntries, selectedExportPhotoIds, deliveryMode]);

  const togglePhotoExportSelection = (photoId: string) => {
    setSelectedExportPhotoIds((prev) => {
      const next = new Set(prev);
      if (next.has(photoId)) {
        next.delete(photoId);
      } else {
        next.add(photoId);
      }
      return next;
    });
  };

  const toggleFolderExportSelection = (folderPhotoIds: readonly string[]) => {
    if (folderPhotoIds.length === 0) return;
    setSelectedExportPhotoIds((prev) => {
      const next = new Set(prev);
      const allSelected = folderPhotoIds.every((id) => next.has(id));
      for (const id of folderPhotoIds) {
        if (allSelected) {
          next.delete(id);
        } else {
          next.add(id);
        }
      }
      return next;
    });
  };

  const clearExportSelection = () => {
    setSelectedExportPhotoIds(new Set());
  };

  const localSizeEstimate = useMemo(() => {
    return estimateExportSizeLocal(
      activeExportScope.activePhotos,
      deliveryMode,
      renderFormat,
      jpegQuality,
    );
  }, [activeExportScope.activePhotos, deliveryMode, renderFormat, jpegQuality]);

  const sessionFolderPath = session?.folderPath ?? null;

  // Measure representative output bytes in Rust using up to 5 evenly-spaced compact previews (or
  // exact on-disk byte sums for Copy/Move) without re-firing on unrelated thumbnail stream updates.
  useEffect(() => {
    if (
      !isDesktopRuntime ||
      !sessionFolderPath ||
      activeExportScope.activePhotoCount === 0
    ) {
      setMeasuredExportSize(null);
      setIsMeasuringSize(false);
      return;
    }

    let cancelled = false;
    setMeasuredExportSize(null);
    setIsMeasuringSize(true);
    const delayMs = deliveryMode === 'copy' || deliveryMode === 'move' ? 0 : 260;
    const mappedFormat = renderFormat === 'tiff16' ? 'tiff' : renderFormat;
    const photoIds = activeExportScope.selectedPhotoIdsKey
      ? activeExportScope.selectedPhotoIdsKey.split('\0')
      : undefined;

    const timer = window.setTimeout(() => {
      void measureExportSizeOnBackend({
        excludeSlatePhoto: outputRule.excludeSlateFromClientPack,
        archiveExcludedSlates: archiveSlatesSeparately,
        mode: deliveryMode,
        format: mappedFormat,
        quality: jpegQuality,
        photoIds,
      }).then((measured) => {
        if (cancelled) return;
        setMeasuredExportSize(measured);
        setIsMeasuringSize(false);
      });
    }, delayMs);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    isDesktopRuntime,
    sessionFolderPath,
    activeExportScope.activePhotoCount,
    activeExportScope.activeGradedCount,
    activeExportScope.selectedPhotoIdsKey,
    outputRule.excludeSlateFromClientPack,
    archiveSlatesSeparately,
    deliveryMode,
    renderFormat,
    jpegQuality,
    measureExportSizeOnBackend,
  ]);

  const sizeEstimate = useMemo(() => {
    if (measuredExportSize) {
      return fromMeasuredExportSize(measuredExportSize, deliveryMode);
    }
    return localSizeEstimate;
  }, [measuredExportSize, deliveryMode, localSizeEstimate]);

  const filteredFolders = useMemo(() => {
    const q = folderQuery.trim().toLowerCase();
    if (!q) return exportPlan.folderEntries;
    return exportPlan.folderEntries.filter(
      (entry) =>
        entry.cluster.anchor.subjectName.toLowerCase().includes(q) ||
        entry.folderLabel.toLowerCase().includes(q) ||
        entry.cluster.anchor.slateCode.toLowerCase().includes(q) ||
        (entry.cluster.anchor.subjectCode?.toLowerCase().includes(q) ?? false),
    );
  }, [exportPlan.folderEntries, folderQuery]);

  const inspectedFolder = useMemo(() => {
    return (
      filteredFolders.find((f) => f.cluster.anchor.id === selectedClusterId) ??
      filteredFolders[0] ??
      exportPlan.folderEntries[0] ??
      null
    );
  }, [filteredFolders, selectedClusterId, exportPlan.folderEntries]);

  if (!session) return null;

  const handleRunExport = async () => {
    const mappedFormat = renderFormat === 'tiff16' ? 'tiff' : renderFormat;

    await runBatchExport({
      outputRoot: session.folderPath || customExportBaseDir ? resolvedDestinationPath : undefined,
      namingPattern: outputRule.pattern,
      excludeSlatePhoto: outputRule.excludeSlateFromClientPack,
      archiveExcludedSlates: archiveSlatesSeparately,
      mode: deliveryMode,
      format: mappedFormat,
      quality: jpegQuality,
      photoIds: activeExportScope.selectedPhotoIdList,
    });
  };

  const isRunning = Boolean(exportJob?.isExporting);
  const activeProgressPct = exportJob ? exportJob.percent : null;
  const activeCompletedCount = exportJob?.completedCount ?? null;
  const activeFailedCount = exportJob?.failedCount ?? 0;
  const activeOutputSummary = exportJob?.outputRoot ?? null;
  const activeError = exportJob?.error ?? null;
  const activeStatusMessage = exportJob?.message ?? 'Exporting files…';
  const sampleFirstFile = exportPlan.folderEntries[0]?.mappedFiles[0];
  const destinationLeafName =
    resolvedDestinationPath.split(/[\\/]/).filter(Boolean).pop() ?? exportRootFolderName;
  const { mode: namingMode, subfolderPerSubject } = inferSimpleExportNaming(outputRule.pattern);
  const moreOptionsSummary = `${namingMode === 'sequential' ? 'Sequential' : 'Original names'}${subfolderPerSubject ? '' : ' • Flat'} • ${outputRule.excludeSlateFromClientPack ? 'Skip slates' : 'Include slates'}`;

  return (
    <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden bg-canvas">
      {/* Left Column: Minimal Export Settings */}
      <aside
        aria-label="Export Settings"
        style={{ width: `${settingsPanel.width}px` }}
        className="shrink-0 bg-surface-low flex flex-col min-h-0 overflow-hidden"
      >
        <div className="h-12 px-5 bg-surface border-b border-hairline flex items-center justify-between gap-2 shrink-0">
          <h1 className="font-display text-[14px] font-bold text-ink-primary truncate">
            Export
          </h1>
          {activeExportScope.isSelective && (
            <button
              type="button"
              onClick={clearExportSelection}
              className="tabular-nums-mono text-[10px] text-amber-soft hover:text-ink-primary cursor-pointer transition-colors"
            >
              {activeExportScope.activePhotoCount.toLocaleString()} selected • Clear
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3.5">
          {/* Prompt if user hasn't run Face Organizer yet (hidden in Single Photo Mode) */}
          {!session.isClustered && session.sessionMode !== 'single' && (
            <div className="p-3 rounded-lg bg-emerald-focus/10 border border-emerald-focus/35 flex items-center justify-between gap-3 shrink-0">
              <div className="text-[11px] text-ink-primary leading-snug">
                Run <strong>Organize by Face</strong> first to group photos into subject folders.
              </div>
              <DarkroomButton
                variant="emerald"
                size="xs"
                icon="face"
                disabled={isScanningFolder}
                onClick={() => void runFaceOrganizer()}
                className="shrink-0"
              >
                Organize
              </DarkroomButton>
            </div>
          )}

          {/* Destination */}
          <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-2 shrink-0">
            <div className="flex items-center justify-between gap-2">
              <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                Destination
              </span>
              {customExportBaseDir && (
                <button
                  type="button"
                  onClick={() => setCustomExportBaseDir(null)}
                  className="tabular-nums-mono text-[9.5px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                >
                  Reset
                </button>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              <div
                title={resolvedDestinationPath}
                className="flex-1 min-w-0 h-8 px-2.5 rounded-md bg-canvas border border-hairline flex items-center gap-1.5"
              >
                <span className="material-symbols-outlined text-[14px] text-amber-soft shrink-0">
                  folder
                </span>
                <span className="tabular-nums-mono text-[11px] text-ink-primary truncate">
                  {resolvedDestinationPath}
                </span>
              </div>
              <DarkroomButton
                variant="secondary"
                size="xs"
                icon="folder_open"
                onClick={async () => {
                  const chosen = await pickExportDirectory();
                  if (chosen) setCustomExportBaseDir(chosen);
                }}
                className="h-8 px-2.5 shrink-0"
              >
                Choose…
              </DarkroomButton>
            </div>

            {customExportBaseDir && (
              <DarkroomToggle
                checked={createProjectSubfolder}
                onChange={setCreateProjectSubfolder}
                label={`Create "${exportRootFolderName}/" subfolder`}
              />
            )}
          </section>

          {/* Files & Format */}
          <section className="p-3 rounded-lg bg-surface border border-hairline flex flex-col gap-3 shrink-0">
            <div className="flex flex-col gap-1.5">
              <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                Files
              </span>
              <SegmentedControl
                layoutId="export-delivery-mode"
                options={DELIVERY_MODES}
                value={deliveryMode}
                onChange={setDeliveryMode}
                className="w-full justify-between"
              />
              {deliveryMode === 'move' && (
                <div className="px-2.5 py-1.5 rounded-md bg-amber-core/10 border border-amber-core/35 text-[10.5px] text-amber-soft leading-snug">
                  Moves original files out of your source folder (cannot be undone).
                </div>
              )}
            </div>

            <AnimatePresence initial={false}>
              {rendersEdited && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={FAST_FADE}
                  className="overflow-hidden"
                >
                  <div className="pt-2.5 border-t border-hairline flex flex-col gap-2.5">
                    <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                      Format
                    </span>
                    <SegmentedControl
                      layoutId="export-render-format"
                      options={RENDER_FORMATS}
                      value={renderFormat}
                      onChange={setRenderFormat}
                      className="w-full justify-between"
                    />
                    {renderFormat === 'jpeg' && (
                      <DarkroomScrubber
                        label="Quality"
                        value={jpegQuality}
                        min={70}
                        max={100}
                        defaultValue={94}
                        formatValue={(v) => `${v}%`}
                        onChange={setJpegQuality}
                      />
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>

          {/* Collapsible More Options (Naming & Name-Card Handling) */}
          <section className="rounded-lg bg-surface border border-hairline overflow-hidden shrink-0">
            <button
              type="button"
              onClick={() => setShowMoreOptions((prev) => !prev)}
              aria-expanded={showMoreOptions}
              className="w-full px-3 py-2.5 flex items-center justify-between gap-2 text-left hover:bg-surface-elevated/60 transition-colors cursor-pointer"
            >
              <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                More Options
              </span>
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="tabular-nums-mono text-[10px] text-ink-telemetry truncate">
                  {moreOptionsSummary}
                </span>
                <span
                  className={`material-symbols-outlined text-[15px] text-ink-telemetry transition-transform ${
                    showMoreOptions ? 'rotate-180' : ''
                  }`}
                >
                  expand_more
                </span>
              </div>
            </button>

            <AnimatePresence initial={false}>
              {showMoreOptions && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={FAST_FADE}
                  className="overflow-hidden"
                >
                  <div className="px-3 pb-3 pt-2 border-t border-hairline flex flex-col gap-3">
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center justify-between">
                        <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                          File Naming
                        </span>
                        <button
                          type="button"
                          onClick={() => setShowCustomPattern((prev) => !prev)}
                          className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                        >
                          {showCustomPattern ? 'Hide pattern' : 'Custom pattern'}
                        </button>
                      </div>

                      <SegmentedControl
                        layoutId="export-file-naming-mode"
                        options={FILE_NAMING_MODES}
                        value={namingMode}
                        onChange={(nextMode) =>
                          setOutputPattern(buildSimpleExportPattern(nextMode, subfolderPerSubject))
                        }
                        className="w-full justify-between"
                      />

                      {showCustomPattern && (
                        <div className="flex flex-col gap-1.5 pt-1">
                          <input
                            type="text"
                            value={outputRule.pattern}
                            onChange={(e) => setOutputPattern(e.target.value)}
                            aria-label="Custom naming pattern"
                            className="w-full px-2.5 py-1.5 rounded-md bg-canvas border border-hairline focus:border-amber-core tabular-nums-mono text-[11px] text-amber-soft outline-none"
                          />
                          <div className="flex items-center flex-wrap gap-1.5">
                            {NAMING_TOKENS.map(({ token, label }) => (
                              <button
                                key={token}
                                type="button"
                                onClick={() => setOutputPattern(`${outputRule.pattern}${token}`)}
                                className="px-2 py-0.5 rounded bg-canvas border border-hairline hover:border-amber-core/60 text-[10px] text-ink-secondary hover:text-amber-soft cursor-pointer transition-colors"
                              >
                                + {label}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="pt-2.5 border-t border-hairline flex flex-col gap-2">
                      <DarkroomToggle
                        checked={subfolderPerSubject}
                        onChange={(checked) =>
                          setOutputPattern(buildSimpleExportPattern(namingMode, checked))
                        }
                        label="Subfolder per person"
                      />
                      <DarkroomToggle
                        checked={outputRule.excludeSlateFromClientPack}
                        onChange={setExcludeSlateFromClientPack}
                        label="Skip name-card photos"
                      />
                      {outputRule.excludeSlateFromClientPack && (
                        <DarkroomToggle
                          checked={archiveSlatesSeparately}
                          onChange={setArchiveSlatesSeparately}
                          label="Archive name-cards in /_Slates_Archive/"
                        />
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </div>

        {/* Pinned Export Summary, Size Estimate & CTA Footer */}
        <div className="p-4 bg-surface border-t border-hairline flex flex-col gap-2.5 shrink-0">
          <div className="px-2.5 py-2 rounded-lg bg-canvas border border-hairline flex flex-col gap-0.5 tabular-nums-mono text-[10.5px]">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 text-ink-secondary">
                <span className="material-symbols-outlined text-[14px] text-ink-telemetry">
                  hard_drive
                </span>
                <span>
                  {activeExportScope.isSelective ? 'Selected Size' : 'Estimated Size'}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                {isMeasuringSize && (
                  <span className="text-[9.5px] text-amber-soft animate-pulse">Measuring…</span>
                )}
                <span className="font-semibold text-ink-primary">
                  {sizeEstimate.formattedExtraSize}
                </span>
              </div>
            </div>
            {(sizeEstimate.isMoveRelocation || deliveryMode === 'develop_and_raw') && (
              <div className="text-[9.5px] text-ink-telemetry leading-snug truncate">
                {sizeEstimate.isMoveRelocation
                  ? `Relocates ${sizeEstimate.formattedTotalSourceSize} • 0 MB extra space`
                  : `${sizeEstimate.formattedEditedSize} edited + ${sizeEstimate.formattedOriginalsSize} originals`}
              </div>
            )}
          </div>

          {activeError && (
            <div className="px-2.5 py-2 rounded-lg bg-crimson-danger/10 border border-crimson-danger/40 tabular-nums-mono text-[10px] text-crimson-danger leading-snug">
              Export failed: {activeError}
            </div>
          )}

          {activeProgressPct !== null && !activeError && (
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between gap-2 tabular-nums-mono text-[10px]">
                <span
                  title={
                    activeProgressPct === 100 && activeOutputSummary
                      ? activeOutputSummary
                      : activeStatusMessage
                  }
                  className={`truncate ${activeProgressPct === 100 ? 'text-emerald-bright' : 'text-amber-soft'
                    }`}
                >
                  {activeProgressPct === 100
                    ? (() => {
                      const completed = activeCompletedCount ?? activeExportScope.activeOutputFiles;
                      return `Done — ${completed.toLocaleString()} ${completed === 1 ? 'file' : 'files'}${activeFailedCount > 0 ? ` (${activeFailedCount} failed)` : ''}${activeOutputSummary ? ` → ${activeOutputSummary}` : ''}`;
                    })()
                    : activeStatusMessage}
                </span>
                <span className="text-ink-primary font-semibold shrink-0">
                  {activeProgressPct}%
                </span>
              </div>
              <div className="h-2 rounded-full bg-canvas border border-hairline overflow-hidden">
                <motion.div
                  initial={false}
                  animate={{ width: `${activeProgressPct}%` }}
                  transition={SOFT_SPRING}
                  className={
                    activeProgressPct === 100 ? 'h-full bg-emerald-focus' : 'h-full bg-amber-core'
                  }
                />
              </div>
            </div>
          )}

          <DarkroomButton
            variant="primary"
            size="md"
            icon={isRunning ? 'autorenew' : 'folder_zip'}
            disabled={isRunning || activeExportScope.activePhotoCount === 0}
            onClick={() => void handleRunExport()}
            className="w-full h-10 text-[13px] justify-center"
          >
            <span>
              {isRunning
                ? exportJob && exportJob.total > 0
                  ? `Exporting ${exportJob.completed.toLocaleString()}/${exportJob.total.toLocaleString()} (${exportJob.percent}%)`
                  : 'Exporting…'
                : (() => {
                  const files = activeExportScope.activeOutputFiles;
                  const folders = activeExportScope.activeFolderCount;
                  const fileWord = files === 1 ? 'File' : 'Files';
                  const folderWord = folders === 1 ? 'Folder' : 'Folders';
                  if (activeExportScope.isSelective) {
                    return files === 1
                      ? 'Export 1 Selected File'
                      : `Export ${files.toLocaleString()} Selected ${fileWord} (${folders.toLocaleString()} ${folderWord})`;
                  }
                  return `Export ${files.toLocaleString()} ${fileWord} (${folders.toLocaleString()} ${folderWord})`;
                })()}
            </span>
          </DarkroomButton>
        </div>
      </aside>

      <PanelResizeHandle
        panelSide="left"
        width={settingsPanel.width}
        minWidth={340}
        maxWidth={490}
        onChangeWidth={settingsPanel.setWidth}
        onResetWidth={settingsPanel.resetWidth}
        label="Resize Export settings sidebar"
      />

      {/* Right Column: Instant Visual Output Preview (Virtualized for 500+ Folders) */}
      <section
        aria-label="Output Folder Preview"
        className="flex-1 min-w-0 flex flex-col overflow-hidden bg-canvas"
      >
        {/* Preview Header Toolbar */}
        <div className="h-12 px-5 bg-surface border-b border-hairline flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="material-symbols-outlined text-[18px] text-amber-core shrink-0">
              account_tree
            </span>
            <h2 className="font-display text-[14px] font-bold text-ink-primary shrink-0">
              Output Preview
            </h2>
            <span
              title={resolvedDestinationPath}
              className="tabular-nums-mono text-[11px] text-ink-telemetry truncate"
            >
              {resolvedDestinationPath}
            </span>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <TelemetryPill tone="neutral">{sizeEstimate.formattedExtraSize}</TelemetryPill>
            <TelemetryPill tone={activeExportScope.isSelective ? 'amber' : 'emerald'}>
              {deliveryMode === 'develop_and_raw'
                ? `${activeExportScope.activePhotoCount.toLocaleString()} Edited + ${activeExportScope.activePhotoCount.toLocaleString()} RAW${activeExportScope.isSelective ? ' Selected' : ''}`
                : `${activeExportScope.activeOutputFiles.toLocaleString()} ${activeExportScope.isSelective ? 'Selected ' : ''}${activeExportScope.activeOutputFiles === 1 ? 'File' : 'Files'}`}
            </TelemetryPill>
            {ambiguousReviewQueue.length > 0 && (
              <button
                type="button"
                onClick={() => setActiveWorkspace('review_queue')}
                title="Review borderline matches before exporting"
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-tangerine-alert/15 hover:bg-tangerine-alert/25 border border-tangerine-alert/40 text-tangerine-alert tabular-nums-mono text-[10px] font-medium transition-colors cursor-pointer"
              >
                <span>{ambiguousReviewQueue.length} Needs Review</span>
                <span className="material-symbols-outlined text-[12px]">arrow_forward</span>
              </button>
            )}
          </div>
        </div>

        {/* Visual Blueprint Strip — Shows First-Time Users Exactly What Gets Created */}
        {(() => {
          const previewFile = inspectedFolder?.mappedFiles[0] ?? sampleFirstFile;
          if (!previewFile) return null;
          return (
            <div className="px-5 py-3 bg-surface-low/80 border-b border-hairline flex flex-wrap items-center justify-between gap-3 shrink-0">
              <div className="flex flex-col gap-1 min-w-0">
                <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                  Example Output Path
                </span>
                <div className="flex flex-wrap items-center gap-1.5 tabular-nums-mono text-[11px]">
                  <span className="text-ink-secondary">{resolvedDestinationPath}</span>
                  <span className="text-amber-soft font-semibold truncate">
                    {previewFile.finalPath}
                  </span>
                  {previewFile.rawCompanionPath && (
                    <>
                      <span className="text-ink-telemetry">+</span>
                      <span className="text-ink-secondary truncate">
                        {previewFile.rawCompanionPath}
                      </span>
                    </>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-2 w-[210px] h-7 px-2.5 rounded-md bg-canvas border border-hairline focus-within:border-amber-core shrink-0">
                <span className="material-symbols-outlined text-[14px] text-ink-telemetry shrink-0">
                  search
                </span>
                <input
                  ref={folderSearchInputRef}
                  type="text"
                  value={folderQuery}
                  onChange={(e) => setFolderQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape' && folderQuery) {
                      e.stopPropagation();
                      setFolderQuery('');
                    }
                  }}
                  placeholder={`Filter ${exportPlan.folderEntries.length} folders...`}
                  className="bg-transparent text-[11px] text-ink-primary placeholder:text-ink-telemetry flex-1 min-w-0 outline-none"
                />
                {folderQuery && (
                  <button
                    type="button"
                    onClick={() => setFolderQuery('')}
                    className="text-ink-telemetry hover:text-ink-primary cursor-pointer flex items-center"
                  >
                    <span className="material-symbols-outlined text-[13px]">close</span>
                  </button>
                )}
              </div>
            </div>
          );
        })()}

        {/* Master-Detail Folder Preview: Virtualized Folders (Left) + Live Photo Files (Right) */}
        <div className="flex-1 min-h-0 flex overflow-hidden">
          {/* Virtualized Folder List (renders 547+ folders effortlessly with zero squishing!) */}
          <div className="flex-1 min-w-0 flex flex-col p-4 overflow-hidden">
            <div className="pb-2 flex items-center justify-between gap-2 tabular-nums-mono text-[10px] text-ink-telemetry shrink-0">
              <span>
                FOLDERS ({filteredFolders.length.toLocaleString()}
                {folderQuery ? ` of ${exportPlan.folderEntries.length.toLocaleString()}` : ''})
              </span>
              {activeExportScope.isSelective ? (
                <div className="flex items-center gap-2">
                  <span className="text-amber-soft font-semibold">
                    {activeExportScope.activePhotoCount.toLocaleString()}{' '}
                    {activeExportScope.activePhotoCount === 1 ? 'file' : 'files'} selected
                  </span>
                  <button
                    type="button"
                    onClick={clearExportSelection}
                    className="text-ink-secondary hover:text-ink-primary underline underline-offset-2 cursor-pointer"
                  >
                    Clear
                  </button>
                </div>
              ) : (
                <span>Check folders or files to export a subset</span>
              )}
            </div>

            {filteredFolders.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center gap-1.5 text-ink-telemetry">
                <span className="material-symbols-outlined text-[24px]">folder_off</span>
                <span className="text-[12px] text-ink-secondary">
                  No folders matching &ldquo;{folderQuery}&rdquo;
                </span>
              </div>
            ) : (
              <VirtualVerticalList
                items={filteredFolders}
                rowHeightPx={60}
                gapPx={6}
                overscan={6}
                getKey={(entry) => entry.cluster.anchor.id}
                className="flex-1 min-h-0 pr-1"
                renderItem={(entry) => {
                  const isSelected =
                    inspectedFolder?.cluster.anchor.id === entry.cluster.anchor.id;
                  const faceOnly = isFaceOnlySlateCode(entry.cluster.anchor.slateCode);
                  const folderPhotoIds = entry.mappedFiles.map((f) => f.photo.id);
                  const selectedInFolder = folderPhotoIds.reduce(
                    (acc, id) => acc + (activeExportScope.effectiveSelectedIds.has(id) ? 1 : 0),
                    0,
                  );
                  const folderCheckState: 'none' | 'partial' | 'all' =
                    selectedInFolder === 0
                      ? 'none'
                      : selectedInFolder === folderPhotoIds.length
                        ? 'all'
                        : 'partial';
                  const fileCount =
                    deliveryMode === 'develop_and_raw'
                      ? entry.mappedFiles.length * 2
                      : entry.mappedFiles.length;

                  return (
                    <div
                      role="button"
                      tabIndex={0}
                      onClick={() => setSelectedClusterId(entry.cluster.anchor.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          setSelectedClusterId(entry.cluster.anchor.id);
                        }
                      }}
                      className={`w-full h-full px-3 rounded-lg border flex items-center justify-between gap-3 text-left transition-colors cursor-pointer ${isSelected
                          ? 'bg-surface-active border-amber-core/60 text-ink-primary'
                          : folderCheckState !== 'none'
                            ? 'bg-surface-elevated/70 border-amber-core/35 hover:border-amber-core/55'
                            : 'bg-surface border-hairline hover:bg-surface-elevated hover:border-hairline-strong'
                        }`}
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        <button
                          type="button"
                          role="checkbox"
                          aria-checked={
                            folderCheckState === 'all'
                              ? true
                              : folderCheckState === 'partial'
                                ? 'mixed'
                                : false
                          }
                          aria-label={`Include ${entry.cluster.anchor.subjectName} in export`}
                          disabled={folderPhotoIds.length === 0}
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleFolderExportSelection(folderPhotoIds);
                          }}
                          className={`w-4 h-4 rounded flex items-center justify-center border shrink-0 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed ${folderCheckState === 'all'
                              ? 'bg-amber-core border-amber-core text-canvas'
                              : folderCheckState === 'partial'
                                ? 'bg-amber-core/20 border-amber-core text-amber-soft'
                                : 'bg-canvas/70 border-hairline-strong text-transparent hover:border-amber-core/70'
                            }`}
                        >
                          <span className="material-symbols-outlined text-[12px] leading-none font-bold">
                            {folderCheckState === 'partial' ? 'remove' : 'check'}
                          </span>
                        </button>

                        <div className="w-9 h-10 rounded overflow-hidden bg-surface-lowest border border-hairline shrink-0">
                          {entry.cluster.anchor.thumbnailUrl ? (
                            <img
                              src={entry.cluster.anchor.thumbnailUrl}
                              alt={entry.cluster.anchor.subjectName}
                              loading="lazy"
                              decoding="async"
                              className="w-full h-full object-cover object-top"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-ink-telemetry">
                              <span className="material-symbols-outlined text-[16px]">folder</span>
                            </div>
                          )}
                        </div>

                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className="material-symbols-outlined text-[15px] text-amber-soft shrink-0">
                              {isSelected ? 'folder_open' : 'folder'}
                            </span>
                            <span className="tabular-nums-mono text-[12px] font-semibold text-ink-primary truncate">
                              {entry.folderLabel === '/' ? destinationLeafName : entry.folderLabel}/
                            </span>
                          </div>
                          <div className="tabular-nums-mono text-[10px] text-ink-telemetry truncate">
                            {entry.cluster.anchor.subjectName} •{' '}
                            {!session.isClustered
                              ? 'All Photos'
                              : faceOnly
                                ? 'Grouped by Face'
                                : `Slate ${entry.cluster.anchor.slateCode}`}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-1.5 shrink-0">
                        {entry.gradedInFolder > 0 && rendersEdited && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-core/15 border border-amber-core/30 tabular-nums-mono text-[9.5px] text-amber-soft">
                            {entry.gradedInFolder} edited
                          </span>
                        )}
                        {folderCheckState === 'partial' ? (
                          <TelemetryPill tone="amber">
                            {selectedInFolder}/{entry.mappedFiles.length}
                          </TelemetryPill>
                        ) : (
                          <TelemetryPill tone={folderCheckState === 'all' ? 'amber' : 'emerald'}>
                            {fileCount} {fileCount === 1 ? 'file' : 'files'}
                          </TelemetryPill>
                        )}
                        {outputRule.excludeSlateFromClientPack && entry.slateCount > 0 && (
                          <span className="tabular-nums-mono text-[9.5px] text-ink-telemetry">
                            -{entry.slateCount} ID
                          </span>
                        )}
                      </div>
                    </div>
                  );
                }}
              />
            )}
          </div>

          {/* Right Inspector: Selected Folder Visual Contents */}
          {inspectedFolder && (() => {
            const inspectedPhotoIds = inspectedFolder.mappedFiles.map((f) => f.photo.id);
            const inspectedAllSelected =
              inspectedPhotoIds.length > 0 &&
              inspectedPhotoIds.every((id) => activeExportScope.effectiveSelectedIds.has(id));

            return (
              <aside
                aria-label="Selected Folder Contents"
                className="w-[330px] xl:w-[370px] shrink-0 bg-surface-low border-l border-hairline flex flex-col min-h-0 overflow-hidden"
              >
                <div className="px-4 py-3 bg-surface border-b border-hairline shrink-0">
                  <div className="flex items-center justify-between gap-2 min-w-0">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="material-symbols-outlined text-[16px] text-amber-soft shrink-0">
                        folder_open
                      </span>
                      <span className="font-display text-[13px] font-bold text-ink-primary truncate">
                        {inspectedFolder.cluster.anchor.subjectName}
                      </span>
                    </div>
                    {inspectedPhotoIds.length > 0 && (
                      <button
                        type="button"
                        onClick={() => toggleFolderExportSelection(inspectedPhotoIds)}
                        className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-amber-soft shrink-0 cursor-pointer transition-colors"
                      >
                        {inspectedAllSelected ? 'Unselect folder' : 'Select folder'}
                      </button>
                    )}
                  </div>
                  <div className="tabular-nums-mono text-[10px] text-ink-telemetry truncate mt-0.5">
                    {destinationLeafName}
                    {inspectedFolder.folderLabel === '/' ? '' : inspectedFolder.folderLabel}/
                    {deliveryMode === 'develop_and_raw' ? ' {Edited, RAW}' : ''}
                  </div>
                </div>

                <div className="flex-1 min-h-0 flex flex-col p-3 overflow-hidden">
                  {inspectedFolder.mappedFiles.length === 0 ? (
                    <div className="py-10 text-center text-[11.5px] text-ink-telemetry">
                      No client photos in this folder (only excluded name-card frames).
                    </div>
                  ) : (
                    <VirtualVerticalList
                      key={inspectedFolder.cluster.anchor.id}
                      items={inspectedFolder.mappedFiles}
                      rowHeightPx={deliveryMode === 'develop_and_raw' ? 76 : 68}
                      gapPx={8}
                      overscan={6}
                      getKey={(item) => item.photo.id}
                      className="flex-1 min-h-0 pr-1"
                      renderItem={({ photo, finalPath, rawCompanionPath, graded }) => {
                        const shortOut = finalPath.split('/').slice(-2).join('/');
                        const shortRaw = rawCompanionPath
                          ? rawCompanionPath.split('/').slice(-2).join('/')
                          : null;
                        const isPhotoChecked = activeExportScope.effectiveSelectedIds.has(photo.id);

                        return (
                          <button
                            type="button"
                            role="checkbox"
                            aria-checked={isPhotoChecked}
                            aria-label={`Select ${photo.fileName} for export`}
                            onClick={() => togglePhotoExportSelection(photo.id)}
                            className={`w-full h-full text-left p-2 rounded-lg border flex items-center gap-2.5 transition-colors cursor-pointer ${
                              isPhotoChecked
                                ? 'bg-surface-active/60 border-amber-core/50'
                                : 'bg-surface border-hairline hover:bg-surface-elevated hover:border-hairline-strong'
                            }`}
                          >
                            <span
                              aria-hidden="true"
                              className={`w-4 h-4 rounded flex items-center justify-center border shrink-0 transition-colors ${
                                isPhotoChecked
                                  ? 'bg-amber-core border-amber-core text-canvas'
                                  : 'bg-canvas/70 border-hairline-strong text-transparent'
                              }`}
                            >
                              <span className="material-symbols-outlined text-[12px] leading-none font-bold">
                                check
                              </span>
                            </span>

                            <div className="relative w-11 h-13 rounded overflow-hidden bg-surface-lowest border border-hairline shrink-0">
                              {rendersEdited ? (
                                <GradedThumbnailImg
                                  src={photo.imageUrl}
                                  renderedSrc={photo.renderedThumbnailUrl}
                                  alt={photo.fileName}
                                  recipe={photo.developRecipe}
                                  loading="lazy"
                                  decoding="async"
                                  className="w-full h-full object-cover object-top"
                                />
                              ) : (
                                <img
                                  src={photo.imageUrl}
                                  alt={photo.fileName}
                                  loading="lazy"
                                  decoding="async"
                                  className="w-full h-full object-cover object-top"
                                />
                              )}
                            </div>

                            <div className="flex-1 min-w-0 flex flex-col gap-0.5 tabular-nums-mono">
                              <div className="flex items-center justify-between gap-1.5">
                                <span className="text-[11px] font-semibold text-ink-primary truncate">
                                  {shortOut}
                                </span>
                                {graded && rendersEdited && (
                                  <span className="shrink-0 px-1 py-0.5 rounded bg-amber-core/15 text-amber-soft text-[8.5px] uppercase font-semibold">
                                    Edited
                                  </span>
                                )}
                              </div>

                              <span className="text-[9.5px] text-ink-telemetry truncate">
                                From {photo.fileName}
                              </span>

                              {shortRaw && (
                                <span className="text-[9.5px] text-emerald-bright/90 truncate">
                                  + {shortRaw} (Original)
                                </span>
                              )}
                            </div>
                          </button>
                        );
                      }}
                    />
                  )}
                </div>
              </aside>
            );
          })()}
        </div>
      </section>
    </div>
  );
}
