import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStudio, type DevelopSourceMode } from '../../app/StudioContext';
import {
  filterDevelopFolders,
} from '../../application/useCases/filterDevelopFolders';
import {
  filterSubjectClusters,
  type SubjectFilterMode,
} from '../../application/useCases/filterSubjectClusters';
import {
  calibrateWhiteBalanceFromSampledRgb,
  isRecipeModified,
  type DevelopRecipe,
} from '../../domain/develop';
import { rgbToHsl } from '../../domain/developShaderMath';
import {
  isRawFileName,
  shouldProbeHighBitDepthBuffer,
} from '../../domain/developWorkingBuffer';
import {
  comparePhotosAlphabetically,
  compareSubjectClustersAlphabetically,
  isFaceOnlySlateCode,
} from '../../domain/identity';
import type { PhotoAsset } from '../../domain/photo';
import {
  aiMaskKindLabel,
  createAiMatteComponent,
  createBrushComponent,
  createDefaultBackgroundRemoval,
  createMaskLayer,
  isBackgroundRemovalActive,
  maskComponentLabel,
  type AiMaskKind,
  type BackgroundRemovalSettings,
  type BrushStroke,
  type MaskCombineOp,
  type MaskComponent,
} from '../../domain/masking';
import {
  hydrateRetouchSettings,
  MAX_INPAINT_PATCHES,
  type HealSpot,
  type InpaintStrokeInput,
} from '../../domain/retouch';
import {
  DarkroomButton,
  PanelResizeHandle,
  SegmentedControl,
  TelemetryPill,
  useResizableWidth,
  VirtualHorizontalStrip,
  VirtualVerticalList,
  type SegmentedOption,
} from '../darkroom-kit';
import {
  SubjectHoverPreviewCard,
  useSubjectHoverPreview,
} from '../slates/SubjectHoverPreviewCard';
import {
  BeforeAfterCanvas,
  DEVELOP_WORKING_MAX_DIM,
  type CompareViewportMode,
} from './BeforeAfterCanvas';
import { formatViewportZoomLabel } from '../../domain/viewportZoom';
import { DevelopInspectorPanel, type DevelopToolMode } from './DevelopInspectorPanel';
import { preloadDevelopBitmap } from './developBitmapCache';
import { FloatingMasksLayersPanel } from './FloatingMasksLayersPanel';
import { GradedThumbnailImg } from './GradedThumbnailImg';
import { useLiveDevelopRecipe } from './liveDevelopDraftStore';
import type { BrushToolOptions } from './MaskCanvasOverlay';
import { createManualMaskComponent } from './MaskingPanel';
import {
  PhotoNameHoverPreview,
  usePhotoNameHoverPreview,
} from './PhotoNameHoverPreview';
import type { HealToolOptions } from './RetouchPanel';
import { SyncSettingsModal } from './SyncSettingsModal';

interface BatchDevelopStudioProps {
  readonly isActive?: boolean;
}

const SOURCE_MODE_OPTIONS: readonly SegmentedOption<DevelopSourceMode>[] = [
  { id: 'folders', label: 'Folders', title: 'Edit by folder' },
  { id: 'subjects', label: 'Subjects', title: 'Edit by subject' },
];

const COMPARE_OPTIONS: readonly SegmentedOption<CompareViewportMode>[] = [
  { id: 'single', label: 'Single', title: 'Single loupe view' },
  { id: 'split', label: 'Split', title: 'Split before/after wipe' },
  { id: 'before_after', label: 'Y | Y', title: 'Side-by-side Before / After' },
];

export const BatchDevelopStudio = memo(function BatchDevelopStudio({
  isActive = true,
}: BatchDevelopStudioProps) {
  const {
    isDesktopRuntime,
    runtimeInfo,
    modelDownloadProgress,
    downloadingOptionalModelId,
    faceParseMapsByPhotoId,
    preferences,
    session,
    activeCluster,
    developSourceMode,
    activeDevelopFolderId,
    developPhotos,
    activeDevelopScopeLabel,
    activeDevelopPhoto: committedActiveDevelopPhoto,
    liveHistogram,
    selectedPhotoIds,
    copiedRecipe,
    canUndoDevelop,
    canRedoDevelop,
    undoDevelopLabel,
    redoDevelopLabel,
    setDevelopSourceMode,
    selectDevelopFolder,
    selectCluster,
    setActiveDevelopPhotoId,
    togglePhotoSelection,
    peekDevelopWorkingBuffer,
    loadDevelopWorkingBuffer,
    generateAiMask,
    detectPhotoFaceOverlay,
    parsePhotoFaceRetouchMap,
    inpaintPhotoRegion,
    downloadOptionalModel,
    cancelOptionalModelDownload,
    updatePhotoRecipe,
    syncRecipeFromPhoto,
    applyRelativeNudgeToClusterSelection,
    copyActiveRecipe,
    pasteRecipeToClusterSelection,
    syncSlateWhiteBalanceForActiveCluster,
    undoDevelopAction,
    redoDevelopAction,
    studioFocusMode,
    setStudioFocusMode,
    toggleStudioFocusMode,
    openShootFolder,
    setActiveWorkspace,
    notifyAction,
  } = useStudio();

  const liveActiveRecipe = useLiveDevelopRecipe(
    committedActiveDevelopPhoto?.id,
    committedActiveDevelopPhoto?.developRecipe,
  );
  const activeDevelopPhoto = useMemo(
    () =>
      committedActiveDevelopPhoto &&
      liveActiveRecipe &&
      liveActiveRecipe !== committedActiveDevelopPhoto.developRecipe
        ? { ...committedActiveDevelopPhoto, developRecipe: liveActiveRecipe }
        : committedActiveDevelopPhoto,
    [committedActiveDevelopPhoto, liveActiveRecipe],
  );

  const [showFocusHint, setShowFocusHint] = useState(false);

  useEffect(() => {
    if (!studioFocusMode) {
      setShowFocusHint(false);
      return;
    }
    setShowFocusHint(true);
    const timer = window.setTimeout(() => setShowFocusHint(false), 2200);
    return () => window.clearTimeout(timer);
  }, [studioFocusMode]);

  const [compareMode, setCompareMode] = useState<CompareViewportMode>('single');
  const [zoomFactor, setZoomFactor] = useState<number>(1);
  const [oneToOneZoom, setOneToOneZoom] = useState<number>(2);
  const [showShadowClip, setShowShadowClip] = useState(false);
  const [showHighlightClip, setShowHighlightClip] = useState(false);
  const [eyedropperActive, setEyedropperActive] = useState(false);
  const [activePhotoAspect, setActivePhotoAspect] = useState(2 / 3);
  const [toolMode, setToolMode] = useState<DevelopToolMode>('edit');
  const [activeMaskId, setActiveMaskId] = useState<string | null>(null);
  const [activeMaskComponentId, setActiveMaskComponentId] = useState<string | null>(null);
  const [creatingNewMask, setCreatingNewMask] = useState(false);
  const [showMaskOverlay, setShowMaskOverlay] = useState(true);
  const [colorPickActive, setColorPickActive] = useState(false);
  const [busyAiMaskKind, setBusyAiMaskKind] = useState<AiMaskKind | null>(null);
  const [busyBgRemoval, setBusyBgRemoval] = useState(false);
  const [brushOptions, setBrushOptions] = useState<BrushToolOptions>({
    size: 0.065,
    feather: 0.55,
    flow: 1,
    erase: false,
  });
  const [healTool, setHealTool] = useState<HealToolOptions>({
    mode: 'heal',
    radius: 0.018,
    feather: 0.55,
    opacity: 1,
  });
  const [selectedSpotId, setSelectedSpotId] = useState<string | null>(null);
  const [showSpotPins, setShowSpotPins] = useState(true);
  const [detectingFace, setDetectingFace] = useState(false);
  const [parsingFace, setParsingFace] = useState(false);
  const [pendingRemoveStrokes, setPendingRemoveStrokes] = useState<readonly InpaintStrokeInput[]>([]);
  const [busyInpainting, setBusyInpainting] = useState(false);
  const autoFaceScanAttemptedRef = useRef<Set<string>>(new Set());
  const autoFaceParseAttemptedRef = useRef<Set<string>>(new Set());
  const activeStrokeTargetRef = useRef<{ layerId: string; compId: string; strokeIndex: number } | null>(
    null,
  );
  const [syncModalOpen, setSyncModalOpen] = useState(false);
  const [folderQuery, setFolderQuery] = useState('');
  const folderSearchInputRef = useRef<HTMLInputElement>(null);
  const [expandedFolderIds, setExpandedFolderIds] = useState<ReadonlySet<string>>(() => new Set());
  const [collapsedSearchFolderIds, setCollapsedSearchFolderIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [expandedFullFolderIds, setExpandedFullFolderIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [subjectQuery, setSubjectQuery] = useState('');
  const [subjectFilter, setSubjectFilter] = useState<SubjectFilterMode>('all');
  const subjectSearchInputRef = useRef<HTMLInputElement>(null);
  const leftPanel = useResizableWidth({
    storageKey: 'studio-left',
    defaultWidth: 264,
    minWidth: 204,
    maxWidth: 420,
  });
  const rightPanel = useResizableWidth({
    storageKey: 'studio-right',
    defaultWidth: 336,
    minWidth: 300,
    maxWidth: 480,
  });
  const {
    preview: hoveredSubject,
    schedulePreview: scheduleSubjectHover,
    clearPreview: clearSubjectHover,
  } = useSubjectHoverPreview();
  const {
    preview: hoveredFolderPhoto,
    schedulePreview: scheduleFolderPhotoHover,
    clearPreview: clearFolderPhotoHover,
  } = usePhotoNameHoverPreview();

  const toggleFolderExpanded = useCallback((folderId: string, isAutoExpanded: boolean) => {
    if (isAutoExpanded) {
      setCollapsedSearchFolderIds((prev) => {
        const next = new Set(prev);
        if (next.has(folderId)) next.delete(folderId);
        else next.add(folderId);
        return next;
      });
      return;
    }
    setExpandedFolderIds((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) {
        next.delete(folderId);
      } else {
        next.add(folderId);
      }
      return next;
    });
  }, []);

  const cropModeActive = toolMode === 'crop';
  const retouchActive = toolMode === 'retouch';
  const maskingActive = toolMode === 'mask';

  // Single O(N) memoized pass for total, per-directory photo lists/counts, and per-cluster edit counters
  const {
    totalPhotoCount,
    totalEditedCount,
    dirPhotosById,
    dirEditedCountById,
    clusterEditedCountById,
  } = useMemo(() => {
    const dirPhotos = new Map<string, PhotoAsset[]>();
    const dirEdited = new Map<string, number>();
    const clusterEdited = new Map<string, number>();
    let total = 0;
    let editedTotal = 0;

    if (session) {
      for (const cluster of session.clusters) {
        let clusterEdits = 0;
        for (const photo of cluster.photos) {
          total += 1;
          const isEdited = isRecipeModified(photo.developRecipe);
          if (isEdited) {
            editedTotal += 1;
            clusterEdits += 1;
            dirEdited.set(photo.directoryId, (dirEdited.get(photo.directoryId) ?? 0) + 1);
          }
          const bucket = dirPhotos.get(photo.directoryId);
          if (bucket) {
            bucket.push(photo);
          } else {
            dirPhotos.set(photo.directoryId, [photo]);
          }
        }
        clusterEdited.set(cluster.anchor.id, clusterEdits);
      }
      for (const list of dirPhotos.values()) {
        list.sort(comparePhotosAlphabetically);
      }
    }

    return {
      totalPhotoCount: total,
      totalEditedCount: editedTotal,
      dirPhotosById: dirPhotos as ReadonlyMap<string, readonly PhotoAsset[]>,
      dirEditedCountById: dirEdited as ReadonlyMap<string, number>,
      clusterEditedCountById: clusterEdited,
    };
  }, [session]);

  const filteredDevelopFolders = useMemo(
    () =>
      filterDevelopFolders(
        session?.directories ?? [],
        dirPhotosById,
        dirEditedCountById,
        folderQuery,
      ),
    [session?.directories, dirPhotosById, dirEditedCountById, folderQuery],
  );

  const allProjectPhotos = useMemo(
    () => (syncModalOpen && session ? session.clusters.flatMap((c) => c.photos) : []),
    [syncModalOpen, session],
  );

  const dirNameById = useMemo(
    () => new Map((session?.directories ?? []).map((d) => [d.id, d.name])),
    [session?.directories],
  );

  const filteredSubjects = useMemo(
    () =>
      [
        ...filterSubjectClusters(session?.clusters ?? [], {
          query: subjectQuery,
          filter: subjectFilter,
          editedCountByClusterId: clusterEditedCountById,
        }),
      ].sort(compareSubjectClustersAlphabetically),
    [session?.clusters, subjectQuery, subjectFilter, clusterEditedCountById],
  );

  const studioSubjectCounts = useMemo(() => {
    const clusters = session?.clusters ?? [];
    let edited = 0;
    for (const cluster of clusters) {
      if ((clusterEditedCountById.get(cluster.anchor.id) ?? 0) > 0) {
        edited += 1;
      }
    }
    return { total: clusters.length, edited, unedited: clusters.length - edited };
  }, [session?.clusters, clusterEditedCountById]);

  useEffect(() => {
    setFolderQuery('');
    setExpandedFolderIds(new Set());
    setExpandedFullFolderIds(new Set());
    setSubjectQuery('');
    setSubjectFilter('all');
    setZoomFactor(1);
    setCompareMode('single');
    setActiveMaskId(null);
    setActiveMaskComponentId(null);
    setCreatingNewMask(false);
    setSelectedSpotId(null);
    setPendingRemoveStrokes([]);
    autoFaceScanAttemptedRef.current.clear();
    autoFaceParseAttemptedRef.current.clear();
    setEyedropperActive(false);
    setColorPickActive(false);
  }, [session?.folderPath]);

  const committedActivePhotoId = committedActiveDevelopPhoto?.id ?? null;
  const activePhotoIndex = useMemo(
    () => (committedActivePhotoId ? developPhotos.findIndex((p) => p.id === committedActivePhotoId) : -1),
    [developPhotos, committedActivePhotoId],
  );

  const stepPhoto = useCallback(
    (offset: number) => {
      if (developPhotos.length === 0 || activePhotoIndex < 0) return;
      const nextIdx = (activePhotoIndex + offset + developPhotos.length) % developPhotos.length;
      const nextPhoto = developPhotos[nextIdx];
      if (nextPhoto) setActiveDevelopPhotoId(nextPhoto.id);
    },
    [developPhotos, activePhotoIndex, setActiveDevelopPhotoId],
  );

  useEffect(() => {
    if (!isActive || developSourceMode !== 'subjects') {
      clearSubjectHover();
    }
    if (!isActive || developSourceMode !== 'folders') {
      clearFolderPhotoHover();
    }
  }, [isActive, developSourceMode, clearSubjectHover, clearFolderPhotoHover]);

  const workingMaxDim = preferences.studioPreviewMaxDim ?? DEVELOP_WORKING_MAX_DIM;

  // Quietly pre-decode neighboring JPEG/PNG frames (+1, -1, +2, -2) into the shared downscaled
  // ImageBitmap LRU cache so stepping between standard photos uploads synchronously in <3 ms.
  useEffect(() => {
    if (!isActive || activePhotoIndex < 0 || developPhotos.length <= 1) return;
    const timer = window.setTimeout(() => {
      for (const offset of [1, -1, 2, -2]) {
        const idx = activePhotoIndex + offset;
        if (idx < 0 || idx >= developPhotos.length) continue;
        const neighbor = developPhotos[idx];
        const neighborSrc = neighbor?.studioSourceUrl ?? neighbor?.imageUrl;
        if (!neighborSrc) continue;
        if (isDesktopRuntime && shouldProbeHighBitDepthBuffer(neighbor.fileName)) continue;
        preloadDevelopBitmap(neighborSrc, workingMaxDim);
      }
    }, 60);
    return () => window.clearTimeout(timer);
  }, [isActive, isDesktopRuntime, activePhotoIndex, developPhotos, workingMaxDim]);

  // Quietly pre-decode adjacent RAW working buffers (+1, -1, +2) after a brief dwell so stepping
  // forward or back in Studio hits the in-memory LRU cache in 0 ms without competing with the active frame.
  useEffect(() => {
    if (!isActive || !isDesktopRuntime || activePhotoIndex < 0 || developPhotos.length <= 1) {
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void (async () => {
        for (const offset of [1, -1, 2]) {
          if (cancelled) return;
          const idx = activePhotoIndex + offset;
          if (idx < 0 || idx >= developPhotos.length) continue;
          const neighbor = developPhotos[idx];
          if (!neighbor || !shouldProbeHighBitDepthBuffer(neighbor.fileName)) continue;
          if (peekDevelopWorkingBuffer(neighbor.id, workingMaxDim, 'f16')) continue;
          const loaded = await loadDevelopWorkingBuffer(
            neighbor.id,
            workingMaxDim,
            'f16',
            'prefetch',
          );
          if (!loaded && isRawFileName(neighbor.fileName)) {
            // Yield immediately if an active click or 1:1 zoom preempted background prefetching.
            break;
          }
        }
      })();
    }, 260);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    isActive,
    isDesktopRuntime,
    activePhotoIndex,
    developPhotos,
    workingMaxDim,
    peekDevelopWorkingBuffer,
    loadDevelopWorkingBuffer,
  ]);

  const currentMasks = activeDevelopPhoto?.developRecipe.masks ?? [];
  const currentBgRemoval =
    activeDevelopPhoto?.developRecipe.backgroundRemoval ?? createDefaultBackgroundRemoval();
  const activeMaskLayer =
    currentMasks.find((m) => m.id === activeMaskId) ?? currentMasks[0] ?? null;

  const handleChangeToolMode = useCallback(
    (nextMode: DevelopToolMode) => {
      setEyedropperActive(false);
      setColorPickActive(false);
      setToolMode(nextMode);
      if (nextMode === 'mask') {
        setCreatingNewMask(currentMasks.length === 0);
      } else {
        setCreatingNewMask(false);
      }
    },
    [currentMasks.length],
  );

  // Lightroom-compatible Studio keyboard shortcuts
  useEffect(() => {
    if (!isActive) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        if (developSourceMode === 'subjects') {
          subjectSearchInputRef.current?.focus();
          subjectSearchInputRef.current?.select();
        } else {
          folderSearchInputRef.current?.focus();
          folderSearchInputRef.current?.select();
        }
        return;
      }

      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }

      if ((event.ctrlKey || event.metaKey) && event.shiftKey) {
        const key = event.key.toLowerCase();
        if (key === 's') {
          if (session?.sessionMode === 'single') return;
          event.preventDefault();
          setSyncModalOpen(true);
        } else if (key === 'c') {
          event.preventDefault();
          copyActiveRecipe();
        } else if (key === 'v') {
          event.preventDefault();
          pasteRecipeToClusterSelection();
        }
        return;
      }

      if (!event.ctrlKey && !event.metaKey && !event.altKey) {
        if (event.key === 'Tab' && !syncModalOpen) {
          event.preventDefault();
          toggleStudioFocusMode();
          return;
        }
        const lower = event.key.toLowerCase();
        if (lower === 'e') {
          event.preventDefault();
          handleChangeToolMode('edit');
          return;
        }
        if (lower === 'r') {
          event.preventDefault();
          handleChangeToolMode(toolMode === 'crop' ? 'edit' : 'crop');
          return;
        }
        if (lower === 'q') {
          event.preventDefault();
          handleChangeToolMode(toolMode === 'retouch' ? 'edit' : 'retouch');
          return;
        }
        if (lower === 'm') {
          event.preventDefault();
          handleChangeToolMode(toolMode === 'mask' ? 'edit' : 'mask');
          return;
        }
        if (lower === 'p') {
          event.preventDefault();
          handleChangeToolMode(toolMode === 'presets' ? 'edit' : 'presets');
          return;
        }
        if (lower === 'h' && toolMode === 'retouch') {
          event.preventDefault();
          setShowSpotPins((v) => !v);
          return;
        }
        if (lower === 'w') {
          event.preventDefault();
          if (toolMode === 'crop') setToolMode('edit');
          setColorPickActive(false);
          setEyedropperActive((prev) => !prev);
          return;
        }
        if (lower === 'j') {
          event.preventDefault();
          const anyOn = showShadowClip || showHighlightClip;
          setShowShadowClip(!anyOn);
          setShowHighlightClip(!anyOn);
          return;
        }
        if (lower === 'o') {
          event.preventDefault();
          setShowMaskOverlay((prev) => !prev);
          return;
        }
        if (lower === 'z') {
          event.preventDefault();
          setZoomFactor((z) => (z > 1.02 ? 1 : Number(oneToOneZoom.toFixed(4))));
          return;
        }
        if (toolMode === 'mask' && (event.key === '[' || event.key === ']')) {
          event.preventDefault();
          setBrushOptions((prev) => ({
            ...prev,
            size: Math.min(0.25, Math.max(0.008, prev.size + (event.key === ']' ? 0.012 : -0.012))),
          }));
          return;
        }
        if (toolMode === 'retouch' && (event.key === '[' || event.key === ']')) {
          event.preventDefault();
          setHealTool((prev) => ({
            ...prev,
            radius: Math.min(
              0.15,
              Math.max(0.004, Number((prev.radius + (event.key === ']' ? 0.004 : -0.004)).toFixed(4))),
            ),
          }));
          return;
        }
        if (
          toolMode === 'retouch' &&
          selectedSpotId &&
          (event.key === 'Delete' || event.key === 'Backspace')
        ) {
          event.preventDefault();
          handleDeleteSpotRef.current(selectedSpotId);
          return;
        }
        if (event.key === 'Escape') {
          let handled = false;
          if (eyedropperActive) {
            setEyedropperActive(false);
            handled = true;
          }
          if (colorPickActive) {
            setColorPickActive(false);
            handled = true;
          }
          if (creatingNewMask && currentMasks.length > 0) {
            setCreatingNewMask(false);
            handled = true;
          }
          if (toolMode === 'crop') {
            setToolMode('edit');
            handled = true;
          }
          if (toolMode === 'retouch') {
            if (pendingRemoveStrokes.length > 0) {
              setPendingRemoveStrokes([]);
            } else if (selectedSpotId) {
              setSelectedSpotId(null);
            } else {
              setToolMode('edit');
            }
            handled = true;
          }
          if (!handled && studioFocusMode) {
            setStudioFocusMode(false);
          }
          return;
        }
        if (event.key === 'Enter' && toolMode === 'crop') {
          event.preventDefault();
          setToolMode('edit');
          return;
        }
        if (
          event.key === 'Enter' &&
          toolMode === 'retouch' &&
          healTool.mode === 'remove' &&
          pendingRemoveStrokes.length > 0
        ) {
          event.preventDefault();
          void handleApplyInpaintRef.current();
          return;
        }
      }

      if (session?.sessionMode !== 'single') {
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          stepPhoto(1);
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          stepPhoto(-1);
        }
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    isActive,
    session?.sessionMode,
    developSourceMode,
    copyActiveRecipe,
    pasteRecipeToClusterSelection,
    stepPhoto,
    showShadowClip,
    showHighlightClip,
    eyedropperActive,
    colorPickActive,
    creatingNewMask,
    currentMasks.length,
    toolMode,
    healTool.mode,
    pendingRemoveStrokes.length,
    selectedSpotId,
    handleChangeToolMode,
    syncModalOpen,
    studioFocusMode,
    setStudioFocusMode,
    toggleStudioFocusMode,
    oneToOneZoom,
  ]);

  const activePhotoId = activeDevelopPhoto?.id ?? null;
  const activePhotoName = activeDevelopPhoto?.fileName ?? '';
  const handleChangeActiveRecipe = useCallback(
    (next: DevelopRecipe, label?: string, coalesce?: boolean) => {
      if (activePhotoId) {
        updatePhotoRecipe(activePhotoId, next, label, coalesce);
      }
    },
    [activePhotoId, updatePhotoRecipe],
  );

  const handleApplyPreset = useCallback(
    (presetName: string, recipe: DevelopRecipe) => {
      if (!activePhotoId) return;
      updatePhotoRecipe(activePhotoId, recipe, `Preset: ${presetName}`, false);
      notifyAction(`Applied preset "${presetName}" to ${activePhotoName}`);
    },
    [activePhotoId, activePhotoName, updatePhotoRecipe, notifyAction],
  );

  const handleOpenSyncModal = useCallback(() => setSyncModalOpen(true), []);

  useEffect(() => {
    setColorPickActive(false);
    setZoomFactor(1);
    setCreatingNewMask(false);
    setSelectedSpotId(null);
    setPendingRemoveStrokes([]);
    activeStrokeTargetRef.current = null;
  }, [activePhotoId]);

  useEffect(() => {
    if (!retouchActive || healTool.mode !== 'remove') {
      setPendingRemoveStrokes([]);
    }
  }, [retouchActive, healTool.mode]);

  useEffect(() => {
    autoFaceParseAttemptedRef.current.clear();
  }, [runtimeInfo?.faceParserModelPath]);

  const handleDetectFace = useCallback(
    async (silent = false) => {
      if (!activePhotoId || !isDesktopRuntime || detectingFace) return;
      autoFaceScanAttemptedRef.current.add(activePhotoId);
      setDetectingFace(true);
      try {
        const overlay = await detectPhotoFaceOverlay(activePhotoId);
        if (!silent) {
          notifyAction(
            overlay
              ? 'YuNet 5-point facial landmarks locked for portrait retouch'
              : 'No face detected — skin-tone retouch mode active',
          );
        }
      } finally {
        setDetectingFace(false);
      }
    },
    [activePhotoId, isDesktopRuntime, detectingFace, detectPhotoFaceOverlay, notifyAction],
  );

  // Auto-run fast (~12ms) YuNet landmark scan when entering Retouch mode on an un-scanned frame
  useEffect(() => {
    if (
      !retouchActive ||
      !isDesktopRuntime ||
      !activeDevelopPhoto ||
      activeDevelopPhoto.faceOverlay ||
      detectingFace
    ) {
      return;
    }
    if (autoFaceScanAttemptedRef.current.has(activeDevelopPhoto.id)) return;
    void handleDetectFace(true);
  }, [retouchActive, isDesktopRuntime, activeDevelopPhoto, detectingFace, handleDetectFace]);

  // Auto-run optional Phase 3 BiSeNet 19-class neural face parsing whenever BiSeNet is installed
  const activeFaceParseMap = activeDevelopPhoto
    ? (faceParseMapsByPhotoId[activeDevelopPhoto.id] ?? null)
    : null;
  const activeCropOriKey = activeDevelopPhoto
    ? `${activeDevelopPhoto.id}:${(activeDevelopPhoto.developRecipe.cropTransform.quarterTurns ?? 0) % 4}:${activeDevelopPhoto.developRecipe.cropTransform.flipHorizontal ? 1 : 0}:${activeDevelopPhoto.developRecipe.cropTransform.flipVertical ? 1 : 0}`
    : '';
  const faceParserReady = Boolean(runtimeInfo?.faceParserReady);
  const lamaReady = Boolean(runtimeInfo?.lamaReady);

  useEffect(() => {
    if (
      !retouchActive ||
      !isDesktopRuntime ||
      !faceParserReady ||
      !activePhotoId ||
      !activeCropOriKey ||
      parsingFace
    ) {
      return;
    }
    if (autoFaceParseAttemptedRef.current.has(activeCropOriKey)) return;
    autoFaceParseAttemptedRef.current.add(activeCropOriKey);
    let cancelled = false;
    setParsingFace(true);
    void parsePhotoFaceRetouchMap(activePhotoId).finally(() => {
      if (!cancelled) {
        setParsingFace(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [
    retouchActive,
    isDesktopRuntime,
    faceParserReady,
    activePhotoId,
    activeCropOriKey,
    parsingFace,
    parsePhotoFaceRetouchMap,
  ]);

  const handleApplyInpaint = useCallback(async () => {
    if (
      !activeDevelopPhoto ||
      !isDesktopRuntime ||
      busyInpainting ||
      pendingRemoveStrokes.length === 0
    ) {
      return;
    }
    setBusyInpainting(true);
    try {
      const generated = await inpaintPhotoRegion(activeDevelopPhoto.id, pendingRemoveStrokes);
      if (!generated) return;
      const curRetouch = hydrateRetouchSettings(activeDevelopPhoto.developRecipe.retouch);
      const nextPatches = [...(curRetouch.inpaintPatches ?? []), generated.patch].slice(
        -MAX_INPAINT_PATCHES,
      );
      handleChangeActiveRecipe(
        {
          ...activeDevelopPhoto.developRecipe,
          retouch: { ...curRetouch, inpaintPatches: nextPatches },
        },
        'AI Object Remove',
        false,
      );
      setPendingRemoveStrokes([]);
      notifyAction(
        `Removed object with ${generated.patch.engineLabel} (${generated.elapsedMs}ms)`,
      );
    } finally {
      setBusyInpainting(false);
    }
  }, [
    activeDevelopPhoto,
    isDesktopRuntime,
    busyInpainting,
    pendingRemoveStrokes,
    inpaintPhotoRegion,
    handleChangeActiveRecipe,
    notifyAction,
  ]);
  const handleApplyInpaintRef = useRef(handleApplyInpaint);
  handleApplyInpaintRef.current = handleApplyInpaint;

  const handleDeleteInpaintPatch = useCallback(
    (patchId: string) => {
      if (!activeDevelopPhoto) return;
      const curRetouch = hydrateRetouchSettings(activeDevelopPhoto.developRecipe.retouch);
      const curPatches = curRetouch.inpaintPatches ?? [];
      const nextPatches = curPatches.filter((p) => p.id !== patchId);
      if (nextPatches.length === curPatches.length) return;
      handleChangeActiveRecipe(
        {
          ...activeDevelopPhoto.developRecipe,
          retouch: { ...curRetouch, inpaintPatches: nextPatches },
        },
        'Delete Object Removal',
        false,
      );
    },
    [activeDevelopPhoto, handleChangeActiveRecipe],
  );

  const handleClearAllInpaintPatches = useCallback(() => {
    if (!activeDevelopPhoto) return;
    const curRetouch = hydrateRetouchSettings(activeDevelopPhoto.developRecipe.retouch);
    if ((curRetouch.inpaintPatches ?? []).length === 0) return;
    handleChangeActiveRecipe(
      {
        ...activeDevelopPhoto.developRecipe,
        retouch: { ...curRetouch, inpaintPatches: [] },
      },
      'Clear Object Removals',
      false,
    );
  }, [activeDevelopPhoto, handleChangeActiveRecipe]);

  const handleChangeHealSpots = useCallback(
    (nextSpots: readonly HealSpot[], label: string, coalesce = true) => {
      if (!activeDevelopPhoto) return;
      const curRetouch = hydrateRetouchSettings(activeDevelopPhoto.developRecipe.retouch);
      handleChangeActiveRecipe(
        {
          ...activeDevelopPhoto.developRecipe,
          retouch: { ...curRetouch, healSpots: [...nextSpots] },
        },
        label,
        coalesce,
      );
    },
    [activeDevelopPhoto, handleChangeActiveRecipe],
  );

  const handleDeleteSpot = useCallback(
    (spotId: string) => {
      if (!activeDevelopPhoto) return;
      const curRetouch = hydrateRetouchSettings(activeDevelopPhoto.developRecipe.retouch);
      const nextSpots = curRetouch.healSpots.filter((s) => s.id !== spotId);
      if (nextSpots.length === curRetouch.healSpots.length) return;
      handleChangeHealSpots(nextSpots, 'Delete Heal Spot', false);
      setSelectedSpotId((prev) => (prev === spotId ? null : prev));
    },
    [activeDevelopPhoto, handleChangeHealSpots],
  );
  const handleDeleteSpotRef = useRef(handleDeleteSpot);
  handleDeleteSpotRef.current = handleDeleteSpot;

  const handleClearAllSpots = useCallback(() => {
    if (!activeDevelopPhoto) return;
    const curRetouch = hydrateRetouchSettings(activeDevelopPhoto.developRecipe.retouch);
    if (curRetouch.healSpots.length === 0) return;
    handleChangeHealSpots([], 'Clear All Heal Spots', false);
    setSelectedSpotId(null);
  }, [activeDevelopPhoto, handleChangeHealSpots]);

  const handleSelectMask = useCallback((layerId: string | null, componentId?: string | null) => {
    setActiveMaskId(layerId);
    setActiveMaskComponentId(componentId ?? null);
    setCreatingNewMask(false);
    setColorPickActive(false);
  }, []);

  const handleChangeBackgroundRemoval = useCallback(
    (nextBg: BackgroundRemovalSettings, label: string, coalesce = true) => {
      if (!activeDevelopPhoto) return;
      handleChangeActiveRecipe(
        { ...activeDevelopPhoto.developRecipe, backgroundRemoval: nextBg },
        label,
        coalesce,
      );
    },
    [activeDevelopPhoto, handleChangeActiveRecipe],
  );

  const handleRemoveBackground = useCallback(async () => {
    if (!activeDevelopPhoto) return;
    const existingBg =
      activeDevelopPhoto.developRecipe.backgroundRemoval ?? createDefaultBackgroundRemoval();

    // 1. Instant reuse if this photo already has a cached background removal matte
    if (existingBg.matteWidth > 0 && existingBg.matteHeight > 0 && existingBg.matteRleB64) {
      handleChangeBackgroundRemoval(
        {
          ...existingBg,
          enabled: true,
          mode: existingBg.mode === 'original' ? 'transparent' : existingBg.mode,
        },
        'Remove Background',
        false,
      );
      return;
    }

    // 2. Instant reuse if an AI Subject matte already exists in any local mask layer
    for (const layer of activeDevelopPhoto.developRecipe.masks ?? []) {
      const subjectComp = layer.components.find(
        (c) => c.type === 'ai_matte' && c.kind === 'subject' && c.matteRleB64.length > 0,
      );
      if (subjectComp && subjectComp.type === 'ai_matte') {
        handleChangeBackgroundRemoval(
          {
            ...createDefaultBackgroundRemoval(),
            enabled: true,
            mode: 'transparent',
            matteWidth: subjectComp.matteWidth,
            matteHeight: subjectComp.matteHeight,
            matteRleB64: subjectComp.matteRleB64,
          },
          'Remove Background',
          false,
        );
        notifyAction('Background removed (reused Subject matte)');
        return;
      }
    }

    // 3. Generate real RMBG-1.4 Subject matte via Rust ONNX backend
    setBusyBgRemoval(true);
    try {
      const generated = await generateAiMask(activeDevelopPhoto.id, 'subject');
      if (!generated) return;
      handleChangeBackgroundRemoval(
        {
          ...createDefaultBackgroundRemoval(),
          enabled: true,
          mode: 'transparent',
          matteWidth: generated.matteWidth,
          matteHeight: generated.matteHeight,
          matteRleB64: generated.matteRleB64,
        },
        'Remove Background',
        false,
      );
      notifyAction('Background removed');
    } finally {
      setBusyBgRemoval(false);
    }
  }, [activeDevelopPhoto, generateAiMask, handleChangeBackgroundRemoval, notifyAction]);

  const handleToggleBackgroundRemoval = useCallback(() => {
    if (!activeDevelopPhoto) return;
    const bg =
      activeDevelopPhoto.developRecipe.backgroundRemoval ?? createDefaultBackgroundRemoval();
    if (isBackgroundRemovalActive(bg)) {
      handleChangeBackgroundRemoval(
        { ...bg, enabled: false, mode: 'original' },
        'Restore Original Background',
        false,
      );
    } else {
      void handleRemoveBackground();
    }
  }, [activeDevelopPhoto, handleChangeBackgroundRemoval, handleRemoveBackground]);

  const handleCreateAiMask = useCallback(
    async (kind: AiMaskKind, attachToLayerId?: string, op: MaskCombineOp = 'add') => {
      if (!activeDevelopPhoto) return;
      setBusyAiMaskKind(kind);
      try {
        const generated = await generateAiMask(activeDevelopPhoto.id, kind);
        if (!generated) return;
        const comp = createAiMatteComponent(generated, op);
        const existingMasks = activeDevelopPhoto.developRecipe.masks ?? [];
        if (attachToLayerId) {
          const nextMasks = existingMasks.map((m) =>
            m.id === attachToLayerId ? { ...m, components: [...m.components, comp] } : m,
          );
          handleChangeActiveRecipe(
            { ...activeDevelopPhoto.developRecipe, masks: nextMasks },
            `${op} AI ${aiMaskKindLabel(kind)}`,
            false,
          );
          setActiveMaskId(attachToLayerId);
          setActiveMaskComponentId(comp.id);
        } else {
          const nextLayer = createMaskLayer(`AI ${aiMaskKindLabel(kind)}`, comp);
          handleChangeActiveRecipe(
            { ...activeDevelopPhoto.developRecipe, masks: [...existingMasks, nextLayer] },
            `AI ${aiMaskKindLabel(kind)} Mask`,
            false,
          );
          setActiveMaskId(nextLayer.id);
          setActiveMaskComponentId(comp.id);
        }
        setCreatingNewMask(false);
        setShowMaskOverlay(true);
        notifyAction(`${aiMaskKindLabel(kind)} mask ready`);
      } finally {
        setBusyAiMaskKind(null);
      }
    },
    [activeDevelopPhoto, generateAiMask, handleChangeActiveRecipe, notifyAction],
  );

  const handleAttachManualComponent = useCallback(
    (
      layerId: string,
      type: 'brush' | 'linear_gradient' | 'radial_gradient' | 'luminance_range' | 'color_range',
      op: MaskCombineOp,
    ) => {
      if (!activeDevelopPhoto) return;
      const comp = createManualMaskComponent(type, op);
      const existingMasks = activeDevelopPhoto.developRecipe.masks ?? [];
      const nextMasks = existingMasks.map((m) =>
        m.id === layerId ? { ...m, components: [...m.components, comp] } : m,
      );
      handleChangeActiveRecipe(
        { ...activeDevelopPhoto.developRecipe, masks: nextMasks },
        `${op} ${maskComponentLabel(comp)}`,
        false,
      );
      setActiveMaskId(layerId);
      setActiveMaskComponentId(comp.id);
      setCreatingNewMask(false);
      if (type === 'brush') {
        setBrushOptions((prev) => ({ ...prev, erase: op === 'subtract' }));
      }
    },
    [activeDevelopPhoto, handleChangeActiveRecipe],
  );

  const handleUpdateMaskComponent = useCallback(
    (layerId: string, nextComp: MaskComponent, coalesce = true) => {
      if (!activeDevelopPhoto) return;
      const existingMasks = activeDevelopPhoto.developRecipe.masks ?? [];
      const nextMasks = existingMasks.map((m) =>
        m.id === layerId
          ? {
              ...m,
              components: m.components.map((c) => (c.id === nextComp.id ? nextComp : c)),
            }
          : m,
      );
      handleChangeActiveRecipe(
        { ...activeDevelopPhoto.developRecipe, masks: nextMasks },
        'Adjust Mask',
        coalesce,
      );
    },
    [activeDevelopPhoto, handleChangeActiveRecipe],
  );

  const handleEnsureBrushStroke = useCallback(
    (layerId: string, mode: 'add' | 'subtract', stroke: BrushStroke, isFinal: boolean) => {
      if (!activeDevelopPhoto) return;
      const existingMasks = activeDevelopPhoto.developRecipe.masks ?? [];
      const targetLayer = existingMasks.find((m) => m.id === layerId);
      if (!targetLayer) return;

      const inFlight = activeStrokeTargetRef.current;
      let compId = inFlight && inFlight.layerId === layerId ? inFlight.compId : null;
      let strokeIndex = inFlight && inFlight.layerId === layerId ? inFlight.strokeIndex : -1;

      const nextMasks = existingMasks.map((m) => {
        if (m.id !== layerId) return m;
        if (compId !== null && strokeIndex >= 0) {
          return {
            ...m,
            components: m.components.map((c) => {
              if (c.id !== compId || c.type !== 'brush') return c;
              const nextStrokes = [...c.strokes];
              nextStrokes[strokeIndex] = stroke;
              return { ...c, strokes: nextStrokes };
            }),
          };
        }
        const selectedComp = m.components.find((c) => c.id === activeMaskComponentId);
        const lastComp = m.components[m.components.length - 1];
        const reuseComp =
          selectedComp && selectedComp.type === 'brush' && selectedComp.op === mode
            ? selectedComp
            : lastComp && lastComp.type === 'brush' && lastComp.op === mode
              ? lastComp
              : null;

        if (reuseComp) {
          compId = reuseComp.id;
          strokeIndex = reuseComp.strokes.length;
          return {
            ...m,
            components: m.components.map((c) =>
              c.id === reuseComp.id && c.type === 'brush'
                ? { ...c, strokes: [...c.strokes, stroke] }
                : c,
            ),
          };
        }

        const created = { ...createBrushComponent(mode), strokes: [stroke] };
        compId = created.id;
        strokeIndex = 0;
        return { ...m, components: [...m.components, created] };
      });

      if (isFinal) {
        activeStrokeTargetRef.current = null;
      } else if (compId !== null && strokeIndex >= 0) {
        activeStrokeTargetRef.current = { layerId, compId, strokeIndex };
        setActiveMaskComponentId(compId);
      }

      handleChangeActiveRecipe(
        { ...activeDevelopPhoto.developRecipe, masks: nextMasks },
        mode === 'subtract' ? 'Erase Mask Brush' : 'Paint Mask Brush',
        !isFinal,
      );
    },
    [activeDevelopPhoto, activeMaskComponentId, handleChangeActiveRecipe],
  );

  const handleSampleColorRangeRgb = useCallback(
    (r: number, g: number, b: number) => {
      if (!activeDevelopPhoto || !activeMaskLayer) return;
      const [h, s, l] = rgbToHsl([r, g, b]);
      const targetComp =
        activeMaskLayer.components.find(
          (c) => c.id === activeMaskComponentId && c.type === 'color_range',
        ) ?? activeMaskLayer.components.find((c) => c.type === 'color_range');
      if (!targetComp || targetComp.type !== 'color_range') {
        setColorPickActive(false);
        return;
      }
      handleUpdateMaskComponent(
        activeMaskLayer.id,
        {
          ...targetComp,
          targetHueDeg: Math.round(h),
          targetSat: Number(s.toFixed(3)),
          targetLuma: Number(l.toFixed(3)),
        },
        false,
      );
      setColorPickActive(false);
      notifyAction(
        `Color Range picked: H ${Math.round(h)}° · S ${Math.round(s * 100)}% · L ${Math.round(l * 100)}%`,
      );
    },
    [
      activeDevelopPhoto,
      activeMaskLayer,
      activeMaskComponentId,
      handleUpdateMaskComponent,
      notifyAction,
    ],
  );

  if (!session || !activeDevelopPhoto) return null;

  const isSinglePhoto = session.sessionMode === 'single';
  const hideLeftSidebar =
    studioFocusMode || isSinglePhoto || (session.sessionMode === 'files' && !session.isClustered);
  const hideFilmstrip = studioFocusMode || isSinglePhoto;

  const hasSlateAnchor =
    !isSinglePhoto &&
    developSourceMode === 'subjects' &&
    activeCluster !== null &&
    !isFaceOnlySlateCode(activeCluster.anchor.slateCode);
  const modified = isRecipeModified(activeDevelopPhoto.developRecipe);

  return (
    <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden bg-canvas">
      {/* Left Sidebar: Full-Height RAW Folders & Subjects Browser (hidden in Focus / Single Photo Mode) */}
      {!hideLeftSidebar && (
        <>
          <aside
            aria-label="RAW Folders and Subjects"
            style={{ width: `${leftPanel.width}px` }}
            className="shrink-0 bg-surface-low flex flex-col min-h-0 overflow-hidden px-3 py-3 gap-3"
          >
        <div className="shrink-0">
          {session.isClustered ? (
            <SegmentedControl
              layoutId="develop-source-mode"
              options={SOURCE_MODE_OPTIONS}
              value={developSourceMode}
              onChange={setDevelopSourceMode}
              className="w-full grid grid-cols-2"
            />
          ) : (
            <div className="flex items-center justify-between px-1">
              <span className="font-display text-[11px] uppercase tracking-[0.08em] text-ink-telemetry font-semibold">
                Folders
              </span>
              <button
                type="button"
                onClick={() => void openShootFolder()}
                className="tabular-nums-mono text-[10.5px] text-amber-soft hover:text-ink-primary transition-colors cursor-pointer"
                title="Import another folder"
              >
                + Open
              </button>
            </div>
          )}
        </div>

        {developSourceMode === 'folders' ? (
          <div className="flex-1 min-h-0 flex flex-col gap-2.5">
            <div className="shrink-0 h-8 px-2.5 rounded-lg bg-surface-lowest/80 border border-hairline/70 focus-within:border-amber-core/60 transition-colors flex items-center gap-2">
              <span className="material-symbols-outlined text-[14px] text-ink-telemetry shrink-0">
                search
              </span>
              <input
                ref={folderSearchInputRef}
                type="text"
                value={folderQuery}
                onChange={(e) => {
                  setFolderQuery(e.target.value);
                  setCollapsedSearchFolderIds(new Set());
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Escape' && folderQuery) {
                    e.stopPropagation();
                    setFolderQuery('');
                  }
                }}
                placeholder={`Search ${session.directories.length} folders or files…`}
                aria-label="Search Studio folders or filenames"
                className="bg-transparent text-ink-primary placeholder:text-ink-telemetry text-[11.5px] flex-1 min-w-0 outline-none border-none"
              />
              {folderQuery ? (
                <button
                  type="button"
                  onClick={() => setFolderQuery('')}
                  aria-label="Clear folder search"
                  className="text-ink-telemetry hover:text-ink-primary cursor-pointer flex items-center shrink-0"
                >
                  <span className="material-symbols-outlined text-[13px]">close</span>
                </button>
              ) : (
                <kbd className="tabular-nums-mono text-[9px] text-ink-telemetry/60 shrink-0">
                  Ctrl+K
                </kbd>
              )}
            </div>

            <div
              onMouseLeave={clearFolderPhotoHover}
              onScroll={clearFolderPhotoHover}
              className="flex-1 min-h-0 flex flex-col gap-0.5 overflow-y-auto pr-0.5"
            >
              {folderQuery.trim().length === 0 && (
                <button
                  type="button"
                  onClick={() => selectDevelopFolder(null)}
                  className={`shrink-0 relative flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-left transition-colors duration-150 cursor-pointer ${
                    activeDevelopFolderId === null
                      ? 'bg-surface-active text-ink-primary'
                      : 'text-ink-secondary hover:bg-surface/65 hover:text-ink-primary'
                  }`}
                >
                  {activeDevelopFolderId === null && (
                    <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-amber-core" />
                  )}
                  <div className="min-w-0 pl-0.5">
                    <div className="text-[12px] font-display font-semibold truncate text-ink-primary">
                      All Photos
                    </div>
                    <div className="tabular-nums-mono text-[10px] text-ink-telemetry truncate">
                      {session.telemetry.projectName}
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0 tabular-nums-mono text-[10.5px]">
                    {totalEditedCount > 0 && (
                      <span
                        className="w-1.5 h-1.5 rounded-full bg-amber-core/85"
                        title={`${totalEditedCount} edited`}
                      />
                    )}
                    <span className="text-ink-telemetry">{totalPhotoCount}</span>
                  </div>
                </button>
              )}

              {filteredDevelopFolders.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 px-3 py-8 text-ink-telemetry">
                  <span className="text-[12px] text-ink-secondary">No matching folders or files</span>
                  <DarkroomButton
                    variant="ghost"
                    size="xs"
                    onClick={() => setFolderQuery('')}
                  >
                    Reset Filter
                  </DarkroomButton>
                </div>
              ) : (
                filteredDevelopFolders.map((entry) => {
                  const { directory: dir, totalPhotos: folderTotal, editedPhotos: folderEdited } = entry;
                  const isSelected = activeDevelopFolderId === dir.id;
                  const isExpanded = entry.autoExpand
                    ? !collapsedSearchFolderIds.has(dir.id)
                    : expandedFolderIds.has(dir.id);
                  const showAllPhotos = expandedFullFolderIds.has(dir.id);
                  const maxInlinePhotos = 100;
                  const renderedPhotos =
                    showAllPhotos || entry.visiblePhotos.length <= maxInlinePhotos
                      ? entry.visiblePhotos
                      : entry.visiblePhotos.slice(0, maxInlinePhotos);
                  const hiddenCount = entry.visiblePhotos.length - renderedPhotos.length;

                  return (
                    <div key={dir.id} className="shrink-0 flex flex-col">
                      <div
                        className={`relative flex items-center gap-1 pr-2.5 pl-1 py-1.5 rounded-lg transition-colors duration-150 ${
                          isSelected
                            ? 'bg-surface-active text-ink-primary'
                            : 'text-ink-secondary hover:bg-surface/65 hover:text-ink-primary'
                        }`}
                      >
                        {isSelected && (
                          <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-amber-core" />
                        )}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleFolderExpanded(dir.id, entry.autoExpand);
                          }}
                          aria-expanded={isExpanded}
                          aria-label={isExpanded ? `Collapse ${dir.name}` : `Expand ${dir.name}`}
                          title={isExpanded ? 'Hide photos in folder' : 'Show photos in folder'}
                          className="w-5 h-5 rounded flex items-center justify-center text-ink-telemetry hover:text-ink-primary hover:bg-white/6 transition-colors cursor-pointer shrink-0"
                        >
                          <span className="material-symbols-outlined text-[15px] leading-none">
                            {isExpanded ? 'expand_more' : 'chevron_right'}
                          </span>
                        </button>
                        <button
                          type="button"
                          onClick={() => selectDevelopFolder(dir.id)}
                          title={`${dir.name} (${folderTotal} ${folderTotal === 1 ? 'photo' : 'photos'})`}
                          className="flex-1 min-w-0 flex items-center justify-between gap-2 text-left cursor-pointer py-0.5"
                        >
                          <span className="tabular-nums-mono text-[11.5px] truncate">
                            {dir.name}
                          </span>
                          <div className="flex items-center gap-1.5 shrink-0 tabular-nums-mono text-[10.5px]">
                            {folderEdited > 0 && (
                              <span
                                className="w-1.5 h-1.5 rounded-full bg-amber-core/85"
                                title={`${folderEdited} edited`}
                              />
                            )}
                            <span className="text-ink-telemetry">{folderTotal}</span>
                          </div>
                        </button>
                      </div>

                      {isExpanded && entry.visiblePhotos.length > 0 && (
                        <div className="ml-3.5 pl-2.5 border-l border-hairline/55 flex flex-col gap-0.5 py-0.5 my-0.5">
                          {renderedPhotos.map((photo) => {
                            const isPhotoActive = photo.id === activeDevelopPhoto.id;
                            const isPhotoEdited = isRecipeModified(photo.developRecipe);
                            return (
                              <button
                                key={photo.id}
                                type="button"
                                onClick={() => {
                                  clearFolderPhotoHover();
                                  selectDevelopFolder(dir.id, photo.id);
                                }}
                                onMouseEnter={(e) => scheduleFolderPhotoHover(photo, e.currentTarget)}
                                onMouseLeave={clearFolderPhotoHover}
                                title={photo.fileName}
                                className={`h-7 px-2 rounded-md flex items-center justify-between gap-2 text-left transition-colors duration-100 cursor-pointer ${
                                  isPhotoActive
                                    ? 'bg-surface-active/85 text-amber-soft font-medium'
                                    : 'text-ink-secondary hover:bg-surface/60 hover:text-ink-primary'
                                }`}
                              >
                                <span className="tabular-nums-mono text-[11px] truncate">
                                  {photo.fileName}
                                </span>
                                <div className="flex items-center gap-1.5 shrink-0">
                                  {photo.isSlateAnchor && (
                                    <span className="tabular-nums-mono text-[8.5px] uppercase tracking-wider text-ink-telemetry">
                                      SLATE
                                    </span>
                                  )}
                                  {isPhotoEdited && (
                                    <span
                                      className="w-1.5 h-1.5 rounded-full bg-amber-core/80"
                                      title="Edited"
                                    />
                                  )}
                                </div>
                              </button>
                            );
                          })}
                          {hiddenCount > 0 && (
                            <button
                              type="button"
                              onClick={() =>
                                setExpandedFullFolderIds((prev) => new Set(prev).add(dir.id))
                              }
                              className="h-7 px-2 rounded-md text-left tabular-nums-mono text-[10px] text-amber-soft hover:bg-surface/60 transition-colors cursor-pointer"
                            >
                              Show {hiddenCount} more photo{hiddenCount === 1 ? '' : 's'}…
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        ) : (
          <div className="flex-1 min-h-0 flex flex-col gap-2.5">
            <div className="shrink-0 flex flex-col gap-2">
              <div className="h-8 px-2.5 rounded-lg bg-surface-lowest/80 border border-hairline/70 focus-within:border-amber-core/60 transition-colors flex items-center gap-2">
                <span className="material-symbols-outlined text-[14px] text-ink-telemetry shrink-0">
                  search
                </span>
                <input
                  ref={subjectSearchInputRef}
                  type="text"
                  value={subjectQuery}
                  onChange={(e) => setSubjectQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape' && subjectQuery) {
                      e.stopPropagation();
                      setSubjectQuery('');
                    }
                  }}
                  placeholder={`Search ${studioSubjectCounts.total} subjects…`}
                  aria-label="Search Studio subjects"
                  className="bg-transparent text-ink-primary placeholder:text-ink-telemetry text-[11.5px] flex-1 min-w-0 outline-none border-none"
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
                  <kbd className="tabular-nums-mono text-[9px] text-ink-telemetry/60 shrink-0">
                    Ctrl+K
                  </kbd>
                )}
              </div>

              <div className="flex items-center gap-1 px-0.5">
                {(
                  [
                    { id: 'all', label: 'All', count: studioSubjectCounts.total },
                    { id: 'edited', label: 'Edited', count: studioSubjectCounts.edited },
                    { id: 'unedited', label: 'Unedited', count: studioSubjectCounts.unedited },
                  ] as const
                ).map((chip) => {
                  const active = subjectFilter === chip.id;
                  return (
                    <button
                      key={chip.id}
                      type="button"
                      onClick={() => setSubjectFilter(chip.id)}
                      className={`px-2 py-1 rounded-md tabular-nums-mono text-[10px] transition-colors duration-150 cursor-pointer shrink-0 flex items-center gap-1 ${
                        active
                          ? 'bg-surface-active text-ink-primary font-medium'
                          : 'text-ink-telemetry hover:text-ink-secondary hover:bg-surface/50'
                      }`}
                    >
                      <span>{chip.label}</span>
                      <span className={active ? 'text-amber-soft' : 'opacity-65'}>{chip.count}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {filteredSubjects.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 px-3 text-ink-telemetry">
                <span className="text-[12px] text-ink-secondary">No matching subjects</span>
                <DarkroomButton
                  variant="ghost"
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
              <VirtualVerticalList
                items={filteredSubjects}
                rowHeightPx={54}
                gapPx={2}
                overscan={6}
                getKey={(cluster) => cluster.anchor.id}
                onMouseLeave={clearSubjectHover}
                onScroll={clearSubjectHover}
                className="flex-1 min-h-0 pr-0.5"
                renderItem={(cluster) => {
                  const isCurrent = cluster.anchor.id === activeCluster?.anchor.id;
                  const editedCount = clusterEditedCountById.get(cluster.anchor.id) ?? 0;
                  const faceOnly = isFaceOnlySlateCode(cluster.anchor.slateCode);

                  return (
                    <button
                      type="button"
                      onClick={() => selectCluster(cluster.anchor.id)}
                      onMouseEnter={(e) => scheduleSubjectHover(cluster, e.currentTarget)}
                      className={`relative w-full h-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg text-left transition-colors duration-150 cursor-pointer ${
                        isCurrent
                          ? 'bg-surface-active text-ink-primary'
                          : 'text-ink-secondary hover:bg-surface/65 hover:text-ink-primary'
                      }`}
                    >
                      {isCurrent && (
                        <span className="absolute left-0 top-2.5 bottom-2.5 w-0.5 rounded-full bg-amber-core" />
                      )}
                      <img
                        src={cluster.anchor.thumbnailUrl}
                        alt={cluster.anchor.subjectName}
                        loading="lazy"
                        decoding="async"
                        className="w-8 h-10 rounded-md object-cover object-top shrink-0 bg-canvas ring-1 ring-white/5"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-[12px] font-display font-medium truncate text-ink-primary">
                          {cluster.anchor.subjectName}
                        </div>
                        <div className="tabular-nums-mono text-[9.5px] text-ink-telemetry flex items-center gap-1.5 mt-0.5">
                          <span>{faceOnly ? 'Face-Only' : cluster.anchor.slateCode}</span>
                          <span className="opacity-45">·</span>
                          <span>{cluster.photos.length}</span>
                        </div>
                      </div>
                      {editedCount > 0 && (
                        <span
                          className="flex items-center gap-1 tabular-nums-mono text-[10px] text-ink-telemetry shrink-0 pr-0.5"
                          title={`${editedCount} edited photo(s)`}
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-core/80" />
                          <span>{editedCount}</span>
                        </span>
                      )}
                    </button>
                  );
                }}
              />
            )}
          </div>
        )}
      </aside>

      <PanelResizeHandle
        panelSide="left"
        width={leftPanel.width}
        minWidth={204}
        maxWidth={420}
        onChangeWidth={leftPanel.setWidth}
        onResetWidth={leftPanel.resetWidth}
        label="Resize Folders and Subjects sidebar"
      />
        </>
      )}

      {/* Center Stage: Quiet Header + Photo Viewport (+ Floating Masks Panel) + Lightroom Bottom Bar + Filmstrip */}
      <section className="flex-1 min-w-0 flex flex-col overflow-hidden bg-canvas relative">
        {/* Quiet Top Viewport Bar (hidden in Tab Focus Mode) */}
        {!studioFocusMode && (
        <div className="h-10 px-4 bg-canvas border-b border-hairline/60 flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            {!isSinglePhoto && (
              <DarkroomButton
                variant="ghost"
                size="xs"
                icon="grid_view"
                onClick={() => setActiveWorkspace('culling')}
                title="Back to Grid"
                aria-label="Back to Grid"
              />
            )}
            <span
              className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                modified ? 'bg-amber-core' : 'bg-ink-telemetry/45'
              }`}
              title={modified ? 'Edited' : 'Original RAW'}
            />
            <span className="tabular-nums-mono text-[12px] font-medium text-ink-primary truncate">
              {activeDevelopPhoto.fileName}
            </span>
            {!isSinglePhoto && (
              <span className="hidden xl:inline tabular-nums-mono text-[10.5px] text-ink-telemetry truncate">
                {activeDevelopScopeLabel} · {activePhotoIndex + 1} / {developPhotos.length}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <DarkroomButton
              variant="ghost"
              size="xs"
              icon="undo"
              disabled={!canUndoDevelop}
              onClick={undoDevelopAction}
              title={undoDevelopLabel ? `Undo ${undoDevelopLabel} (Ctrl+Z)` : 'Undo (Ctrl+Z)'}
              aria-label="Undo Studio edit"
            />
            <DarkroomButton
              variant="ghost"
              size="xs"
              icon="redo"
              disabled={!canRedoDevelop}
              onClick={redoDevelopAction}
              title={
                redoDevelopLabel
                  ? `Redo ${redoDevelopLabel} (Ctrl+Shift+Z)`
                  : 'Redo (Ctrl+Shift+Z)'
              }
              aria-label="Redo Studio edit"
            />

            {!isSinglePhoto && (
              <>
                <div className="h-3.5 w-px bg-hairline/70 mx-1 shrink-0" />
                <DarkroomButton
                  variant="ghost"
                  size="xs"
                  icon="chevron_left"
                  onClick={() => stepPhoto(-1)}
                  title="Previous frame (←)"
                  aria-label="Previous photo"
                />
                <DarkroomButton
                  variant="ghost"
                  size="xs"
                  icon="chevron_right"
                  onClick={() => stepPhoto(1)}
                  title="Next frame (→)"
                  aria-label="Next photo"
                />
              </>
            )}
          </div>
        </div>
        )}

        {/* Discrete corner exit button & brief auto-fading hint in Tab Focus Mode */}
        {studioFocusMode && (
          <>
            <button
              type="button"
              onClick={() => setStudioFocusMode(false)}
              title="Show panels (Tab or Esc)"
              aria-label="Exit focus view"
              className="absolute top-3 right-3 z-40 h-8 w-8 rounded-lg bg-surface/75 hover:bg-surface-active border border-hairline-strong text-ink-secondary hover:text-ink-primary flex items-center justify-center opacity-20 hover:opacity-100 focus:opacity-100 transition-opacity cursor-pointer"
            >
              <span className="material-symbols-outlined text-[18px]">fullscreen_exit</span>
            </button>
            {showFocusHint && (
              <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-40 pointer-events-none">
                <TelemetryPill tone="neutral" className="bg-surface-lowest/90 backdrop-blur shadow-lg">
                  Press Tab or Esc to show panels • ← / → to browse
                </TelemetryPill>
              </div>
            )}
          </>
        )}

        {/* Interactive Full-Stage Loupe / Split-Wipe / Side-by-Side + Floating Lightroom Masks Layers Card */}
        <BeforeAfterCanvas
          photo={activeDevelopPhoto}
          compareMode={compareMode}
          zoomFactor={zoomFactor}
          onChangeZoomFactor={setZoomFactor}
          onOneToOneZoomChange={setOneToOneZoom}
          showShadowClip={showShadowClip}
          showHighlightClip={showHighlightClip}
          cropModeActive={cropModeActive && !studioFocusMode}
          onChangeCrop={(nextCrop) =>
            handleChangeActiveRecipe(
              { ...activeDevelopPhoto.developRecipe, cropTransform: nextCrop },
              'Crop & Transform',
            )
          }
          onNaturalAspectChange={setActivePhotoAspect}
          eyedropperActive={eyedropperActive && !studioFocusMode}
          onSampleNeutralRgb={(r, g, b) => {
            const wb = calibrateWhiteBalanceFromSampledRgb(r, g, b);
            handleChangeActiveRecipe(
              { ...activeDevelopPhoto.developRecipe, whiteBalance: wb },
              'WB Eyedropper',
              false,
            );
            setEyedropperActive(false);
            notifyAction(
              `White Balance sampled: ${wb.temperatureKelvin}K (${wb.tint >= 0 ? '+' : ''}${wb.tint})`,
            );
          }}
          maskingActive={maskingActive && !studioFocusMode}
          activeMaskLayer={activeMaskLayer}
          activeMaskComponentId={activeMaskComponentId}
          showMaskOverlay={showMaskOverlay}
          brushOptions={brushOptions}
          colorPickActive={colorPickActive && !studioFocusMode}
          onSampleColorRangeRgb={handleSampleColorRangeRgb}
          onUpdateMaskComponent={handleUpdateMaskComponent}
          onEnsureBrushStroke={handleEnsureBrushStroke}
          retouchActive={retouchActive && !studioFocusMode}
          healTool={healTool}
          selectedSpotId={selectedSpotId}
          showSpotPins={showSpotPins}
          faceParseMap={activeFaceParseMap}
          pendingRemoveStrokes={pendingRemoveStrokes}
          busyInpainting={busyInpainting}
          onSelectSpot={setSelectedSpotId}
          onChangeHealSpots={handleChangeHealSpots}
          onChangePendingRemoveStrokes={setPendingRemoveStrokes}
          focusMode={studioFocusMode}
          floatingOverlay={
            maskingActive && !studioFocusMode ? (
              <FloatingMasksLayersPanel
                masks={currentMasks}
                backgroundRemoval={currentBgRemoval}
                activeMaskId={activeMaskLayer?.id ?? null}
                activeComponentId={activeMaskComponentId}
                showMaskOverlay={showMaskOverlay}
                busyAiKind={busyAiMaskKind}
                onSelectMask={handleSelectMask}
                onToggleMaskOverlay={() => setShowMaskOverlay((v) => !v)}
                onAttachManualComponent={handleAttachManualComponent}
                onAttachAiComponent={(layerId, kind, op) =>
                  void handleCreateAiMask(kind, layerId, op)
                }
                onChangeMasks={(nextMasks, label, coalesce = false) =>
                  handleChangeActiveRecipe(
                    { ...activeDevelopPhoto.developRecipe, masks: nextMasks },
                    label,
                    coalesce,
                  )
                }
                onToggleBackgroundRemoval={handleToggleBackgroundRemoval}
              />
            ) : undefined
          }
        />

        {/* Minimal Darkroom Bottom Bar & Larger Live-Graded Filmstrip (hidden in Tab Focus Mode) */}
        {!studioFocusMode && (
        <>
        <div className="h-10 px-4 bg-canvas border-t border-hairline/55 flex items-center justify-between gap-3 shrink-0">
          <SegmentedControl
            layoutId="develop-compare-mode"
            options={COMPARE_OPTIONS}
            value={compareMode}
            onChange={(mode) => {
              if (toolMode === 'crop') setToolMode('edit');
              setCompareMode(mode);
            }}
          />

          <div className="flex items-center gap-1 shrink-0">
            {retouchActive && (
              <DarkroomButton
                variant="ghost"
                size="xs"
                active={showSpotPins}
                onClick={() => setShowSpotPins((v) => !v)}
                title="Toggle Heal Spot Pins (H)"
              >
                Pins
              </DarkroomButton>
            )}
            {maskingActive && (
              <DarkroomButton
                variant="ghost"
                size="xs"
                active={showMaskOverlay}
                onClick={() => setShowMaskOverlay((v) => !v)}
                title="Toggle Mask Overlay (O)"
              >
                Overlay
              </DarkroomButton>
            )}
            <DarkroomButton
              variant="ghost"
              size="xs"
              active={showShadowClip || showHighlightClip}
              onClick={() => {
                const anyOn = showShadowClip || showHighlightClip;
                setShowShadowClip(!anyOn);
                setShowHighlightClip(!anyOn);
              }}
              title="Highlight & Shadow Clipping Warnings (J)"
              aria-label="Toggle Clipping Warnings"
            >
              Clipping
            </DarkroomButton>
            <div className="h-3.5 w-px bg-hairline/65 mx-0.5 shrink-0" />
            <DarkroomButton
              variant="ghost"
              size="xs"
              active={zoomFactor > 1.02}
              onClick={() => {
                setZoomFactor((z) => {
                  if (z <= 1.02) return Number(oneToOneZoom.toFixed(4));
                  if (z < oneToOneZoom * 1.6) return Number((oneToOneZoom * 2).toFixed(4));
                  return 1;
                });
              }}
              title="Cycle Zoom (Fit / 100% 1:1 / 200% • Z, Double-Click, or Ctrl+Scroll)"
              className="tabular-nums-mono min-w-[46px] justify-center"
            >
              {formatViewportZoomLabel(zoomFactor, oneToOneZoom)}
            </DarkroomButton>
            <DarkroomButton
              variant="ghost"
              size="xs"
              icon="fullscreen"
              onClick={toggleStudioFocusMode}
              title="Hide Panels / Focus View (Tab)"
              aria-label="Hide panels and focus on photo"
            />
          </div>
        </div>

        {/* Bottom Virtualized Live-Graded Filmstrip (hidden in Single Photo & Focus Mode) */}
        {!hideFilmstrip && (
        <VirtualHorizontalStrip
          items={developPhotos}
          activeIndex={activePhotoIndex}
          itemWidthPx={96}
          gapPx={12}
          getKey={(photo) => photo.id}
          className="kroma-filmstrip-scroll h-[144px] shrink-0 bg-surface-lowest/90 border-t border-hairline/60 px-4 py-2.5 flex items-center overflow-x-auto"
          renderItem={(photo) => {
            const isActive = photo.id === activeDevelopPhoto.id;
            const isChecked = selectedPhotoIds.has(photo.id);
            const photoFolderName = dirNameById.get(photo.directoryId);

            return (
              <div
                onClick={() => setActiveDevelopPhotoId(photo.id)}
                title={photoFolderName ? `${photo.fileName} · ${photoFolderName}` : photo.fileName}
                className={`relative h-[116px] w-[96px] shrink-0 rounded-xl overflow-hidden bg-surface-lowest cursor-pointer transition-all duration-150 group ${
                  isActive
                    ? 'ring-2 ring-amber-core shadow-[0_6px_18px_rgba(0,0,0,0.55)] opacity-100'
                    : isChecked
                      ? 'ring-1 ring-amber-core/55 opacity-95'
                      : 'ring-1 ring-white/8 hover:ring-white/25 opacity-80 hover:opacity-100'
                }`}
              >
                <GradedThumbnailImg
                  src={photo.imageUrl}
                  renderedSrc={photo.renderedThumbnailUrl}
                  alt={photo.fileName}
                  recipe={photo.developRecipe}
                  loading="lazy"
                  decoding="async"
                  className="w-full h-full object-cover"
                />

                {!isActive && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      togglePhotoSelection(photo.id);
                    }}
                    title={isChecked ? 'Selected for Sync' : 'Select photo to sync edits to'}
                    aria-label={isChecked ? `Deselect ${photo.fileName}` : `Select ${photo.fileName} for sync`}
                    className={`absolute top-1.5 left-1.5 z-20 w-4.5 h-4.5 rounded-md flex items-center justify-center border transition-opacity duration-150 cursor-pointer ${
                      isChecked
                        ? 'bg-amber-core border-amber-core text-amber-ink opacity-100'
                        : 'bg-black/55 border-white/30 opacity-0 group-hover:opacity-100 focus-visible:opacity-100'
                    }`}
                  >
                    {isChecked && (
                      <span className="material-symbols-outlined text-[12px] font-bold">check</span>
                    )}
                  </button>
                )}

                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/85 via-black/45 to-transparent pt-4 pb-1 px-1.5 tabular-nums-mono text-[9px] text-ink-primary/95 truncate text-center">
                  {photo.isSlateAnchor ? 'SLATE' : photo.fileName.replace(/\.[^.]+$/, '')}
                </div>
              </div>
            );
          }}
        />
        )}
        </>
        )}
      </section>

      {!studioFocusMode && (
        <>
          <PanelResizeHandle
            panelSide="right"
            width={rightPanel.width}
            minWidth={300}
            maxWidth={480}
            onChangeWidth={rightPanel.setWidth}
            onResetWidth={rightPanel.resetWidth}
            label="Resize Studio Inspector sidebar"
          />

          {/* Right Lightroom Studio Inspector Panel */}
          <DevelopInspectorPanel
            photo={activeDevelopPhoto}
            liveHistogram={liveHistogram}
            hasSlateAnchor={hasSlateAnchor}
            hasCopiedRecipe={copiedRecipe !== null}
            selectedCount={selectedPhotoIds.size}
            widthPx={rightPanel.width}
            showShadowClip={showShadowClip}
            showHighlightClip={showHighlightClip}
            eyedropperActive={eyedropperActive}
            naturalAspect={activePhotoAspect}
            toolMode={toolMode}
            onChangeToolMode={handleChangeToolMode}
            activeMaskId={activeMaskLayer?.id ?? null}
            activeMaskComponentId={activeMaskComponentId}
            creatingNewMask={creatingNewMask}
            brushOptions={brushOptions}
            colorPickActive={colorPickActive}
            busyAiMaskKind={busyAiMaskKind}
            busyBgRemoval={busyBgRemoval}
            onSelectMask={handleSelectMask}
            onRequestNewMask={() => setCreatingNewMask(true)}
            onCancelNewMask={() => setCreatingNewMask(false)}
            onChangeBrushOptions={setBrushOptions}
            onToggleColorPick={() => {
              setEyedropperActive(false);
              setColorPickActive((v) => !v);
            }}
            onCreateAiMask={(kind, attachId, op) => void handleCreateAiMask(kind, attachId, op)}
            onRemoveBackground={() => void handleRemoveBackground()}
            onChangeBackgroundRemoval={handleChangeBackgroundRemoval}
            healTool={healTool}
            selectedSpotId={selectedSpotId}
            showSpotPins={showSpotPins}
            detectingFace={detectingFace}
            faceParseMap={activeFaceParseMap}
            parsingFace={parsingFace}
            faceParserReady={faceParserReady}
            lamaReady={lamaReady}
            downloadingOptionalModelId={downloadingOptionalModelId}
            modelDownloadProgress={modelDownloadProgress}
            pendingRemoveStrokesCount={pendingRemoveStrokes.length}
            busyInpainting={busyInpainting}
            isDesktopRuntime={isDesktopRuntime}
            onChangeHealTool={setHealTool}
            onSelectSpot={setSelectedSpotId}
            onToggleShowSpotPins={() => setShowSpotPins((v) => !v)}
            onDeleteSpot={handleDeleteSpot}
            onClearAllSpots={handleClearAllSpots}
            onApplyInpaint={() => void handleApplyInpaint()}
            onClearPendingRemoveStrokes={() => setPendingRemoveStrokes([])}
            onDeleteInpaintPatch={handleDeleteInpaintPatch}
            onClearAllInpaintPatches={handleClearAllInpaintPatches}
            onDownloadOptionalModel={(modelId) => void downloadOptionalModel(modelId)}
            onCancelOptionalModelDownload={() => void cancelOptionalModelDownload()}
            onDetectFace={() => void handleDetectFace(false)}
            onToggleShadowClip={() => setShowShadowClip((v) => !v)}
            onToggleHighlightClip={() => setShowHighlightClip((v) => !v)}
            onToggleEyedropper={() => {
              if (toolMode === 'crop') setToolMode('edit');
              setColorPickActive(false);
              setEyedropperActive((v) => !v);
            }}
            onChangeRecipe={handleChangeActiveRecipe}
            onApplyPreset={handleApplyPreset}
            onApplyRelativeNudge={applyRelativeNudgeToClusterSelection}
            onSyncSlateWb={syncSlateWhiteBalanceForActiveCluster}
            onOpenSyncModal={handleOpenSyncModal}
            isSinglePhoto={isSinglePhoto}
            onOpenExport={() => setActiveWorkspace('export')}
            onCopyRecipe={copyActiveRecipe}
            onPasteRecipe={pasteRecipeToClusterSelection}
          />
        </>
      )}

      {/* Apply Adjustments Modal (Ctrl+Shift+S) */}
      <SyncSettingsModal
        open={syncModalOpen}
        onClose={() => setSyncModalOpen(false)}
        sourcePhoto={activeDevelopPhoto}
        scopeLabel={activeDevelopScopeLabel}
        scopePhotos={developPhotos}
        allProjectPhotos={allProjectPhotos}
        selectedPhotoIds={selectedPhotoIds}
        onConfirmSync={(mask, scope) => syncRecipeFromPhoto(activeDevelopPhoto.id, mask, scope)}
      />

      <SubjectHoverPreviewCard
        cluster={hoveredSubject?.cluster ?? null}
        anchorRect={hoveredSubject?.rect ?? null}
      />

      <PhotoNameHoverPreview
        photo={hoveredFolderPhoto?.photo ?? null}
        anchorRect={hoveredFolderPhoto?.rect ?? null}
      />
    </div>
  );
});
