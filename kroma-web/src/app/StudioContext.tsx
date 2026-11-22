import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import {
  loadStudioPreferencesFromStorage,
  saveStudioPreferencesToStorage,
} from '../adapters/storage/preferencesStorage';
import {
  loadRecentProjectsFromStorage,
  saveRecentProjectsToStorage,
} from '../adapters/storage/recentProjectsStorage';
import type {
  AiModelRole,
  BatchExportRequest,
  BatchExportResult,
  ClearCacheResult,
  ClearCacheTarget,
  ExportSizeEstimateRequest,
  LiveRgbHistogram,
  MeasuredExportSize,
  ModelDownloadProgress,
  ModelInspectionResult,
  ModelManifestStatus,
  RuntimeInfo,
  ScanProgressUpdate,
  StorageCacheSummary,
  StudioGateway,
  StudioSessionSnapshot,
  ThumbnailRenderProgress,
} from '../application/ports/studioGateway';
import {
  filterClusterPhotos,
  type SmartBinFilter,
} from '../application/useCases/filterClusterPhotos';
import {
  checkGithubForAppUpdate,
  KROMA_FALLBACK_VERSION,
  type AppUpdateCheckResult,
} from '../domain/appUpdates';
import {
  applyRelativeDevelopDelta,
  syncDevelopRecipe,
  type DevelopRecipe,
  type DevelopSyncMask,
  type RelativeDevelopDelta,
} from '../domain/develop';
import type {
  DevelopWorkingBuffer,
  WorkingBufferPrecision,
} from '../domain/developWorkingBuffer';
import {
  EMPTY_DEVELOP_HISTORY,
  recordDevelopHistory,
  redoDevelopHistory,
  undoDevelopHistory,
  type DevelopHistoryStack,
  type PhotoRecipeDiff,
} from '../domain/developHistory';
import {
  countClusterReviewWarnings,
  needsVisualMatchReview,
  type SubjectCluster,
} from '../domain/identity';
import type { AiMaskKind, GeneratedAiMaskResult } from '../domain/masking';
import type { OutputNamingRule } from '../domain/organization';
import type { PhotoAsset } from '../domain/photo';
import {
  mergeSlateOcrResults,
  type ClusterSlateOcrResult,
  type PhotoOcrResult,
  type SlateOcrScanOptions,
  type SlateRenameApplyItem,
} from '../domain/slateOcr';
import {
  collectDescendantFolderIds,
  matchesVirtualFolderFilter,
  type VirtualFolderFilter,
} from '../domain/virtualFolder';
import { formatShortcutCombo } from '../domain/platform';
import {
  DEFAULT_STUDIO_PREFERENCES,
  normalizeStudioPreferences,
  type StudioPreferences,
  type ThumbnailPreviewMode,
} from '../domain/preferences';
import {
  removeRecentProject,
  togglePinRecentProject,
  upsertRecentProject,
  type RecentProjectEntry,
} from '../domain/recentProjects';

export type WorkspaceView =
  | 'home'
  | 'culling'
  | 'slate_ocr'
  | 'develop'
  | 'review_queue'
  | 'export';
export type StartupBootPhase =
  | 'checking_models'
  | 'downloading_models'
  | 'models_error'
  | 'checking_updates'
  | 'update_prompt'
  | 'ready';
export type GridDisplayMode = 'grid' | 'split' | 'loupe';
export type AnchorSortMode = 'capture_time' | 'confidence' | 'exposures';
export type SyncTargetScope = 'selected_in_cluster' | 'entire_cluster' | 'entire_shoot';
export type DevelopSourceMode = 'folders' | 'subjects';

export interface ExportJobState {
  readonly isExporting: boolean;
  readonly completed: number;
  readonly total: number;
  readonly percent: number;
  readonly message: string;
  readonly completedCount: number | null;
  readonly failedCount: number;
  readonly outputRoot: string | null;
  readonly error: string | null;
}

export interface AmbiguousReviewItem {
  readonly photo: PhotoAsset;
  readonly candidateCluster: SubjectCluster;
  readonly reason: string;
  readonly source: 'backend_queue' | 'cluster_low_confidence';
}

export interface DevelopViewportStatus {
  readonly state: 'decoding' | 'ready';
  readonly precision: WorkingBufferPrecision;
  readonly isRawDemosaic: boolean;
  readonly backend: 'webgpu' | 'webgl2';
  readonly width?: number;
  readonly height?: number;
}

interface StudioContextValue {
  // Queries
  isDesktopRuntime: boolean;
  isBootReady: boolean;
  bootPhase: StartupBootPhase;
  modelManifestStatus: ModelManifestStatus | null;
  modelDownloadProgress: ModelDownloadProgress | null;
  modelDownloadError: string | null;
  startupUpdate: AppUpdateCheckResult | null;
  isScanningFolder: boolean;
  isScanningOcr: boolean;
  slateOcrResults: readonly ClusterSlateOcrResult[];
  scanProgress: ScanProgressUpdate | null;
  thumbnailRenderProgress: ThumbnailRenderProgress | null;
  developViewportStatus: DevelopViewportStatus | null;
  exportJob: ExportJobState | null;
  runtimeInfo: RuntimeInfo | null;
  session: StudioSessionSnapshot | null;
  preferences: StudioPreferences;
  recentProjects: readonly RecentProjectEntry[];
  activeWorkspace: WorkspaceView;
  settingsOpen: boolean;
  activeCluster: SubjectCluster | null;
  developSourceMode: DevelopSourceMode;
  activeDevelopFolderId: string | null;
  developPhotos: readonly PhotoAsset[];
  activeDevelopScopeLabel: string;
  activeDevelopPhoto: PhotoAsset | null;
  liveHistogram: LiveRgbHistogram | null;
  visibleClusterPhotos: readonly PhotoAsset[];
  ambiguousReviewQueue: readonly AmbiguousReviewItem[];
  selectedPhotoIds: ReadonlySet<string>;
  copiedRecipe: DevelopRecipe | null;
  activeDirectoryId: string | null;
  activeVirtualFolderId: VirtualFolderFilter;
  activeVirtualFolderAllowedIds: ReadonlySet<string> | null;
  folderCreateNonce: number;
  smartBin: SmartBinFilter;
  searchQuery: string;
  sfaceThreshold: number;
  lastClusteredThreshold: number;
  reviewWarningThreshold: number;
  approvedReviewPhotoIds: ReadonlySet<string>;
  useTemporalAssist: boolean;
  lastClusteredTemporalAssist: boolean;
  manualMatchSplitOpen: boolean;
  studioFocusMode: boolean;
  displayMode: GridDisplayMode;
  anchorSortMode: AnchorSortMode;
  showLandmarksOverlay: boolean;
  outputRule: OutputNamingRule;
  statusBanner: string | null;
  canUndo: boolean;
  canRedo: boolean;
  undoLabel: string | null;
  redoLabel: string | null;
  canUndoDevelop: boolean;
  canRedoDevelop: boolean;
  undoDevelopLabel: string | null;
  redoDevelopLabel: string | null;
  isWindowMaximized: boolean;

  // Commands
  retryModelDownload: () => Promise<void>;
  continueWithoutModels: () => Promise<void>;
  dismissStartupUpdate: () => void;
  setActiveWorkspace: (view: WorkspaceView) => void;
  openSettings: () => void;
  closeSettings: () => void;
  toggleSettings: () => void;
  updatePreferences: (patch: Partial<StudioPreferences>) => void;
  savePreferences: (next: StudioPreferences) => Promise<boolean>;
  resetPreferences: () => void;
  pickOnnxModelFile: () => Promise<string | null>;
  inspectOnnxModel: (modelPath: string, role: AiModelRole) => Promise<ModelInspectionResult | null>;
  applyCustomAiModel: (role: AiModelRole, modelPath: string | null) => Promise<boolean>;
  setDevelopSourceMode: (mode: DevelopSourceMode) => void;
  selectDevelopFolder: (folderId: string | null) => void;
  openShootFolder: () => Promise<void>;
  openRecentShootFolder: (folderPath: string) => Promise<void>;
  removeRecentProjectEntry: (folderPath: string) => void;
  clearRecentProjects: () => void;
  togglePinRecentProjectEntry: (folderPath: string) => void;
  getStorageCacheSummary: () => Promise<StorageCacheSummary | null>;
  pickCacheDirectory: () => Promise<string | null>;
  clearStorageCache: (target: ClearCacheTarget) => Promise<ClearCacheResult | null>;
  runFaceOrganizer: () => Promise<void>;
  scanProjectSlatesOcr: (options?: SlateOcrScanOptions) => Promise<void>;
  cancelSlateOcrScan: () => Promise<void>;
  scanSinglePhotoOcr: (photoId: string) => Promise<PhotoOcrResult | null>;
  applySlateOcrRenames: (items: readonly SlateRenameApplyItem[]) => Promise<boolean>;
  openPhotoInDevelop: (photoId: string, clusterId?: string) => void;
  selectCluster: (clusterId: string) => void;
  selectNextCluster: () => void;
  selectNextSingletonCluster: () => void;
  setManualMatchSplitOpen: (open: boolean) => void;
  setStudioFocusMode: (enabled: boolean) => void;
  toggleStudioFocusMode: () => void;
  setDevelopViewportStatus: (status: DevelopViewportStatus | null) => void;
  setActiveDevelopPhotoId: (photoId: string) => void;
  toggleDirectoryFilter: (directoryId: string) => void;
  setActiveVirtualFolderId: (filter: VirtualFolderFilter) => void;
  requestNewVirtualFolder: () => void;
  createVirtualFolder: (name: string, parentId?: string | null) => Promise<boolean>;
  renameVirtualFolder: (folderId: string, nextName: string) => Promise<boolean>;
  moveVirtualFolder: (folderId: string, nextParentId: string | null) => Promise<boolean>;
  deleteVirtualFolder: (folderId: string) => Promise<boolean>;
  assignPhotosToVirtualFolder: (
    photoIds: readonly string[],
    folderId: string | null,
  ) => Promise<boolean>;
  assignClustersToVirtualFolder: (
    clusterIds: readonly string[],
    folderId: string | null,
  ) => Promise<boolean>;
  setSmartBin: (bin: SmartBinFilter) => void;
  setSearchQuery: (query: string) => void;
  setSfaceThreshold: (threshold: number) => void;
  setReviewWarningThreshold: (threshold: number) => void;
  setUseTemporalAssist: (enabled: boolean) => void;
  setDisplayMode: (mode: GridDisplayMode) => void;
  setAnchorSortMode: (mode: AnchorSortMode) => void;
  toggleLandmarksOverlay: () => void;
  togglePhotoSelection: (photoId: string, options?: { readonly shiftKey?: boolean }) => void;
  selectAllVisiblePhotos: () => void;
  peekDevelopWorkingBuffer: (
    photoId: string,
    maxDim?: number,
    precision?: WorkingBufferPrecision,
  ) => DevelopWorkingBuffer | null;
  loadDevelopWorkingBuffer: (
    photoId: string,
    maxDim?: number,
    precision?: WorkingBufferPrecision,
  ) => Promise<DevelopWorkingBuffer | null>;
  generateAiMask: (photoId: string, kind: AiMaskKind) => Promise<GeneratedAiMaskResult | null>;
  updatePhotoRecipe: (
    photoId: string,
    nextRecipe: DevelopRecipe,
    label?: string,
    coalesce?: boolean,
  ) => void;
  syncRecipeFromPhoto: (
    sourcePhotoId: string,
    mask: DevelopSyncMask,
    scope: SyncTargetScope,
  ) => void;
  applyRelativeNudgeToClusterSelection: (delta: RelativeDevelopDelta, label: string) => void;
  copyActiveRecipe: () => void;
  pasteRecipeToClusterSelection: () => void;
  mergeActiveClusterInto: (targetClusterId: string) => Promise<void>;
  mergeClustersById: (sourceClusterId: string, targetClusterId: string) => Promise<void>;
  movePhotosToCluster: (photoIds: readonly string[], targetClusterId: string) => Promise<void>;
  splitSelectedPhotos: () => Promise<void>;
  splitPhotoIdsToNewCluster: (photoIds: readonly string[]) => Promise<void>;
  autoLevelActiveCluster: () => Promise<void>;
  resolveAmbiguousReview: (photoId: string, decision: 'approve' | 'split') => void;
  batchResolveAmbiguousReview: (
    photoIds: readonly string[],
    decision: 'approve' | 'split_together',
  ) => Promise<void>;
  renameActiveSlate: (clusterId: string, nextName: string) => Promise<void>;
  setExcludeSlateFromClientPack: (exclude: boolean) => void;
  setOutputPattern: (pattern: string) => void;
  syncSlateWhiteBalanceForActiveCluster: () => void;
  undoOrganizationAction: () => Promise<void>;
  redoOrganizationAction: () => Promise<void>;
  undoDevelopAction: () => void;
  redoDevelopAction: () => void;
  pickExportDirectory: () => Promise<string | null>;
  estimateExportSize: (request: ExportSizeEstimateRequest) => Promise<MeasuredExportSize | null>;
  runBatchExport: (request: BatchExportRequest) => Promise<BatchExportResult | null>;
  minimizeWindow: () => Promise<void>;
  toggleMaximizeWindow: () => Promise<void>;
  exitApp: () => Promise<void>;
  notifyAction: (message: string) => void;
}

const StudioContext = createContext<StudioContextValue | null>(null);

interface StudioProviderProps {
  gateway: StudioGateway;
  children: ReactNode;
}

export function StudioProvider({ gateway, children }: StudioProviderProps) {
  const [runtimeInfo, setRuntimeInfo] = useState<RuntimeInfo | null>(null);
  const [session, setSession] = useState<StudioSessionSnapshot | null>(null);
  const [preferences, setPreferences] = useState<StudioPreferences>(() =>
    loadStudioPreferencesFromStorage(),
  );
  const [recentProjects, setRecentProjects] = useState<readonly RecentProjectEntry[]>(() =>
    loadRecentProjectsFromStorage(),
  );
  const [isBootReady, setIsBootReady] = useState(false);
  const [bootPhase, setBootPhase] = useState<StartupBootPhase>('checking_models');
  const [modelManifestStatus, setModelManifestStatus] = useState<ModelManifestStatus | null>(null);
  const [modelDownloadProgress, setModelDownloadProgress] = useState<ModelDownloadProgress | null>(
    null,
  );
  const [modelDownloadError, setModelDownloadError] = useState<string | null>(null);
  const [startupUpdate, setStartupUpdate] = useState<AppUpdateCheckResult | null>(null);
  const mountedRef = useRef(true);
  const initialSessionPromiseRef = useRef<Promise<void>>(Promise.resolve());
  const detectedAppVersionRef = useRef<string>(KROMA_FALLBACK_VERSION);
  const [isScanningFolder, setIsScanningFolder] = useState(false);
  const [isScanningOcr, setIsScanningOcr] = useState(false);
  const [slateOcrResults, setSlateOcrResults] = useState<readonly ClusterSlateOcrResult[]>([]);
  const [scanProgress, setScanProgress] = useState<ScanProgressUpdate | null>(null);
  const [thumbnailRenderProgress, setThumbnailRenderProgress] =
    useState<ThumbnailRenderProgress | null>(null);
  const [developViewportStatus, setDevelopViewportStatus] =
    useState<DevelopViewportStatus | null>(null);
  const [exportJob, setExportJob] = useState<ExportJobState | null>(null);
  const [activeWorkspace, setActiveWorkspaceState] = useState<WorkspaceView>('home');
  const [settingsOpen, setSettingsOpen] = useState(false);

  const openSettings = useCallback(() => {
    setSettingsOpen(true);
  }, []);

  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
  }, []);

  const toggleSettings = useCallback(() => {
    setSettingsOpen((prev) => !prev);
  }, []);

  const updatePreferences = useCallback((patch: Partial<StudioPreferences>) => {
    setPreferences((prev) => {
      const next = normalizeStudioPreferences({ ...prev, ...patch });
      saveStudioPreferencesToStorage(next);
      return next;
    });
  }, []);

  const setActiveWorkspace = useCallback((next: WorkspaceView) => {
    setSettingsOpen(false);
    if (next !== 'develop') {
      setStudioFocusMode(false);
    }
    setActiveWorkspaceState((prev) => {
      if ((prev === 'home') !== (next === 'home')) {
        setSearchQuery('');
      }
      return next;
    });
  }, []);

  const [activeClusterId, setActiveClusterId] = useState<string>('');
  const [developSourceMode, setDevelopSourceModeState] = useState<DevelopSourceMode>('folders');
  const [activeDevelopFolderId, setActiveDevelopFolderId] = useState<string | null>(null);
  const [activeDevelopPhotoId, setActiveDevelopPhotoId] = useState<string | null>(null);
  const [liveHistogram, setLiveHistogram] = useState<LiveRgbHistogram | null>(null);
  const [selectedPhotoIds, setSelectedPhotoIds] = useState<Set<string>>(new Set());
  const [copiedRecipe, setCopiedRecipe] = useState<DevelopRecipe | null>(null);
  const [resolvedAmbiguousIds, setResolvedAmbiguousIds] = useState<Set<string>>(new Set());
  const [activeDirectoryId, setActiveDirectoryId] = useState<string | null>(null);
  const [activeVirtualFolderId, setActiveVirtualFolderIdState] =
    useState<VirtualFolderFilter>('all');
  const [folderCreateNonce, setFolderCreateNonce] = useState(0);
  const [smartBin, setSmartBin] = useState<SmartBinFilter>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [lastClusteredThreshold, setLastClusteredThreshold] = useState(
    () => preferences.sfaceThreshold,
  );
  const [lastClusteredTemporalAssist, setLastClusteredTemporalAssist] = useState(
    () => preferences.useTemporalAssist,
  );
  const [manualMatchSplitOpen, setManualMatchSplitOpen] = useState(false);
  const [studioFocusMode, setStudioFocusMode] = useState(false);
  const toggleStudioFocusMode = useCallback(() => setStudioFocusMode((prev) => !prev), []);
  const [displayMode, setDisplayMode] = useState<GridDisplayMode>('grid');
  const [anchorSortMode, setAnchorSortMode] = useState<AnchorSortMode>(
    () => preferences.defaultSubjectSortMode,
  );
  const [statusBanner, setStatusBanner] = useState<string | null>(null);
  const [developHistory, setDevelopHistory] = useState<DevelopHistoryStack>(EMPTY_DEVELOP_HISTORY);
  const [isWindowMaximized, setIsWindowMaximized] = useState(true);

  const sfaceThreshold = preferences.sfaceThreshold;
  const reviewWarningThreshold = preferences.reviewWarningThreshold;
  const useTemporalAssist = preferences.useTemporalAssist;
  const showLandmarksOverlay = preferences.showLandmarksOverlay;
  const outputRule = useMemo<OutputNamingRule>(
    () => ({
      pattern: preferences.defaultNamingPattern,
      excludeSlateFromClientPack: preferences.excludeSlateFromClientPack,
    }),
    [preferences.defaultNamingPattern, preferences.excludeSlateFromClientPack],
  );

  const setSfaceThreshold = useCallback(
    (threshold: number) => updatePreferences({ sfaceThreshold: threshold }),
    [updatePreferences],
  );
  const setReviewWarningThreshold = useCallback(
    (threshold: number) => updatePreferences({ reviewWarningThreshold: threshold }),
    [updatePreferences],
  );
  const setUseTemporalAssist = useCallback(
    (enabled: boolean) => updatePreferences({ useTemporalAssist: enabled }),
    [updatePreferences],
  );

  const pendingRecipeSavesRef = useRef<
    Map<string, { timerId: number; recipe: DevelopRecipe }>
  >(new Map());
  const activeFolderPathRef = useRef<string | null>(null);
  const lastSelectedPhotoIdRef = useRef<string | null>(null);
  const sessionRef = useRef<StudioSessionSnapshot | null>(null);
  const developHistoryRef = useRef<DevelopHistoryStack>(EMPTY_DEVELOP_HISTORY);
  const thumbnailPreviewModeRef = useRef<ThumbnailPreviewMode>(preferences.thumbnailPreviewMode);
  const isDesktopRuntime = gateway.isDesktopRuntime();

  sessionRef.current = session;
  developHistoryRef.current = developHistory;
  thumbnailPreviewModeRef.current = preferences.thumbnailPreviewMode;

  const request32BitThumbnails = useCallback(
    (mode: ThumbnailPreviewMode, photoIds?: readonly string[]) => {
      if (!isDesktopRuntime || !gateway.render32BitThumbnails) return;
      void gateway.render32BitThumbnails(mode, photoIds).catch(() => { });
    },
    [isDesktopRuntime, gateway],
  );

  const findPhotoRecipe = useCallback((photoId: string): DevelopRecipe | null => {
    const current = sessionRef.current;
    if (!current) return null;
    for (const cluster of current.clusters) {
      const found = cluster.photos.find((p) => p.id === photoId);
      if (found) return found.developRecipe;
    }
    return null;
  }, []);

  const applyRecipeChangesToSession = useCallback(
    (changes: readonly PhotoRecipeDiff[], target: 'before' | 'after') => {
      if (changes.length === 0) return;
      const patchMap = new Map<string, DevelopRecipe>(
        changes.map((c) => [c.photoId, target === 'before' ? c.before : c.after]),
      );
      setSession((prev) => {
        if (!prev) return prev;
        let touched = false;
        const nextClusters = prev.clusters.map((cluster) => {
          let clusterTouched = false;
          const nextPhotos = cluster.photos.map((photo) => {
            const replacement = patchMap.get(photo.id);
            if (!replacement) return photo;
            clusterTouched = true;
            touched = true;
            return { ...photo, developRecipe: replacement };
          });
          return clusterTouched ? { ...cluster, photos: nextPhotos } : cluster;
        });
        return touched ? { ...prev, clusters: nextClusters } : prev;
      });

      const batchItems: { photoId: string; recipe: DevelopRecipe }[] = [];
      for (const change of changes) {
        const pending = pendingRecipeSavesRef.current.get(change.photoId);
        if (pending) {
          window.clearTimeout(pending.timerId);
          pendingRecipeSavesRef.current.delete(change.photoId);
        }
        const nextRecipe = target === 'before' ? change.before : change.after;
        batchItems.push({ photoId: change.photoId, recipe: nextRecipe });
      }

      const changedIds = batchItems.map((item) => item.photoId);
      if (gateway.savePhotoDevelopRecipesBatch) {
        void gateway
          .savePhotoDevelopRecipesBatch(batchItems, activeDevelopPhotoId)
          .then((hist) => {
            if (hist) setLiveHistogram(hist);
            if (thumbnailPreviewModeRef.current !== 'fast') {
              request32BitThumbnails(thumbnailPreviewModeRef.current, changedIds);
            }
          })
          .catch((err) => {
            setStatusBanner(`Undo/redo save error: ${String(err)}`);
          });
      } else {
        for (const item of batchItems) {
          void gateway
            .savePhotoDevelopRecipe(item.photoId, item.recipe)
            .then((hist) => {
              if (hist && item.photoId === activeDevelopPhotoId) {
                setLiveHistogram(hist);
              }
              if (thumbnailPreviewModeRef.current !== 'fast') {
                request32BitThumbnails(thumbnailPreviewModeRef.current, [item.photoId]);
              }
            })
            .catch(() => { });
        }
      }
    },
    [gateway, activeDevelopPhotoId, request32BitThumbnails],
  );

  useEffect(() => {
    activeFolderPathRef.current = session?.folderPath ?? null;
  }, [session?.folderPath]);

  // Keep Recent Projects catalog badges (Organized / subject count) synced with live organization edits
  useEffect(() => {
    if (!session || !session.folderPath) return;
    const folderPath = session.folderPath;
    const subjectCount = session.isClustered
      ? session.clusters.filter((c) => !c.anchor.id.startsWith('_')).length
      : 0;
    const isClustered = session.isClustered;
    const totalPhotos = session.telemetry.totalPhotos;

    setRecentProjects((prev) => {
      const existing = prev.find((p) => p.folderPath === folderPath);
      if (
        !existing ||
        (existing.isClustered === isClustered &&
          existing.clusterCount === subjectCount &&
          existing.photoCount === totalPhotos)
      ) {
        return prev;
      }
      const next = prev.map((p) =>
        p.id === existing.id
          ? { ...p, isClustered, clusterCount: subjectCount, photoCount: totalPhotos }
          : p,
      );
      saveRecentProjectsToStorage(next);
      return next;
    });
  }, [session]);

  const notifyAction = useCallback((message: string) => {
    setStatusBanner(message);
  }, []);

  const savePreferences = useCallback(
    async (rawNext: StudioPreferences): Promise<boolean> => {
      const next = normalizeStudioPreferences(rawNext);
      const modelsChanged =
        next.customDetectorModelPath !== preferences.customDetectorModelPath ||
        next.customRecognizerModelPath !== preferences.customRecognizerModelPath ||
        next.customSegmenterModelPath !== preferences.customSegmenterModelPath;
      const gpuChanged = next.enableGpuAcceleration !== preferences.enableGpuAcceleration;
      const cacheDirChanged = next.customCacheDir !== preferences.customCacheDir;
      const thumbnailModeChanged =
        next.thumbnailPreviewMode !== preferences.thumbnailPreviewMode;

      if (modelsChanged && isDesktopRuntime && gateway.setCustomAiModels) {
        try {
          const updatedInfo = await gateway.setCustomAiModels(
            next.customDetectorModelPath,
            next.customRecognizerModelPath,
            next.customSegmenterModelPath,
          );
          if (updatedInfo) {
            setRuntimeInfo(updatedInfo);
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          notifyAction(msg);
          return false;
        }
      }

      if (gpuChanged && isDesktopRuntime && gateway.setGpuAcceleration) {
        try {
          const updatedInfo = await gateway.setGpuAcceleration(next.enableGpuAcceleration);
          if (updatedInfo) {
            setRuntimeInfo(updatedInfo);
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          notifyAction(msg);
          return false;
        }
      }

      if (cacheDirChanged && isDesktopRuntime && gateway.setStorageCacheDir) {
        try {
          await gateway.setStorageCacheDir(next.customCacheDir);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          notifyAction(msg);
          return false;
        }
      }

      if (next.defaultSubjectSortMode !== preferences.defaultSubjectSortMode) {
        setAnchorSortMode(next.defaultSubjectSortMode);
      }

      setPreferences(next);
      saveStudioPreferencesToStorage(next);
      if (sessionRef.current && (thumbnailModeChanged || cacheDirChanged)) {
        if (next.thumbnailPreviewMode === 'fast') {
          setThumbnailRenderProgress(null);
        }
        request32BitThumbnails(next.thumbnailPreviewMode);
      }
      notifyAction('Settings applied');
      return true;
    },
    [
      preferences.customDetectorModelPath,
      preferences.customRecognizerModelPath,
      preferences.customSegmenterModelPath,
      preferences.enableGpuAcceleration,
      preferences.customCacheDir,
      preferences.thumbnailPreviewMode,
      preferences.defaultSubjectSortMode,
      isDesktopRuntime,
      gateway,
      request32BitThumbnails,
      notifyAction,
    ],
  );

  const resetPreferences = useCallback(() => {
    const defaults = { ...DEFAULT_STUDIO_PREFERENCES };
    const modeChanged = preferences.thumbnailPreviewMode !== defaults.thumbnailPreviewMode;
    setPreferences(defaults);
    saveStudioPreferencesToStorage(defaults);
    if (isDesktopRuntime && gateway.setCustomAiModels) {
      void gateway
        .setCustomAiModels(null, null, null)
        .then((info) => {
          if (info) setRuntimeInfo(info);
        })
        .catch(() => { });
    }
    if (sessionRef.current && modeChanged) {
      request32BitThumbnails(defaults.thumbnailPreviewMode);
    }
    notifyAction('Settings restored to factory defaults');
  }, [
    isDesktopRuntime,
    gateway,
    preferences.thumbnailPreviewMode,
    request32BitThumbnails,
    notifyAction,
  ]);

  const pickOnnxModelFile = useCallback(async (): Promise<string | null> => {
    if (!isDesktopRuntime || !gateway.pickOnnxModelFile) {
      notifyAction('Selecting custom .onnx models requires Kroma Desktop (`npm run tauri:dev`)');
      return null;
    }
    try {
      return await gateway.pickOnnxModelFile();
    } catch (err) {
      notifyAction(`Model picker error: ${String(err)}`);
      return null;
    }
  }, [isDesktopRuntime, gateway, notifyAction]);

  const inspectOnnxModel = useCallback(
    async (modelPath: string, role: AiModelRole): Promise<ModelInspectionResult | null> => {
      if (!isDesktopRuntime || !gateway.inspectOnnxModel) {
        return null;
      }
      try {
        return await gateway.inspectOnnxModel(modelPath, role);
      } catch (err) {
        notifyAction(`Model inspection failed: ${String(err)}`);
        return null;
      }
    },
    [isDesktopRuntime, gateway, notifyAction],
  );

  const applyCustomAiModel = useCallback(
    async (role: AiModelRole, modelPath: string | null): Promise<boolean> => {
      const nextDetector =
        role === 'detector' ? modelPath : preferences.customDetectorModelPath;
      const nextRecognizer =
        role === 'recognizer' ? modelPath : preferences.customRecognizerModelPath;
      const nextSegmenter =
        role === 'segmenter' ? modelPath : preferences.customSegmenterModelPath;

      if (isDesktopRuntime && gateway.setCustomAiModels) {
        try {
          const updatedInfo = await gateway.setCustomAiModels(
            nextDetector,
            nextRecognizer,
            nextSegmenter,
          );
          if (updatedInfo) {
            setRuntimeInfo(updatedInfo);
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          notifyAction(msg);
          return false;
        }
      }

      updatePreferences(
        role === 'detector'
          ? { customDetectorModelPath: modelPath }
          : role === 'recognizer'
            ? { customRecognizerModelPath: modelPath }
            : { customSegmenterModelPath: modelPath },
      );

      const roleLabel =
        role === 'detector'
          ? 'face detector'
          : role === 'recognizer'
            ? 'face recognition'
            : 'subject masking';
      if (modelPath) {
        const fileName = modelPath.split(/[\\/]/).pop() ?? modelPath;
        notifyAction(`Activated custom ${roleLabel} model: ${fileName}`);
      } else {
        notifyAction(`Restored default ${roleLabel} model`);
      }
      return true;
    },
    [
      isDesktopRuntime,
      gateway,
      preferences.customDetectorModelPath,
      preferences.customRecognizerModelPath,
      preferences.customSegmenterModelPath,
      updatePreferences,
      notifyAction,
    ],
  );

  const recordRecentFromSnapshot = useCallback((loaded: StudioSessionSnapshot) => {
    if (!loaded.folderPath) return;
    const coverThumbnails: string[] = [];
    const primaryClusters = loaded.clusters.filter((c) => !c.anchor.id.startsWith('_'));
    const sourceClusters = primaryClusters.length > 0 ? primaryClusters : loaded.clusters;

    for (const cluster of sourceClusters) {
      if (coverThumbnails.length >= 3) break;
      const candidate = cluster.anchor.thumbnailUrl || cluster.photos[0]?.imageUrl;
      if (candidate && !coverThumbnails.includes(candidate)) {
        coverThumbnails.push(candidate);
      }
    }
    if (coverThumbnails.length < 3) {
      for (const photo of sourceClusters[0]?.photos ?? []) {
        if (coverThumbnails.length >= 3) break;
        if (photo.imageUrl && !coverThumbnails.includes(photo.imageUrl)) {
          coverThumbnails.push(photo.imageUrl);
        }
      }
    }

    setRecentProjects((prev) => {
      const next = upsertRecentProject(prev, {
        projectName: loaded.telemetry.projectName,
        folderPath: loaded.folderPath!,
        photoCount: loaded.telemetry.totalPhotos,
        clusterCount: loaded.isClustered ? primaryClusters.length : 0,
        isClustered: loaded.isClustered,
        coverThumbnails,
        lastOpenedAtIso: new Date().toISOString(),
      });
      saveRecentProjectsToStorage(next);
      return next;
    });
  }, []);

  const removeRecentProjectEntry = useCallback(
    (folderPath: string) => {
      setRecentProjects((prev) => {
        const target = prev.find((p) => p.folderPath === folderPath);
        const next = removeRecentProject(prev, folderPath);
        saveRecentProjectsToStorage(next);
        if (target) {
          notifyAction(`Removed "${target.projectName}" from recent projects`);
        }
        return next;
      });
    },
    [notifyAction],
  );

  const togglePinRecentProjectEntry = useCallback(
    (folderPath: string) => {
      setRecentProjects((prev) => {
        const target = prev.find((p) => p.folderPath === folderPath);
        const next = togglePinRecentProject(prev, folderPath);
        saveRecentProjectsToStorage(next);
        if (target) {
          notifyAction(
            target.isPinned
              ? `Unpinned "${target.projectName}"`
              : `Pinned "${target.projectName}" to Catalog`,
          );
        }
        return next;
      });
    },
    [notifyAction],
  );

  const clearRecentProjects = useCallback(() => {
    setRecentProjects([]);
    saveRecentProjectsToStorage([]);
    notifyAction('Cleared recent projects list');
  }, [notifyAction]);

  const getStorageCacheSummary = useCallback(async (): Promise<StorageCacheSummary | null> => {
    if (!gateway.getStorageCacheSummary) return null;
    try {
      return await gateway.getStorageCacheSummary();
    } catch {
      return null;
    }
  }, [gateway]);

  const pickCacheDirectory = useCallback(async (): Promise<string | null> => {
    if (!isDesktopRuntime || !gateway.pickCacheDirectory) {
      notifyAction('Choosing a custom cache folder requires Kroma Desktop (`npm run tauri:dev`)');
      return null;
    }
    try {
      return await gateway.pickCacheDirectory();
    } catch (err) {
      notifyAction(`Cache folder picker error: ${String(err)}`);
      return null;
    }
  }, [isDesktopRuntime, gateway, notifyAction]);

  const clearStorageCache = useCallback(
    async (target: ClearCacheTarget): Promise<ClearCacheResult | null> => {
      if (!gateway.clearStorageCache) return null;
      try {
        const result = await gateway.clearStorageCache(target);
        if (result) {
          if (target === 'previews' || target === 'all') {
            setThumbnailRenderProgress(null);
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                clusters: prev.clusters.map((cluster) => ({
                  ...cluster,
                  photos: cluster.photos.map((photo) =>
                    photo.renderedThumbnailUrl ? { ...photo, renderedThumbnailUrl: null } : photo,
                  ),
                })),
              };
            });
            if (sessionRef.current && thumbnailPreviewModeRef.current !== 'fast') {
              request32BitThumbnails(thumbnailPreviewModeRef.current);
            }
          }
          const label =
            target === 'previews'
              ? 'RAW preview cache'
              : target === 'face_cache'
                ? 'face scan cache'
                : 'caches';
          notifyAction(
            result.removedFiles > 0
              ? `Cleared ${label} (${result.removedFiles.toLocaleString()} ${result.removedFiles === 1 ? 'file' : 'files'})`
              : `${label.charAt(0).toUpperCase()}${label.slice(1)} is already empty`,
          );
        }
        return result;
      } catch (err) {
        notifyAction(`Could not clear cache: ${String(err)}`);
        return null;
      }
    },
    [gateway, request32BitThumbnails, notifyAction],
  );

  const applyLoadedSession = useCallback(
    (loaded: StudioSessionSnapshot, preferredClusterId?: string) => {
      setSession(loaded);
      setResolvedAmbiguousIds(new Set());
      setActiveDirectoryId(null);
      setActiveVirtualFolderIdState('all');
      setActiveDevelopFolderId(null);
      lastSelectedPhotoIdRef.current = null;
      setDevelopSourceModeState(loaded.isClustered && preferredClusterId ? 'subjects' : 'folders');
      setLiveHistogram(null);

      const allPhotos = loaded.clusters.flatMap((c) => c.photos);
      const targetCluster =
        (preferredClusterId
          ? loaded.clusters.find((c) => c.anchor.id === preferredClusterId)
          : undefined) ?? loaded.clusters[0];

      if (targetCluster) {
        setActiveClusterId(targetCluster.anchor.id);
        const initialPool =
          loaded.isClustered && preferredClusterId ? targetCluster.photos : allPhotos;
        const nonSlates = initialPool.filter((p) => !p.isSlateAnchor);
        setSelectedPhotoIds(new Set());
        setActiveDevelopPhotoId(nonSlates[0]?.id ?? initialPool[0]?.id ?? null);
      } else {
        setActiveClusterId('');
        setSelectedPhotoIds(new Set());
        setActiveDevelopPhotoId(null);
      }
    },
    [],
  );

  useEffect(() => {
    if (!statusBanner) return;
    const timer = window.setTimeout(() => setStatusBanner(null), 4000);
    return () => window.clearTimeout(timer);
  }, [statusBanner]);

  const runStartupUpdateCheckPhase = useCallback(async () => {
    if (!mountedRef.current) return;
    setBootPhase('checking_updates');
    const [updateResult] = await Promise.all([
      checkGithubForAppUpdate(detectedAppVersionRef.current),
      initialSessionPromiseRef.current.catch(() => { }),
    ]);
    if (!mountedRef.current) return;
    if (updateResult.status === 'update_available') {
      setStartupUpdate(updateResult);
      setBootPhase('update_prompt');
      return;
    }
    setStartupUpdate(null);
    setBootPhase('ready');
    setIsBootReady(true);
  }, []);

  const downloadModelsForStartup = useCallback(async (): Promise<boolean> => {
    if (!gateway.downloadMissingModels) return true;
    setModelDownloadError(null);
    setModelDownloadProgress(null);
    setBootPhase('downloading_models');
    try {
      const outcome = await gateway.downloadMissingModels();
      if (!mountedRef.current) return false;
      if (outcome) {
        setModelManifestStatus(outcome.manifest);
        setRuntimeInfo(outcome.runtime);
        if (outcome.runtime.appVersion) {
          detectedAppVersionRef.current = outcome.runtime.appVersion;
        }
      } else {
        const refreshed = await gateway.checkRuntimeInfo();
        if (mountedRef.current && refreshed) {
          setRuntimeInfo(refreshed);
          if (refreshed.appVersion) {
            detectedAppVersionRef.current = refreshed.appVersion;
          }
        }
      }
      setModelDownloadProgress(null);
      return true;
    } catch (err) {
      if (!mountedRef.current) return false;
      const message =
        err instanceof Error
          ? err.message
          : typeof err === 'string'
            ? err
            : 'Could not download AI models.';
      setModelDownloadError(message);
      setBootPhase('models_error');
      return false;
    }
  }, [gateway]);

  const retryModelDownload = useCallback(async () => {
    const ok = await downloadModelsForStartup();
    if (ok && mountedRef.current) {
      await runStartupUpdateCheckPhase();
    }
  }, [downloadModelsForStartup, runStartupUpdateCheckPhase]);

  const continueWithoutModels = useCallback(async () => {
    if (gateway.cancelModelDownloads) {
      try {
        await gateway.cancelModelDownloads();
      } catch {
        // Ignore cancel failures when continuing offline
      }
    }
    if (!mountedRef.current) return;
    setModelDownloadError(null);
    setModelDownloadProgress(null);
    await runStartupUpdateCheckPhase();
  }, [gateway, runStartupUpdateCheckPhase]);

  const dismissStartupUpdate = useCallback(() => {
    setStartupUpdate(null);
    setBootPhase('ready');
    setIsBootReady(true);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    let active = true;
    const unsubscribe = gateway.subscribeScanProgress((progress) => {
      if (!active) return;
      setScanProgress(progress);
      if (progress.phase === 'export') {
        const pct =
          progress.total > 0
            ? Math.min(99, Math.max(0, Math.round((progress.completed / progress.total) * 100)))
            : 0;
        setExportJob((prev) => ({
          isExporting: prev?.isExporting ?? true,
          completed: progress.completed,
          total: progress.total,
          percent: prev && !prev.isExporting ? prev.percent : pct,
          message: progress.message,
          completedCount: prev?.completedCount ?? null,
          failedCount: prev?.failedCount ?? 0,
          outputRoot: prev?.outputRoot ?? null,
          error: null,
        }));
      }
    });
    const unsubscribeModelProgress = gateway.subscribeModelDownloadProgress?.((progress) => {
      if (!active) return;
      setModelDownloadProgress(progress);
    });
    const unsubscribeThumbnailProgress = gateway.subscribeThumbnailRenderProgress?.((progress) => {
      if (!active) return;
      setThumbnailRenderProgress(progress.active && progress.total > 0 ? progress : null);
    });
    const unsubscribePreviews = gateway.subscribePreviewsReady?.((batch) => {
      if (!active || batch.updates.length === 0) return;
      const patchById = new Map(batch.updates.map((u) => [u.photoId, u]));
      setSession((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          clusters: prev.clusters.map((cluster) => {
            const anchorPatch = patchById.get(cluster.anchor.anchorPhotoId);
            let changed = Boolean(anchorPatch?.imageUrl);
            const nextPhotos = cluster.photos.map((photo) => {
              const patch = patchById.get(photo.id);
              if (!patch) return photo;
              changed = true;
              const nextRendered = patch.clearRenderedThumbnail
                ? null
                : patch.renderedThumbnailUrl !== undefined && patch.renderedThumbnailUrl !== null
                  ? patch.renderedThumbnailUrl
                  : photo.renderedThumbnailUrl;
              return {
                ...photo,
                imageUrl: patch.imageUrl ?? photo.imageUrl,
                renderedThumbnailUrl: nextRendered,
                capturedAtEpochSec: patch.capturedAtEpochSec ?? photo.capturedAtEpochSec,
                capturedAtLabel: patch.capturedAtLabel ?? photo.capturedAtLabel,
              };
            });
            if (!changed) return cluster;
            return {
              ...cluster,
              anchor: anchorPatch?.imageUrl
                ? { ...cluster.anchor, thumbnailUrl: anchorPatch.imageUrl }
                : cluster.anchor,
              photos: nextPhotos,
            };
          }),
        };
      });
      const currentFolderPath = activeFolderPathRef.current;
      if (currentFolderPath) {
        const freshUrls = batch.updates
          .map((u) => u.imageUrl)
          .filter((u): u is string => Boolean(u));
        if (freshUrls.length > 0) {
          setRecentProjects((prev) => {
            const existing = prev.find((p) => p.folderPath === currentFolderPath);
            if (!existing || existing.coverThumbnails.length >= 3) return prev;
            const mergedCovers = Array.from(
              new Set([...existing.coverThumbnails, ...freshUrls]),
            ).slice(0, 3);
            if (mergedCovers.length === existing.coverThumbnails.length) return prev;
            const next = prev.map((p) =>
              p.id === existing.id ? { ...p, coverThumbnails: mergedCovers } : p,
            );
            saveRecentProjectsToStorage(next);
            return next;
          });
        }
      }
    });
    const initialPrefs = loadStudioPreferencesFromStorage();
    const syncInitialRuntime = async (): Promise<RuntimeInfo | null> => {
      if (!initialPrefs.enableGpuAcceleration && gateway.setGpuAcceleration) {
        try {
          await gateway.setGpuAcceleration(false);
        } catch {
          // Ignore cleanly and fall back to runtime check
        }
      }
      if (initialPrefs.customCacheDir && gateway.setStorageCacheDir) {
        try {
          await gateway.setStorageCacheDir(initialPrefs.customCacheDir);
        } catch {
          // If a saved external drive/folder is missing, fall back cleanly to system temp
          updatePreferences({ customCacheDir: null });
        }
      }
      if (
        gateway.setCustomAiModels &&
        (initialPrefs.customDetectorModelPath ||
          initialPrefs.customRecognizerModelPath ||
          initialPrefs.customSegmenterModelPath)
      ) {
        try {
          const info = await gateway.setCustomAiModels(
            initialPrefs.customDetectorModelPath,
            initialPrefs.customRecognizerModelPath,
            initialPrefs.customSegmenterModelPath,
          );
          if (active && info) {
            setRuntimeInfo(info);
            if (info.appVersion) {
              detectedAppVersionRef.current = info.appVersion;
            }
            return info;
          }
        } catch {
          // If a saved custom model was moved/deleted, fall back cleanly to defaults
          updatePreferences({
            customDetectorModelPath: null,
            customRecognizerModelPath: null,
            customSegmenterModelPath: null,
          });
        }
      }
      const info = await gateway.checkRuntimeInfo();
      if (active && info) {
        setRuntimeInfo(info);
        if (info.appVersion) {
          detectedAppVersionRef.current = info.appVersion;
        }
      }
      return info;
    };
    const loadInitialSession = async () => {
      try {
        let loaded = await gateway.loadSession();
        if (!loaded && initialPrefs.openLastProjectOnStartup && gateway.isDesktopRuntime()) {
          const lastProject = loadRecentProjectsFromStorage()[0];
          if (lastProject?.folderPath) {
            loaded = await gateway.openShootFolderAtPath(lastProject.folderPath);
          }
        }
        if (!active || !loaded) return;
        applyLoadedSession(loaded);
        recordRecentFromSnapshot(loaded);
        if (initialPrefs.thumbnailPreviewMode !== 'fast') {
          request32BitThumbnails(initialPrefs.thumbnailPreviewMode);
        }
        setActiveWorkspace(loaded.isClustered ? initialPrefs.defaultOpenWorkspace : 'develop');
      } catch {
        // Ignore startup session errors cleanly
      }
    };

    const sessionPromise = loadInitialSession();
    initialSessionPromiseRef.current = sessionPromise;

    const runStartupPipeline = async () => {
      setBootPhase('checking_models');
      await syncInitialRuntime();
      if (!active) return;

      if (gateway.checkModelsManifest) {
        try {
          const manifest = await gateway.checkModelsManifest();
          if (!active) return;
          if (manifest) {
            setModelManifestStatus(manifest);
            if (!manifest.allReady) {
              const downloaded = await downloadModelsForStartup();
              if (!downloaded || !active) {
                return;
              }
            }
          }
        } catch (err) {
          if (!active) return;
          const message =
            err instanceof Error
              ? err.message
              : typeof err === 'string'
                ? err
                : 'Could not verify AI model manifest.';
          setModelDownloadError(message);
          setBootPhase('models_error');
          return;
        }
      }

      if (!active) return;
      await runStartupUpdateCheckPhase();
    };

    void runStartupPipeline();

    return () => {
      active = false;
      mountedRef.current = false;
      unsubscribe();
      unsubscribeModelProgress?.();
      unsubscribeThumbnailProgress?.();
      unsubscribePreviews?.();
      const pendingItems: { photoId: string; recipe: DevelopRecipe }[] = [];
      for (const [pendingPhotoId, pending] of pendingRecipeSavesRef.current.entries()) {
        window.clearTimeout(pending.timerId);
        pendingItems.push({ photoId: pendingPhotoId, recipe: pending.recipe });
      }
      pendingRecipeSavesRef.current.clear();
      if (pendingItems.length > 0) {
        if (gateway.savePhotoDevelopRecipesBatch) {
          void gateway.savePhotoDevelopRecipesBatch(pendingItems, null).catch(() => { });
        } else {
          for (const item of pendingItems) {
            void gateway.savePhotoDevelopRecipe(item.photoId, item.recipe).catch(() => { });
          }
        }
      }
    };
  }, [
    gateway,
    applyLoadedSession,
    recordRecentFromSnapshot,
    request32BitThumbnails,
    updatePreferences,
    setActiveWorkspace,
    downloadModelsForStartup,
    runStartupUpdateCheckPhase,
  ]);

  const sortedClusters = useMemo(() => {
    if (!session) return [];
    const enriched = session.clusters.map((cluster) => {
      if (!session.isClustered || cluster.anchor.id.startsWith('_')) {
        return cluster;
      }
      const lowConfidenceCount = countClusterReviewWarnings(
        cluster,
        reviewWarningThreshold,
        resolvedAmbiguousIds,
      );
      const backendPending = (session.ambiguousQueue ?? []).filter(
        (entry) =>
          entry.candidateClusterId === cluster.anchor.id &&
          !resolvedAmbiguousIds.has(entry.photo.id),
      ).length;
      const totalReviewCount = lowConfidenceCount + backendPending;
      if (totalReviewCount === cluster.anchor.ambiguousCount) {
        return cluster;
      }
      return {
        ...cluster,
        anchor: {
          ...cluster.anchor,
          ambiguousCount: totalReviewCount,
        },
      };
    });

    if (anchorSortMode === 'confidence') {
      enriched.sort((a, b) => {
        if (b.anchor.ambiguousCount !== a.anchor.ambiguousCount) {
          return b.anchor.ambiguousCount - a.anchor.ambiguousCount;
        }
        return a.anchor.ocrConfidencePct - b.anchor.ocrConfidencePct;
      });
    } else if (anchorSortMode === 'exposures') {
      enriched.sort((a, b) => b.photos.length - a.photos.length);
    }
    return enriched;
  }, [session, anchorSortMode, reviewWarningThreshold, resolvedAmbiguousIds]);

  const activeCluster = useMemo(() => {
    if (!session) return null;
    return (
      session.clusters.find((c) => c.anchor.id === activeClusterId) ?? session.clusters[0] ?? null
    );
  }, [session, activeClusterId]);

  const effectiveDevelopSourceMode: DevelopSourceMode =
    session && !session.isClustered ? 'folders' : developSourceMode;

  const developPhotos = useMemo<readonly PhotoAsset[]>(() => {
    if (!session) return [];
    if (effectiveDevelopSourceMode === 'subjects' && activeCluster) {
      return activeCluster.photos;
    }
    const all = session.clusters.flatMap((c) => c.photos);
    if (!activeDevelopFolderId) return all;
    return all.filter((p) => p.directoryId === activeDevelopFolderId);
  }, [session, effectiveDevelopSourceMode, activeCluster, activeDevelopFolderId]);

  const activeDevelopScopeLabel = useMemo(() => {
    if (!session) return '';
    if (effectiveDevelopSourceMode === 'subjects' && activeCluster) {
      return activeCluster.anchor.subjectName;
    }
    if (!activeDevelopFolderId) {
      return `All Imported (${session.telemetry.projectName})`;
    }
    return (
      session.directories.find((d) => d.id === activeDevelopFolderId)?.name ?? 'Selected Folder'
    );
  }, [session, effectiveDevelopSourceMode, activeCluster, activeDevelopFolderId]);

  const activeDevelopPhoto = useMemo(() => {
    if (developPhotos.length === 0) return null;
    if (activeDevelopPhotoId) {
      const found = developPhotos.find((p) => p.id === activeDevelopPhotoId);
      if (found) return found;
    }
    return developPhotos.find((p) => !p.isSlateAnchor) ?? developPhotos[0] ?? null;
  }, [developPhotos, activeDevelopPhotoId]);

  const activeVirtualFolderAllowedIds = useMemo<ReadonlySet<string> | null>(() => {
    if (!session || activeVirtualFolderId === 'all' || activeVirtualFolderId === 'unassigned') {
      return null;
    }
    return collectDescendantFolderIds(session.virtualFolders, activeVirtualFolderId);
  }, [session, activeVirtualFolderId]);

  const visibleClusterPhotos = useMemo(() => {
    if (!activeCluster) return [];
    return filterClusterPhotos(activeCluster.photos, {
      sfaceThreshold,
      reviewWarningThreshold,
      activeDirectoryId,
      activeVirtualFolderId,
      allowedVirtualFolderIds: activeVirtualFolderAllowedIds,
      smartBin,
      searchQuery,
    });
  }, [
    activeCluster,
    sfaceThreshold,
    reviewWarningThreshold,
    activeDirectoryId,
    activeVirtualFolderId,
    activeVirtualFolderAllowedIds,
    smartBin,
    searchQuery,
  ]);

  const ambiguousReviewQueue = useMemo<readonly AmbiguousReviewItem[]>(() => {
    if (!session || !session.isClustered) return [];

    const items: AmbiguousReviewItem[] = [];
    const seenIds = new Set<string>();

    // 1. Borderline singleton candidates from the Rust clustering engine
    if (session.ambiguousQueue !== undefined) {
      for (const entry of session.ambiguousQueue) {
        if (resolvedAmbiguousIds.has(entry.photo.id) || seenIds.has(entry.photo.id)) continue;
        const candidateCluster = session.clusters.find(
          (c) => c.anchor.id === entry.candidateClusterId,
        );
        if (candidateCluster) {
          seenIds.add(entry.photo.id);
          items.push({
            photo: entry.photo,
            candidateCluster,
            reason: entry.reason,
            source: 'backend_queue',
          });
        }
      }
    }

    // 2. Already-grouped frames scoring below the visual review warning threshold (default 75%)
    //    Surfaces rare look-alike cluster merges (e.g. 66%-69%) immediately for 1-click triage.
    const pctLabel = Math.round(reviewWarningThreshold * 100);
    for (const cluster of session.clusters) {
      if (cluster.anchor.id.startsWith('_')) continue;
      for (const photo of cluster.photos) {
        if (resolvedAmbiguousIds.has(photo.id) || seenIds.has(photo.id)) continue;
        if (needsVisualMatchReview(photo, reviewWarningThreshold)) {
          seenIds.add(photo.id);
          items.push({
            photo,
            candidateCluster: cluster,
            reason: `Match below ${pctLabel}%`,
            source: 'cluster_low_confidence',
          });
        }
      }
    }

    return items;
  }, [session, reviewWarningThreshold, resolvedAmbiguousIds]);

  const openShootFolder = useCallback(async () => {
    if (!isDesktopRuntime) {
      notifyAction(
        'You are in Browser Preview — run `npm run tauri:dev` to open local RAW/JPEG folders',
      );
      return;
    }

    try {
      setIsScanningFolder(true);
      setScanProgress(null);
      notifyAction('Loading RAW & standard photo previews...');
      const loaded = await gateway.openShootFolder();
      if (loaded) {
        setDevelopHistory(EMPTY_DEVELOP_HISTORY);
        setSlateOcrResults([]);
        setThumbnailRenderProgress(null);
        applyLoadedSession(loaded);
        recordRecentFromSnapshot(loaded);
        if (thumbnailPreviewModeRef.current !== 'fast') {
          request32BitThumbnails(thumbnailPreviewModeRef.current);
        }
        const restoredSubjects = loaded.clusters.filter((c) => !c.anchor.id.startsWith('_')).length;
        setActiveWorkspace(loaded.isClustered ? preferences.defaultOpenWorkspace : 'develop');
        notifyAction(
          loaded.isClustered
            ? `Restored saved organization (${restoredSubjects} subjects • ${loaded.telemetry.totalPhotos} photos)`
            : `Loaded ${loaded.telemetry.totalPhotos} photo(s) in ${loaded.telemetry.indexElapsedSeconds}s — edit first, then click "Organize by Face"`,
        );
      } else {
        notifyAction('Folder selection cancelled');
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifyAction(`Folder load failed: ${msg}`);
    } finally {
      setIsScanningFolder(false);
      setScanProgress(null);
    }
  }, [
    isDesktopRuntime,
    gateway,
    preferences.defaultOpenWorkspace,
    applyLoadedSession,
    recordRecentFromSnapshot,
    request32BitThumbnails,
    setActiveWorkspace,
    notifyAction,
  ]);

  const openRecentShootFolder = useCallback(
    async (folderPath: string) => {
      if (!isDesktopRuntime) {
        notifyAction('Opening local folders requires Kroma Desktop (`npm run tauri:dev`)');
        return;
      }

      try {
        setIsScanningFolder(true);
        setScanProgress(null);
        notifyAction(`Opening ${folderPath}...`);
        const loaded = await gateway.openShootFolderAtPath(folderPath);
        if (loaded) {
          setDevelopHistory(EMPTY_DEVELOP_HISTORY);
          setThumbnailRenderProgress(null);
          applyLoadedSession(loaded);
          recordRecentFromSnapshot(loaded);
          if (thumbnailPreviewModeRef.current !== 'fast') {
            request32BitThumbnails(thumbnailPreviewModeRef.current);
          }
          const restoredSubjects = loaded.clusters.filter((c) => !c.anchor.id.startsWith('_')).length;
          setActiveWorkspace(loaded.isClustered ? preferences.defaultOpenWorkspace : 'develop');
          notifyAction(
            loaded.isClustered
              ? `Restored saved organization (${restoredSubjects} subjects • ${loaded.telemetry.totalPhotos} photos)`
              : `Loaded ${loaded.telemetry.totalPhotos} photo(s) in ${loaded.telemetry.indexElapsedSeconds}s`,
          );
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        notifyAction(`Could not open recent project: ${msg}`);
      } finally {
        setIsScanningFolder(false);
        setScanProgress(null);
      }
    },
    [
      isDesktopRuntime,
      gateway,
      preferences.defaultOpenWorkspace,
      applyLoadedSession,
      recordRecentFromSnapshot,
      request32BitThumbnails,
      setActiveWorkspace,
      notifyAction,
    ],
  );

  const runFaceOrganizer = useCallback(async () => {
    if (!session) return;
    try {
      setIsScanningFolder(true);
      setScanProgress(null);
      notifyAction(
        `Grouping photos by face (strictness ${sfaceThreshold.toFixed(2)}${useTemporalAssist ? ' + Time Assist' : ''})…`,
      );
      const updated = await gateway.runFaceClustering(sfaceThreshold, useTemporalAssist);
      if (updated) {
        setLastClusteredThreshold(sfaceThreshold);
        setLastClusteredTemporalAssist(useTemporalAssist);
        applyLoadedSession(updated);
        recordRecentFromSnapshot(updated);
        setActiveWorkspace('culling');
        notifyAction(
          `Organized ${updated.telemetry.totalPhotos} photo(s) into ${updated.clusters.length} group(s) in ${updated.telemetry.indexElapsedSeconds}s — all edits preserved`,
        );
      }
    } catch (err) {
      notifyAction(`Face clustering failed: ${String(err)}`);
    } finally {
      setIsScanningFolder(false);
      setScanProgress(null);
    }
  }, [
    session,
    runtimeInfo,
    sfaceThreshold,
    useTemporalAssist,
    gateway,
    applyLoadedSession,
    recordRecentFromSnapshot,
    notifyAction,
  ]);

  const scanProjectSlatesOcr = useCallback(
    async (options?: SlateOcrScanOptions) => {
      if (!session || !gateway.scanProjectSlatesOcr) return;
      try {
        setIsScanningOcr(true);
        setScanProgress(null);
        const count = options?.clusterIds?.length;
        notifyAction(
          count
            ? `Scanning ${count} selected subject${count === 1 ? '' : 's'}…`
            : 'Scanning whiteboards…',
        );
        const results = await gateway.scanProjectSlatesOcr(options);
        setSlateOcrResults((prev) => mergeSlateOcrResults(prev, results));
        const foundCount = results.filter((r) => Boolean(r.proposedName)).length;
        notifyAction(
          `Read ${foundCount} whiteboard${foundCount === 1 ? '' : 's'} across ${results.length} scanned subject${results.length === 1 ? '' : 's'}`,
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        notifyAction(`Slate OCR scan failed: ${msg}`);
      } finally {
        setIsScanningOcr(false);
        setScanProgress(null);
      }
    },
    [session, gateway, notifyAction],
  );

  const cancelSlateOcrScan = useCallback(async () => {
    if (!gateway.cancelSlateOcrScan) return;
    try {
      await gateway.cancelSlateOcrScan();
      notifyAction('Stopping Slate OCR scan…');
    } catch {
      // Ignore cancel errors cleanly
    }
  }, [gateway, notifyAction]);

  const scanSinglePhotoOcr = useCallback(
    async (photoId: string): Promise<PhotoOcrResult | null> => {
      if (!gateway.scanSinglePhotoOcr) return null;
      try {
        const result = await gateway.scanSinglePhotoOcr(photoId);
        if (result) {
          notifyAction(
            result.proposedName
              ? `Read "${result.proposedName}" (${Math.round(result.confidence * 100)}%)`
              : `No whiteboard name detected in ${result.fileName}`,
          );
        }
        return result;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        notifyAction(`Frame OCR failed: ${msg}`);
        return null;
      }
    },
    [gateway, notifyAction],
  );

  const applySlateOcrRenames = useCallback(
    async (items: readonly SlateRenameApplyItem[]): Promise<boolean> => {
      if (items.length === 0 || !gateway.applySlateOcrRenames) return false;
      try {
        const updated = await gateway.applySlateOcrRenames(items);
        if (updated) {
          setSession(updated);
          recordRecentFromSnapshot(updated);
          notifyAction(
            items.length === 1
              ? `Renamed subject to "${items[0].nextName}" via Slate OCR (${formatShortcutCombo('Ctrl+Z')} to undo)`
              : `Applied Slate OCR names to ${items.length} subjects (${formatShortcutCombo('Ctrl+Z')} to undo)`,
          );
          return true;
        }
        return false;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        notifyAction(`Could not apply OCR renames: ${msg}`);
        return false;
      }
    },
    [gateway, recordRecentFromSnapshot, notifyAction],
  );

  const setDevelopSourceMode = useCallback(
    (mode: DevelopSourceMode) => {
      setDevelopSourceModeState(mode);
      setLiveHistogram(null);
      if (!session) return;
      const pool =
        mode === 'subjects' && activeCluster
          ? activeCluster.photos
          : activeDevelopFolderId
            ? session.clusters
              .flatMap((c) => c.photos)
              .filter((p) => p.directoryId === activeDevelopFolderId)
            : session.clusters.flatMap((c) => c.photos);
      const nonSlates = pool.filter((p) => !p.isSlateAnchor);
      setSelectedPhotoIds(new Set());
      setActiveDevelopPhotoId(nonSlates[0]?.id ?? pool[0]?.id ?? null);
    },
    [session, activeCluster, activeDevelopFolderId],
  );

  const selectDevelopFolder = useCallback(
    (folderId: string | null) => {
      setDevelopSourceModeState('folders');
      setActiveDevelopFolderId(folderId);
      setLiveHistogram(null);
      if (!session) return;
      const all = session.clusters.flatMap((c) => c.photos);
      const filtered = folderId ? all.filter((p) => p.directoryId === folderId) : all;
      const nonSlates = filtered.filter((p) => !p.isSlateAnchor);
      setSelectedPhotoIds(new Set());
      setActiveDevelopPhotoId(nonSlates[0]?.id ?? filtered[0]?.id ?? null);
    },
    [session],
  );

  const selectCluster = useCallback(
    (clusterId: string) => {
      setActiveClusterId(clusterId);
      setLiveHistogram(null);
      const next = session?.clusters.find((c) => c.anchor.id === clusterId);
      if (next) {
        const nonSlates = next.photos.filter((p) => !p.isSlateAnchor);
        setSelectedPhotoIds(new Set());
        setActiveDevelopPhotoId(nonSlates[0]?.id ?? next.photos[0]?.id ?? null);
      }
    },
    [session],
  );

  const handleSetActiveDevelopPhotoId = useCallback((photoId: string) => {
    setActiveDevelopPhotoId(photoId);
    setLiveHistogram(null);
  }, []);

  const openPhotoInDevelop = useCallback(
    (photoId: string, clusterId?: string) => {
      if (clusterId && clusterId !== activeClusterId) {
        selectCluster(clusterId);
      }
      if (session?.isClustered) {
        setDevelopSourceModeState('subjects');
      } else {
        setDevelopSourceModeState('folders');
      }
      setActiveDevelopPhotoId(photoId);
      setLiveHistogram(null);
      setActiveWorkspace('develop');
    },
    [activeClusterId, selectCluster, session?.isClustered],
  );

  const selectNextCluster = useCallback(() => {
    if (sortedClusters.length === 0) return;
    const currentIdx = sortedClusters.findIndex((c) => c.anchor.id === activeClusterId);
    const nextCluster = sortedClusters[(currentIdx + 1) % sortedClusters.length];
    if (nextCluster) {
      selectCluster(nextCluster.anchor.id);
      notifyAction(`Switched to ${nextCluster.anchor.subjectName}`);
    }
  }, [sortedClusters, activeClusterId, selectCluster, notifyAction]);

  const selectNextSingletonCluster = useCallback(() => {
    const singletons = sortedClusters.filter(
      (c) => !c.anchor.id.startsWith('_') && c.photos.length === 1,
    );
    if (singletons.length === 0) {
      notifyAction('No remaining 1-photo subjects');
      return;
    }
    const currentIdx = singletons.findIndex((c) => c.anchor.id === activeClusterId);
    const nextSingleton = singletons[(currentIdx + 1) % singletons.length];
    if (nextSingleton) {
      selectCluster(nextSingleton.anchor.id);
      notifyAction(
        `Jumped to ${nextSingleton.anchor.subjectName} (${singletons.length} single-photo subjects left)`,
      );
    }
  }, [sortedClusters, activeClusterId, selectCluster, notifyAction]);

  const toggleDirectoryFilter = useCallback((directoryId: string) => {
    setActiveDirectoryId((prev) => (prev === directoryId ? null : directoryId));
  }, []);

  const setActiveVirtualFolderId = useCallback(
    (nextFilter: VirtualFolderFilter) => {
      setActiveVirtualFolderIdState(nextFilter);
      setSelectedPhotoIds(new Set());
      lastSelectedPhotoIdRef.current = null;
      if (!session || nextFilter === 'all') return;
      const allowed =
        nextFilter === 'unassigned'
          ? null
          : collectDescendantFolderIds(session.virtualFolders, nextFilter);
      const currentHasMatch = activeCluster?.photos.some((p) =>
        matchesVirtualFolderFilter(p.virtualFolderId, nextFilter, allowed),
      );
      if (!currentHasMatch) {
        const firstMatching = session.clusters.find((c) =>
          c.photos.some((p) => matchesVirtualFolderFilter(p.virtualFolderId, nextFilter, allowed)),
        );
        if (firstMatching) {
          setActiveClusterId(firstMatching.anchor.id);
        }
      }
    },
    [session, activeCluster],
  );

  const requestNewVirtualFolder = useCallback(() => {
    setFolderCreateNonce((n) => n + 1);
  }, []);

  const createVirtualFolder = useCallback(
    async (name: string, parentId?: string | null): Promise<boolean> => {
      if (!gateway.createVirtualFolder) return false;
      const trimmed = name.trim();
      if (!trimmed) return false;
      try {
        const updated = await gateway.createVirtualFolder(trimmed, parentId ?? null);
        if (updated) {
          setSession(updated);
          notifyAction(`Created folder "${trimmed}"`);
          return true;
        }
        return false;
      } catch (err) {
        notifyAction(err instanceof Error ? err.message : String(err));
        return false;
      }
    },
    [gateway, notifyAction],
  );

  const renameVirtualFolder = useCallback(
    async (folderId: string, nextName: string): Promise<boolean> => {
      if (!gateway.renameVirtualFolder) return false;
      const trimmed = nextName.trim();
      if (!trimmed) return false;
      try {
        const updated = await gateway.renameVirtualFolder(folderId, trimmed);
        if (updated) {
          setSession(updated);
          notifyAction(`Renamed folder to "${trimmed}"`);
          return true;
        }
        return false;
      } catch (err) {
        notifyAction(err instanceof Error ? err.message : String(err));
        return false;
      }
    },
    [gateway, notifyAction],
  );

  const moveVirtualFolder = useCallback(
    async (folderId: string, nextParentId: string | null): Promise<boolean> => {
      if (!gateway.moveVirtualFolder) return false;
      try {
        const updated = await gateway.moveVirtualFolder(folderId, nextParentId);
        if (updated) {
          setSession(updated);
          const targetName = nextParentId
            ? (updated.virtualFolders.find((f) => f.id === nextParentId)?.name ?? 'folder')
            : 'top level';
          notifyAction(`Moved folder into ${targetName}`);
          return true;
        }
        return false;
      } catch (err) {
        notifyAction(err instanceof Error ? err.message : String(err));
        return false;
      }
    },
    [gateway, notifyAction],
  );

  const deleteVirtualFolder = useCallback(
    async (folderId: string): Promise<boolean> => {
      if (!gateway.deleteVirtualFolder) return false;
      const currentFolders = sessionRef.current?.virtualFolders ?? [];
      const removed = collectDescendantFolderIds(currentFolders, folderId);
      const folderName = currentFolders.find((f) => f.id === folderId)?.name ?? 'folder';
      try {
        const updated = await gateway.deleteVirtualFolder(folderId);
        if (updated) {
          setSession(updated);
          if (
            activeVirtualFolderId !== 'all' &&
            activeVirtualFolderId !== 'unassigned' &&
            removed.has(activeVirtualFolderId)
          ) {
            setActiveVirtualFolderIdState('all');
          }
          notifyAction(`Removed folder "${folderName}" (photos moved to Unassigned)`);
          return true;
        }
        return false;
      } catch (err) {
        notifyAction(err instanceof Error ? err.message : String(err));
        return false;
      }
    },
    [gateway, activeVirtualFolderId, notifyAction],
  );

  const assignPhotosToVirtualFolder = useCallback(
    async (photoIds: readonly string[], folderId: string | null): Promise<boolean> => {
      if (!gateway.assignPhotosToVirtualFolder || photoIds.length === 0) return false;
      try {
        const updated = await gateway.assignPhotosToVirtualFolder(photoIds, folderId);
        if (updated) {
          setSession(updated);
          setSelectedPhotoIds(new Set());
          lastSelectedPhotoIdRef.current = null;
          const label = folderId
            ? (updated.virtualFolders.find((f) => f.id === folderId)?.name ?? 'folder')
            : 'Unassigned';
          notifyAction(
            `Moved ${photoIds.length} photo${photoIds.length === 1 ? '' : 's'} to ${label}`,
          );
          return true;
        }
        return false;
      } catch (err) {
        notifyAction(err instanceof Error ? err.message : String(err));
        return false;
      }
    },
    [gateway, notifyAction],
  );

  const assignClustersToVirtualFolder = useCallback(
    async (clusterIds: readonly string[], folderId: string | null): Promise<boolean> => {
      const current = sessionRef.current;
      if (!current || clusterIds.length === 0 || !gateway.assignPhotosToVirtualFolder) return false;
      const targetSet = new Set(clusterIds);
      const photoIds: string[] = [];
      let matchedSubjects = 0;
      for (const cluster of current.clusters) {
        if (!targetSet.has(cluster.anchor.id)) continue;
        matchedSubjects += 1;
        for (const photo of cluster.photos) {
          photoIds.push(photo.id);
        }
      }
      if (photoIds.length === 0) return false;
      try {
        const updated = await gateway.assignPhotosToVirtualFolder(photoIds, folderId);
        if (updated) {
          setSession(updated);
          const label = folderId
            ? (updated.virtualFolders.find((f) => f.id === folderId)?.name ?? 'folder')
            : 'Unassigned';
          notifyAction(
            `Moved ${matchedSubjects} subject${matchedSubjects === 1 ? '' : 's'} (${photoIds.length} photo${photoIds.length === 1 ? '' : 's'}) to ${label}`,
          );
          return true;
        }
        return false;
      } catch (err) {
        notifyAction(err instanceof Error ? err.message : String(err));
        return false;
      }
    },
    [gateway, notifyAction],
  );

  const toggleLandmarksOverlay = useCallback(() => {
    setPreferences((prev) => {
      const next = { ...prev, showLandmarksOverlay: !prev.showLandmarksOverlay };
      saveStudioPreferencesToStorage(next);
      return next;
    });
  }, []);

  const togglePhotoSelection = useCallback(
    (photoId: string, options?: { readonly shiftKey?: boolean }) => {
      const anchorId = lastSelectedPhotoIdRef.current;
      if (options?.shiftKey && anchorId) {
        const startIdx = visibleClusterPhotos.findIndex((p) => p.id === anchorId);
        const endIdx = visibleClusterPhotos.findIndex((p) => p.id === photoId);
        if (startIdx !== -1 && endIdx !== -1) {
          const lo = Math.min(startIdx, endIdx);
          const hi = Math.max(startIdx, endIdx);
          setSelectedPhotoIds((prev) => {
            const next = new Set(prev);
            for (let i = lo; i <= hi; i += 1) {
              const p = visibleClusterPhotos[i];
              if (p) next.add(p.id);
            }
            return next;
          });
          lastSelectedPhotoIdRef.current = photoId;
          return;
        }
      }

      lastSelectedPhotoIdRef.current = photoId;
      setSelectedPhotoIds((prev) => {
        const next = new Set(prev);
        if (next.has(photoId)) {
          next.delete(photoId);
        } else {
          next.add(photoId);
        }
        return next;
      });
    },
    [visibleClusterPhotos],
  );

  const selectAllVisiblePhotos = useCallback(() => {
    const selectable = visibleClusterPhotos.map((p) => p.id);
    setSelectedPhotoIds((prev) =>
      prev.size === selectable.length ? new Set() : new Set(selectable),
    );
  }, [visibleClusterPhotos]);

  const peekDevelopWorkingBuffer = useCallback(
    (
      photoId: string,
      maxDim = 1600,
      precision: WorkingBufferPrecision = 'f16',
    ): DevelopWorkingBuffer | null => {
      if (!gateway.peekDevelopWorkingBuffer) return null;
      return gateway.peekDevelopWorkingBuffer(photoId, maxDim, precision);
    },
    [gateway],
  );

  const loadDevelopWorkingBuffer = useCallback(
    async (
      photoId: string,
      maxDim = 1600,
      precision: WorkingBufferPrecision = 'f16',
    ): Promise<DevelopWorkingBuffer | null> => {
      if (!gateway.loadDevelopWorkingBuffer) return null;
      return gateway.loadDevelopWorkingBuffer(photoId, maxDim, precision);
    },
    [gateway],
  );

  const generateAiMask = useCallback(
    async (photoId: string, kind: AiMaskKind): Promise<GeneratedAiMaskResult | null> => {
      if (!gateway.generateAiMask) return null;
      try {
        return await gateway.generateAiMask(photoId, kind);
      } catch (err) {
        notifyAction(`AI Mask error: ${err instanceof Error ? err.message : String(err)}`);
        return null;
      }
    },
    [gateway, notifyAction],
  );

  const updatePhotoRecipe = useCallback(
    (
      photoId: string,
      nextRecipe: DevelopRecipe,
      label: string = 'Develop edit',
      coalesce: boolean = true,
    ) => {
      const beforeRecipe = findPhotoRecipe(photoId);
      if (beforeRecipe) {
        setDevelopHistory((prev) =>
          recordDevelopHistory(prev, {
            label,
            coalesceKey: coalesce ? `edit:${photoId}:${label}` : undefined,
            timestampMs: Date.now(),
            changes: [{ photoId, before: beforeRecipe, after: nextRecipe }],
          }),
        );
      }

      setSession((prev) => {
        if (!prev) return prev;
        let touched = false;
        const nextClusters = prev.clusters.map((cluster) => {
          const idx = cluster.photos.findIndex((photo) => photo.id === photoId);
          if (idx === -1) return cluster;
          touched = true;
          const nextPhotos = cluster.photos.slice();
          nextPhotos[idx] = { ...nextPhotos[idx], developRecipe: nextRecipe };
          return { ...cluster, photos: nextPhotos };
        });
        return touched ? { ...prev, clusters: nextClusters } : prev;
      });

      const existingPending = pendingRecipeSavesRef.current.get(photoId);
      if (existingPending) {
        window.clearTimeout(existingPending.timerId);
      }
      const timerId = window.setTimeout(() => {
        pendingRecipeSavesRef.current.delete(photoId);
        void gateway
          .savePhotoDevelopRecipe(photoId, nextRecipe)
          .then((hist) => {
            if (hist) setLiveHistogram(hist);
            if (thumbnailPreviewModeRef.current !== 'fast') {
              request32BitThumbnails(thumbnailPreviewModeRef.current, [photoId]);
            }
          })
          .catch((err) => {
            notifyAction(`Sidecar save error: ${String(err)}`);
          });
      }, 240);
      pendingRecipeSavesRef.current.set(photoId, { timerId, recipe: nextRecipe });
    },
    [findPhotoRecipe, gateway, request32BitThumbnails, notifyAction],
  );

  const flushPendingRecipeSaves = useCallback(async () => {
    const entries = Array.from(pendingRecipeSavesRef.current.entries());
    if (entries.length === 0) return;
    pendingRecipeSavesRef.current.clear();
    const items = entries.map(([photoId, entry]) => {
      window.clearTimeout(entry.timerId);
      return { photoId, recipe: entry.recipe };
    });
    if (gateway.savePhotoDevelopRecipesBatch) {
      await gateway.savePhotoDevelopRecipesBatch(items, null).catch(() => { });
      return;
    }
    await Promise.allSettled(
      items.map((item) => gateway.savePhotoDevelopRecipe(item.photoId, item.recipe)),
    );
  }, [gateway]);

  const syncRecipeFromPhoto = useCallback(
    (sourcePhotoId: string, mask: DevelopSyncMask, scope: SyncTargetScope) => {
      const current = sessionRef.current;
      if (!current) return;
      const src = findPhotoRecipe(sourcePhotoId);
      if (!src) return;

      const activePoolIds = new Set(developPhotos.map((p) => p.id));
      const targetIds: string[] = [];
      const changes: PhotoRecipeDiff[] = [];

      const nextClusters = current.clusters.map((cluster) => {
        let clusterTouched = false;
        const nextPhotos = cluster.photos.map((photo) => {
          if (photo.id === sourcePhotoId) return photo;
          const shouldUpdate =
            scope === 'entire_shoot'
              ? true
              : scope === 'selected_in_cluster'
                ? selectedPhotoIds.has(photo.id)
                : activePoolIds.has(photo.id);
          if (!shouldUpdate) return photo;
          const after = syncDevelopRecipe(photo.developRecipe, src, mask);
          targetIds.push(photo.id);
          changes.push({ photoId: photo.id, before: photo.developRecipe, after });
          clusterTouched = true;
          return { ...photo, developRecipe: after };
        });
        return clusterTouched ? { ...cluster, photos: nextPhotos } : cluster;
      });

      if (changes.length > 0) {
        for (const id of [sourcePhotoId, ...targetIds]) {
          const pending = pendingRecipeSavesRef.current.get(id);
          if (pending) {
            window.clearTimeout(pending.timerId);
            pendingRecipeSavesRef.current.delete(id);
          }
        }
        setDevelopHistory((prev) =>
          recordDevelopHistory(prev, {
            label: `Sync Settings (${changes.length})`,
            timestampMs: Date.now(),
            changes,
          }),
        );
        setSession((prev) => (prev ? { ...prev, clusters: nextClusters } : prev));
        notifyAction(`Synced Develop settings across ${targetIds.length} photo(s)`);
        void gateway
          .syncDevelopRecipeToPhotos(sourcePhotoId, targetIds, src, mask)
          .then(() => {
            if (thumbnailPreviewModeRef.current !== 'fast') {
              request32BitThumbnails(thumbnailPreviewModeRef.current, targetIds);
            }
          })
          .catch((err) => notifyAction(`Sync error: ${String(err)}`));
      }
    },
    [findPhotoRecipe, developPhotos, selectedPhotoIds, gateway, request32BitThumbnails, notifyAction],
  );

  const applyRelativeNudgeToClusterSelection = useCallback(
    (delta: RelativeDevelopDelta, label: string) => {
      const current = sessionRef.current;
      if (!current) return;
      const activePoolIds = new Set(developPhotos.map((p) => p.id));
      const nudgedIds: string[] = [];
      const changes: PhotoRecipeDiff[] = [];

      const nextClusters = current.clusters.map((cluster) => {
        let clusterTouched = false;
        const nextPhotos = cluster.photos.map((photo) => {
          if (!activePoolIds.has(photo.id)) return photo;
          const inTarget =
            photo.id === activeDevelopPhoto?.id ||
            (selectedPhotoIds.size > 0 && selectedPhotoIds.has(photo.id));
          if (!inTarget) return photo;
          const after = applyRelativeDevelopDelta(photo.developRecipe, delta);
          nudgedIds.push(photo.id);
          changes.push({ photoId: photo.id, before: photo.developRecipe, after });
          clusterTouched = true;
          return { ...photo, developRecipe: after };
        });
        return clusterTouched ? { ...cluster, photos: nextPhotos } : cluster;
      });

      if (changes.length > 0) {
        for (const id of nudgedIds) {
          const pending = pendingRecipeSavesRef.current.get(id);
          if (pending) {
            window.clearTimeout(pending.timerId);
            pendingRecipeSavesRef.current.delete(id);
          }
        }
        setDevelopHistory((prev) =>
          recordDevelopHistory(prev, {
            label: `Quick Develop (${label})`,
            timestampMs: Date.now(),
            changes,
          }),
        );
        setSession((prev) => (prev ? { ...prev, clusters: nextClusters } : prev));
        notifyAction(`Quick Develop ${label} applied to ${nudgedIds.length} photo(s)`);
        void gateway
          .applyRelativeDevelopDeltaToPhotos(nudgedIds, delta)
          .then(() => {
            if (thumbnailPreviewModeRef.current !== 'fast') {
              request32BitThumbnails(thumbnailPreviewModeRef.current, nudgedIds);
            }
          })
          .catch((err) => notifyAction(`Quick Develop error: ${String(err)}`));
      }
    },
    [developPhotos, selectedPhotoIds, activeDevelopPhoto, gateway, request32BitThumbnails, notifyAction],
  );

  const copyActiveRecipe = useCallback(() => {
    if (!activeDevelopPhoto) return;
    setCopiedRecipe(activeDevelopPhoto.developRecipe);
    notifyAction(`Copied recipe from ${activeDevelopPhoto.fileName}`);
  }, [activeDevelopPhoto, notifyAction]);

  const pasteRecipeToClusterSelection = useCallback(() => {
    const current = sessionRef.current;
    if (!copiedRecipe || !current) return;
    const activePoolIds = new Set(developPhotos.map((p) => p.id));
    const pastedIds: string[] = [];
    const changes: PhotoRecipeDiff[] = [];

    const nextClusters = current.clusters.map((cluster) => {
      let clusterTouched = false;
      const nextPhotos = cluster.photos.map((photo) => {
        if (!activePoolIds.has(photo.id)) return photo;
        if (!selectedPhotoIds.has(photo.id) && photo.id !== activeDevelopPhoto?.id) {
          return photo;
        }
        pastedIds.push(photo.id);
        changes.push({ photoId: photo.id, before: photo.developRecipe, after: copiedRecipe });
        clusterTouched = true;
        return { ...photo, developRecipe: copiedRecipe };
      });
      return clusterTouched ? { ...cluster, photos: nextPhotos } : cluster;
    });

    if (changes.length > 0) {
      for (const id of pastedIds) {
        const pending = pendingRecipeSavesRef.current.get(id);
        if (pending) {
          window.clearTimeout(pending.timerId);
          pendingRecipeSavesRef.current.delete(id);
        }
      }
      setDevelopHistory((prev) =>
        recordDevelopHistory(prev, {
          label: `Paste Recipe (${changes.length})`,
          timestampMs: Date.now(),
          changes,
        }),
      );
      setSession((prev) => (prev ? { ...prev, clusters: nextClusters } : prev));
      notifyAction(`Pasted recipe onto ${pastedIds.length} photo(s)`);
      const batchItems = pastedIds.map((photoId) => ({ photoId, recipe: copiedRecipe }));
      if (gateway.savePhotoDevelopRecipesBatch) {
        void gateway
          .savePhotoDevelopRecipesBatch(batchItems, activeDevelopPhoto?.id ?? null)
          .then((hist) => {
            if (hist) setLiveHistogram(hist);
            if (thumbnailPreviewModeRef.current !== 'fast') {
              request32BitThumbnails(thumbnailPreviewModeRef.current, pastedIds);
            }
          })
          .catch(() => { });
      } else {
        for (const item of batchItems) {
          void gateway
            .savePhotoDevelopRecipe(item.photoId, item.recipe)
            .then(() => {
              if (thumbnailPreviewModeRef.current !== 'fast') {
                request32BitThumbnails(thumbnailPreviewModeRef.current, [item.photoId]);
              }
            })
            .catch(() => { });
        }
      }
    }
  }, [
    copiedRecipe,
    developPhotos,
    selectedPhotoIds,
    activeDevelopPhoto,
    gateway,
    request32BitThumbnails,
    notifyAction,
  ]);

  const mergeClustersById = useCallback(
    async (sourceClusterId: string, targetClusterId: string) => {
      if (sourceClusterId === targetClusterId) return;
      const sourceName =
        session?.clusters.find((c) => c.anchor.id === sourceClusterId)?.anchor.subjectName ??
        sourceClusterId;
      try {
        const updated = await gateway.mergeClusters(sourceClusterId, targetClusterId);
        if (updated) {
          applyLoadedSession(updated, targetClusterId);
          const targetName =
            updated.clusters.find((c) => c.anchor.id === targetClusterId)?.anchor.subjectName ??
            targetClusterId;
          notifyAction(`Merged ${sourceName} into ${targetName}`);
        }
      } catch (err) {
        notifyAction(`Merge failed: ${String(err)}`);
      }
    },
    [session, gateway, applyLoadedSession, notifyAction],
  );

  const mergeActiveClusterInto = useCallback(
    async (targetClusterId: string) => {
      if (!activeCluster || activeCluster.anchor.id === targetClusterId) return;
      await mergeClustersById(activeCluster.anchor.id, targetClusterId);
    },
    [activeCluster, mergeClustersById],
  );

  const movePhotosToCluster = useCallback(
    async (photoIds: readonly string[], targetClusterId: string) => {
      if (photoIds.length === 0) return;
      try {
        const updated = await gateway.movePhotosToCluster(photoIds, targetClusterId);
        if (updated) {
          applyLoadedSession(updated, targetClusterId);
          const targetName =
            updated.clusters.find((c) => c.anchor.id === targetClusterId)?.anchor.subjectName ??
            targetClusterId;
          notifyAction(`Matched ${photoIds.length} photo(s) into ${targetName}`);
        }
      } catch (err) {
        notifyAction(`Manual match failed: ${String(err)}`);
      }
    },
    [gateway, applyLoadedSession, notifyAction],
  );

  const splitPhotoIdsToNewCluster = useCallback(
    async (photoIds: readonly string[]) => {
      if (photoIds.length === 0) {
        notifyAction('Select at least 1 photo to split into a new subject');
        return;
      }
      try {
        const updated = await gateway.splitPhotosToNewCluster(photoIds);
        if (updated) {
          const newest = updated.clusters[updated.clusters.length - 1];
          applyLoadedSession(updated, newest?.anchor.id);
          notifyAction(
            `Split ${photoIds.length} photo(s) into ${newest?.anchor.subjectName ?? 'new subject'}`,
          );
        }
      } catch (err) {
        notifyAction(`Split failed: ${String(err)}`);
      }
    },
    [gateway, applyLoadedSession, notifyAction],
  );

  const splitSelectedPhotos = useCallback(async () => {
    if (
      activeCluster &&
      activeCluster.photos.length > 0 &&
      activeCluster.photos.every((p) => selectedPhotoIds.has(p.id))
    ) {
      notifyAction('Uncheck at least 1 photo to keep in this subject before splitting');
      return;
    }
    await splitPhotoIdsToNewCluster(Array.from(selectedPhotoIds));
  }, [activeCluster, selectedPhotoIds, splitPhotoIdsToNewCluster, notifyAction]);

  const autoLevelActiveCluster = useCallback(async () => {
    if (!activeCluster) return;
    try {
      const updated = await gateway.autoLevelClusterEyes(activeCluster.anchor.id);
      if (updated) {
        applyLoadedSession(updated, activeCluster.anchor.id);
        if (thumbnailPreviewModeRef.current !== 'fast') {
          request32BitThumbnails(
            thumbnailPreviewModeRef.current,
            activeCluster.photos.map((p) => p.id),
          );
        }
        notifyAction(
          `Auto-leveled eye horizon across ${activeCluster.anchor.subjectName}`,
        );
      }
    } catch (err) {
      notifyAction(`Auto-level failed: ${String(err)}`);
    }
  }, [activeCluster, gateway, applyLoadedSession, request32BitThumbnails, notifyAction]);

  const resolveAmbiguousReview = useCallback(
    (photoId: string, decision: 'approve' | 'split') => {
      const targetItem = ambiguousReviewQueue.find((item) => item.photo.id === photoId);
      setResolvedAmbiguousIds((prev) => new Set(prev).add(photoId));

      if (targetItem?.source === 'cluster_low_confidence') {
        if (decision === 'split') {
          void gateway
            .splitPhotosToNewCluster([photoId])
            .then((updated) => {
              if (updated) {
                const newest = updated.clusters[updated.clusters.length - 1];
                setSession(updated);
                notifyAction(
                  `Split frame into ${newest?.anchor.subjectName ?? 'a new Person cluster'}`,
                );
              }
            })
            .catch((err) => notifyAction(`Split failed: ${String(err)}`));
        } else {
          notifyAction('Approved match for this subject');
        }
        return;
      }

      void gateway
        .resolveAmbiguousMatch(photoId, decision)
        .then((updated) => {
          if (updated) {
            setSession(updated);
          }
        })
        .catch((err) => notifyAction(`Review update error: ${String(err)}`));

      setSession((prev) => {
        if (!prev) return prev;
        if (decision === 'approve') {
          return {
            ...prev,
            telemetry: {
              ...prev.telemetry,
              needsReviewCount: Math.max(0, prev.telemetry.needsReviewCount - 1),
            },
            clusters: prev.clusters.map((cluster) => ({
              ...cluster,
              anchor: {
                ...cluster.anchor,
                ambiguousCount: cluster.photos.some((p) => p.id === photoId)
                  ? Math.max(0, cluster.anchor.ambiguousCount - 1)
                  : cluster.anchor.ambiguousCount,
              },
              photos: cluster.photos.map((p) =>
                p.id === photoId ? { ...p, sfaceCosineSimilarity: 0.965 } : p,
              ),
            })),
          };
        }

        let extractedPhoto: PhotoAsset | null = null;
        const trimmedClusters = prev.clusters.map((cluster) => {
          const found = cluster.photos.find((p) => p.id === photoId);
          if (found) extractedPhoto = found;
          return {
            ...cluster,
            anchor: {
              ...cluster.anchor,
              ambiguousCount: found
                ? Math.max(0, cluster.anchor.ambiguousCount - 1)
                : cluster.anchor.ambiguousCount,
            },
            photos: cluster.photos.filter((p) => p.id !== photoId),
          };
        });

        if (!extractedPhoto) return prev;
        const splitPhoto: PhotoAsset = extractedPhoto;
        const nextIdx = prev.clusters.length + 1;
        const newClusterId = `person-${String(nextIdx).padStart(3, '0')}`;
        const newCluster: SubjectCluster = {
          anchor: {
            id: newClusterId,
            slateCode: 'NO SLATE',
            subjectName: `Person ${String(nextIdx).padStart(3, '0')}`,
            ocrExtractedText: 'Face-Only Cluster',
            ocrConfidencePct: 100,
            anchorPhotoId: splitPhoto.id,
            thumbnailUrl: splitPhoto.imageUrl,
            capturedAtLabel: '1 photo',
            sessionTimeRange: 'Manual Split',
            cameraLabel: splitPhoto.exif.cameraModel,
            lensLabel: splitPhoto.exif.lensModel,
            wbCalibrated: false,
            wbCalibrationLabel: 'Face-First Mode',
            reportedTotalExposures: 1,
            ambiguousCount: 0,
          },
          photos: [{ ...splitPhoto, clusterId: newClusterId, sfaceCosineSimilarity: 1.0 }],
        };

        return {
          ...prev,
          telemetry: {
            ...prev.telemetry,
            needsReviewCount: Math.max(0, prev.telemetry.needsReviewCount - 1),
          },
          clusters: [...trimmedClusters, newCluster],
        };
      });

      notifyAction(
        decision === 'approve'
          ? 'Approved match and updated subject cluster'
          : 'Split frame into a new Person cluster',
      );
    },
    [ambiguousReviewQueue, gateway, notifyAction],
  );

  const batchResolveAmbiguousReview = useCallback(
    async (photoIds: readonly string[], decision: 'approve' | 'split_together') => {
      if (photoIds.length === 0) return;
      const uniqueIds = Array.from(new Set(photoIds));
      setResolvedAmbiguousIds((prev) => {
        const next = new Set(prev);
        for (const id of uniqueIds) next.add(id);
        return next;
      });

      if (decision === 'split_together') {
        try {
          const updated = await gateway.splitPhotosToNewCluster(uniqueIds);
          if (updated) {
            const newest = updated.clusters[updated.clusters.length - 1];
            setSession(updated);
            notifyAction(
              `Split ${uniqueIds.length} photo${uniqueIds.length === 1 ? '' : 's'} into ${newest?.anchor.subjectName ?? '1 new Person'}`,
            );
          }
        } catch (err) {
          notifyAction(`Batch split failed: ${String(err)}`);
        }
        return;
      }

      const backendQueueIds = uniqueIds.filter((id) => {
        const item = ambiguousReviewQueue.find((entry) => entry.photo.id === id);
        return item?.source === 'backend_queue';
      });
      let latestSession: StudioSessionSnapshot | null = null;
      for (const id of backendQueueIds) {
        try {
          const updated = await gateway.resolveAmbiguousMatch(id, 'approve');
          if (updated) latestSession = updated;
        } catch (err) {
          notifyAction(`Review approve error: ${String(err)}`);
        }
      }
      if (latestSession) {
        setSession(latestSession);
      }
      notifyAction(
        `Approved ${uniqueIds.length} match${uniqueIds.length === 1 ? '' : 'es'}`,
      );
    },
    [ambiguousReviewQueue, gateway, notifyAction],
  );

  const renameActiveSlate = useCallback(
    async (clusterId: string, nextName: string) => {
      const updatedFromBackend = await gateway.renameSlateSubject(clusterId, nextName);
      if (updatedFromBackend) {
        setSession(updatedFromBackend);
      } else {
        setSession((prev) =>
          prev
            ? {
              ...prev,
              clusters: prev.clusters.map((cluster) =>
                cluster.anchor.id === clusterId
                  ? {
                    ...cluster,
                    anchor: {
                      ...cluster.anchor,
                      subjectName: nextName.trim(),
                      ocrExtractedText: nextName.trim(),
                    },
                  }
                  : cluster,
              ),
            }
            : prev,
        );
      }
      notifyAction(`Renamed subject to "${nextName.trim()}"`);
    },
    [gateway, notifyAction],
  );

  const setExcludeSlateFromClientPack = useCallback(
    (exclude: boolean) => updatePreferences({ excludeSlateFromClientPack: exclude }),
    [updatePreferences],
  );

  const setOutputPattern = useCallback(
    (pattern: string) => updatePreferences({ defaultNamingPattern: pattern }),
    [updatePreferences],
  );

  const syncSlateWhiteBalanceForActiveCluster = useCallback(() => {
    if (!activeCluster) return;
    const targetIds = activeCluster.photos.map((p) => p.id);
    const refPhotoId = activeCluster.anchor.anchorPhotoId;

    void gateway
      .calibrateWhiteBalanceFromSlate(refPhotoId, targetIds)
      .then((calibrated) => {
        const wb = calibrated ?? { temperatureKelvin: 5550, tint: 3 };
        const changes: PhotoRecipeDiff[] = activeCluster.photos.map((photo) => ({
          photoId: photo.id,
          before: photo.developRecipe,
          after: { ...photo.developRecipe, whiteBalance: wb },
        }));
        if (changes.length > 0) {
          setDevelopHistory((prev) =>
            recordDevelopHistory(prev, {
              label: 'Slate Auto-WB',
              timestampMs: Date.now(),
              changes,
            }),
          );
        }
        setSession((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            clusters: prev.clusters.map((cluster) =>
              cluster.anchor.id === activeCluster.anchor.id
                ? {
                  ...cluster,
                  anchor: {
                    ...cluster.anchor,
                    wbCalibrated: true,
                    wbCalibrationLabel: `WB ${wb.temperatureKelvin}K (${wb.tint >= 0 ? '+' : ''}${wb.tint})`,
                  },
                  photos: cluster.photos.map((photo) => ({
                    ...photo,
                    developRecipe: {
                      ...photo.developRecipe,
                      whiteBalance: wb,
                    },
                  })),
                }
                : cluster,
            ),
          };
        });
        if (thumbnailPreviewModeRef.current !== 'fast') {
          request32BitThumbnails(thumbnailPreviewModeRef.current, targetIds);
        }
        notifyAction(
          `Synced neutral White Balance (${wb.temperatureKelvin}K) across ${activeCluster.anchor.subjectName}`,
        );
      })
      .catch((err) => notifyAction(`Auto WB calibration failed: ${String(err)}`));
  }, [activeCluster, gateway, request32BitThumbnails, notifyAction]);

  const canUndo = session?.history?.canUndo ?? false;
  const canRedo = session?.history?.canRedo ?? false;
  const undoLabel = session?.history?.undoLabel ?? null;
  const redoLabel = session?.history?.redoLabel ?? null;

  const canUndoDevelop = developHistory.undoStack.length > 0;
  const canRedoDevelop = developHistory.redoStack.length > 0;
  const undoDevelopLabel =
    developHistory.undoStack[developHistory.undoStack.length - 1]?.label ?? null;
  const redoDevelopLabel =
    developHistory.redoStack[developHistory.redoStack.length - 1]?.label ?? null;

  const undoDevelopAction = useCallback(() => {
    const { nextStack, entry } = undoDevelopHistory(developHistoryRef.current);
    if (!entry) return;
    setDevelopHistory(nextStack);
    applyRecipeChangesToSession(entry.changes, 'before');
    notifyAction(`Undid: ${entry.label}`);
  }, [applyRecipeChangesToSession, notifyAction]);

  const redoDevelopAction = useCallback(() => {
    const { nextStack, entry } = redoDevelopHistory(developHistoryRef.current);
    if (!entry) return;
    setDevelopHistory(nextStack);
    applyRecipeChangesToSession(entry.changes, 'after');
    notifyAction(`Redid: ${entry.label}`);
  }, [applyRecipeChangesToSession, notifyAction]);

  const undoOrganizationAction = useCallback(async () => {
    if (!canUndo || isScanningFolder) return;
    const undoneAction = undoLabel ?? 'last organization change';
    try {
      const updated = await gateway.undoOrganizationAction();
      if (updated) {
        applyLoadedSession(updated, activeClusterId);
        notifyAction(`Undid: ${undoneAction}`);
      }
    } catch (err) {
      notifyAction(`Undo failed: ${String(err)}`);
    }
  }, [canUndo, isScanningFolder, undoLabel, gateway, applyLoadedSession, activeClusterId, notifyAction]);

  const redoOrganizationAction = useCallback(async () => {
    if (!canRedo || isScanningFolder) return;
    const redoneAction = redoLabel ?? 'organization change';
    try {
      const updated = await gateway.redoOrganizationAction();
      if (updated) {
        applyLoadedSession(updated, activeClusterId);
        notifyAction(`Redid: ${redoneAction}`);
      }
    } catch (err) {
      notifyAction(`Redo failed: ${String(err)}`);
    }
  }, [canRedo, isScanningFolder, redoLabel, gateway, applyLoadedSession, activeClusterId, notifyAction]);

  // Global Ctrl+O (Open Project) & Ctrl+, (Open Settings) shortcuts for native desktop ergonomics
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return;
      const key = event.key.toLowerCase();
      if (key !== 'o' && key !== ',') return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      ) {
        return;
      }
      event.preventDefault();
      if (key === ',') {
        toggleSettings();
        return;
      }
      if (!isScanningFolder) {
        void openShootFolder();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isScanningFolder, openShootFolder, toggleSettings]);

  // Context-aware global Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y) Undo/Redo:
  // - Inside Studio (`develop`), undoes/redoes non-destructive color & curve edits.
  // - Inside Face Organizer / Slates / Review, undoes/redoes subject organization edits.
  useEffect(() => {
    if (!session || settingsOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
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
      const key = event.key.toLowerCase();
      if (key === 'z' && !event.shiftKey) {
        if (activeWorkspace === 'develop') {
          if (!canUndoDevelop) return;
          event.preventDefault();
          undoDevelopAction();
        } else {
          if (!canUndo) return;
          event.preventDefault();
          void undoOrganizationAction();
        }
      } else if ((key === 'z' && event.shiftKey) || (key === 'y' && !event.shiftKey)) {
        if (activeWorkspace === 'develop') {
          if (!canRedoDevelop) return;
          event.preventDefault();
          redoDevelopAction();
        } else {
          if (!canRedo) return;
          event.preventDefault();
          void redoOrganizationAction();
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    session,
    settingsOpen,
    activeWorkspace,
    canUndo,
    canRedo,
    canUndoDevelop,
    canRedoDevelop,
    undoOrganizationAction,
    redoOrganizationAction,
    undoDevelopAction,
    redoDevelopAction,
  ]);

  const pickExportDirectory = useCallback(async (): Promise<string | null> => {
    if (!isDesktopRuntime || !gateway.pickExportDirectory) {
      notifyAction('Choosing local folders requires Kroma Desktop (`npm run tauri:dev`)');
      return null;
    }
    try {
      return await gateway.pickExportDirectory();
    } catch (err) {
      notifyAction(`Folder picker error: ${String(err)}`);
      return null;
    }
  }, [isDesktopRuntime, gateway, notifyAction]);

  const estimateExportSize = useCallback(
    async (request: ExportSizeEstimateRequest): Promise<MeasuredExportSize | null> => {
      if (!isDesktopRuntime || !gateway.estimateExportSize) {
        return null;
      }
      try {
        return await gateway.estimateExportSize(request);
      } catch {
        return null;
      }
    },
    [isDesktopRuntime, gateway],
  );

  const runBatchExport = useCallback(
    async (request: BatchExportRequest): Promise<BatchExportResult | null> => {
      await flushPendingRecipeSaves();
      if (isDesktopRuntime) {
        setExportJob({
          isExporting: true,
          completed: 0,
          total: 0,
          percent: 0,
          message: 'Preparing export…',
          completedCount: null,
          failedCount: 0,
          outputRoot: null,
          error: null,
        });
      }
      try {
        const result = await gateway.exportShoot(request);
        if (result) {
          const totalProcessed = result.organizedCount + result.failedCount;
          setExportJob({
            isExporting: false,
            completed: result.organizedCount,
            total: totalProcessed,
            percent: 100,
            message: 'Export complete',
            completedCount: result.organizedCount,
            failedCount: result.failedCount,
            outputRoot: result.outputRoot ?? null,
            error: null,
          });
          notifyAction(
            `Exported ${result.organizedCount} file(s) to ${result.outputRoot ?? 'destination'}${result.failedCount > 0 ? ` (${result.failedCount} failed)` : ''}`,
          );
        } else if (isDesktopRuntime) {
          setExportJob(null);
        }
        return result;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setExportJob({
          isExporting: false,
          completed: 0,
          total: 0,
          percent: 0,
          message: 'Export failed',
          completedCount: null,
          failedCount: 0,
          outputRoot: null,
          error: msg,
        });
        notifyAction(`Batch export failed: ${msg}`);
        throw err;
      } finally {
        setScanProgress((prev) => (prev?.phase === 'export' ? null : prev));
      }
    },
    [flushPendingRecipeSaves, isDesktopRuntime, gateway, notifyAction],
  );

  useEffect(() => {
    if (!isDesktopRuntime || !gateway.subscribeWindowMaximized) return;
    return gateway.subscribeWindowMaximized(setIsWindowMaximized);
  }, [isDesktopRuntime, gateway]);

  const minimizeWindow = useCallback(async (): Promise<void> => {
    if (isDesktopRuntime && gateway.minimizeWindow) {
      await gateway.minimizeWindow();
      return;
    }
    notifyAction('Minimize is available in Kroma Desktop');
  }, [isDesktopRuntime, gateway, notifyAction]);

  const toggleMaximizeWindow = useCallback(async (): Promise<void> => {
    if (isDesktopRuntime && gateway.toggleMaximizeWindow) {
      const maximized = await gateway.toggleMaximizeWindow();
      setIsWindowMaximized(maximized);
      return;
    }
    notifyAction('Maximize / Restore is available in Kroma Desktop');
  }, [isDesktopRuntime, gateway, notifyAction]);

  const exitApp = useCallback(async (): Promise<void> => {
    await flushPendingRecipeSaves();
    if (isDesktopRuntime && gateway.exitApp) {
      await gateway.exitApp();
      return;
    }
    notifyAction('Exit App is available in Kroma Desktop');
  }, [flushPendingRecipeSaves, isDesktopRuntime, gateway, notifyAction]);

  const sessionWithSortedClusters = useMemo(() => {
    if (!session) return null;
    return { ...session, clusters: sortedClusters };
  }, [session, sortedClusters]);

  const value: StudioContextValue = {
    isDesktopRuntime,
    isBootReady,
    bootPhase,
    modelManifestStatus,
    modelDownloadProgress,
    modelDownloadError,
    startupUpdate,
    isScanningFolder,
    isScanningOcr,
    slateOcrResults,
    scanProgress,
    thumbnailRenderProgress,
    developViewportStatus,
    exportJob,
    runtimeInfo,
    session: sessionWithSortedClusters,
    preferences,
    recentProjects,
    activeWorkspace,
    activeCluster,
    developSourceMode: effectiveDevelopSourceMode,
    activeDevelopFolderId,
    developPhotos,
    activeDevelopScopeLabel,
    activeDevelopPhoto,
    liveHistogram,
    visibleClusterPhotos,
    ambiguousReviewQueue,
    selectedPhotoIds,
    copiedRecipe,
    activeDirectoryId,
    activeVirtualFolderId,
    activeVirtualFolderAllowedIds,
    folderCreateNonce,
    smartBin,
    searchQuery,
    sfaceThreshold,
    lastClusteredThreshold,
    reviewWarningThreshold,
    approvedReviewPhotoIds: resolvedAmbiguousIds,
    useTemporalAssist,
    lastClusteredTemporalAssist,
    manualMatchSplitOpen,
    studioFocusMode,
    displayMode,
    anchorSortMode,
    showLandmarksOverlay,
    outputRule,
    statusBanner,
    canUndo,
    canRedo,
    undoLabel,
    redoLabel,
    canUndoDevelop,
    canRedoDevelop,
    undoDevelopLabel,
    redoDevelopLabel,
    isWindowMaximized,
    settingsOpen,
    retryModelDownload,
    continueWithoutModels,
    dismissStartupUpdate,
    setActiveWorkspace,
    openSettings,
    closeSettings,
    toggleSettings,
    updatePreferences,
    savePreferences,
    resetPreferences,
    pickOnnxModelFile,
    inspectOnnxModel,
    applyCustomAiModel,
    setDevelopSourceMode,
    selectDevelopFolder,
    openShootFolder,
    openRecentShootFolder,
    removeRecentProjectEntry,
    clearRecentProjects,
    togglePinRecentProjectEntry,
    getStorageCacheSummary,
    pickCacheDirectory,
    clearStorageCache,
    runFaceOrganizer,
    scanProjectSlatesOcr,
    cancelSlateOcrScan,
    scanSinglePhotoOcr,
    applySlateOcrRenames,
    openPhotoInDevelop,
    selectCluster,
    selectNextCluster,
    selectNextSingletonCluster,
    setManualMatchSplitOpen,
    setStudioFocusMode,
    toggleStudioFocusMode,
    setDevelopViewportStatus,
    setActiveDevelopPhotoId: handleSetActiveDevelopPhotoId,
    toggleDirectoryFilter,
    setActiveVirtualFolderId,
    requestNewVirtualFolder,
    createVirtualFolder,
    renameVirtualFolder,
    moveVirtualFolder,
    deleteVirtualFolder,
    assignPhotosToVirtualFolder,
    assignClustersToVirtualFolder,
    setSmartBin,
    setSearchQuery,
    setSfaceThreshold,
    setReviewWarningThreshold,
    setUseTemporalAssist,
    setDisplayMode,
    setAnchorSortMode,
    toggleLandmarksOverlay,
    togglePhotoSelection,
    selectAllVisiblePhotos,
    peekDevelopWorkingBuffer,
    loadDevelopWorkingBuffer,
    generateAiMask,
    updatePhotoRecipe,
    syncRecipeFromPhoto,
    applyRelativeNudgeToClusterSelection,
    copyActiveRecipe,
    pasteRecipeToClusterSelection,
    mergeActiveClusterInto,
    mergeClustersById,
    movePhotosToCluster,
    splitSelectedPhotos,
    splitPhotoIdsToNewCluster,
    autoLevelActiveCluster,
    resolveAmbiguousReview,
    batchResolveAmbiguousReview,
    renameActiveSlate,
    setExcludeSlateFromClientPack,
    setOutputPattern,
    syncSlateWhiteBalanceForActiveCluster,
    undoOrganizationAction,
    redoOrganizationAction,
    undoDevelopAction,
    redoDevelopAction,
    pickExportDirectory,
    estimateExportSize,
    runBatchExport,
    minimizeWindow,
    toggleMaximizeWindow,
    exitApp,
    notifyAction,
  };

  return <StudioContext.Provider value={value}>{children}</StudioContext.Provider>;
}

export function useStudio(): StudioContextValue {
  const ctx = useContext(StudioContext);
  if (!ctx) {
    throw new Error('useStudio must be used inside StudioProvider');
  }
  return ctx;
}
