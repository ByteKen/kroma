import type {
  DevelopRecipe,
  DevelopSyncMask,
  RelativeDevelopDelta,
  WhiteBalanceSettings,
} from '../../domain/develop';
import type {
  DevelopWorkingBuffer,
  WorkingBufferPrecision,
} from '../../domain/developWorkingBuffer';
import type { SubjectCluster } from '../../domain/identity';
import type { AiMaskKind, GeneratedAiMaskResult } from '../../domain/masking';
import type { OutputNamingRule } from '../../domain/organization';
import type { PhotoAsset } from '../../domain/photo';
import type {
  ClusterSlateOcrResult,
  PhotoOcrResult,
  SlateOcrScanOptions,
  SlateRenameApplyItem,
} from '../../domain/slateOcr';
import type { VirtualFolder } from '../../domain/virtualFolder';

export interface IngestedDirectorySummary {
  readonly id: string;
  readonly name: string;
  readonly rawCount: number;
}

export interface EngineTelemetrySnapshot {
  readonly sessionName: string;
  readonly projectName: string;
  readonly totalPhotos: number;
  readonly indexedPhotos: number;
  readonly indexElapsedSeconds: number;
  readonly indexErrorCount: number;
  readonly slatesReadCount: number;
  readonly clusteredFacesCount: number;
  readonly needsReviewCount: number;
  readonly unassignedFacesCount: number;
  readonly focusDropsCount: number;
  readonly ocrMatchedCount: number;
  readonly throughputFps: number;
  readonly modelSummary: string;
  readonly runtimeBackendLabel: string;
  readonly vramUsedMb: number;
  readonly vramTotalMb: number;
  readonly engineLoadPct: number;
}

export type AiModelRole = 'detector' | 'recognizer' | 'segmenter';
export type ModelCompatibility = 'compatible' | 'warning' | 'incompatible';

export interface ModelInspectionResult {
  readonly compatibility: ModelCompatibility;
  readonly summary: string;
  readonly warnings: readonly string[];
  readonly detectedLabel: string;
  readonly inputShape: string;
  readonly outputShape: string;
  readonly embeddingDim: number | null;
  readonly fileSizeBytes: number;
}

export interface RuntimeInfo {
  readonly yunetModelPath: string | null;
  readonly sfaceModelPath: string | null;
  readonly detectorModelLabel?: string;
  readonly recognitionModelLabel?: string;
  readonly defaultYunetModelPath?: string | null;
  readonly defaultSfaceModelPath?: string | null;
  readonly defaultDetectorModelLabel?: string;
  readonly defaultRecognitionModelLabel?: string;
  readonly isCustomYunet?: boolean;
  readonly isCustomSface?: boolean;
  readonly modelsReady: boolean;
  readonly ocrModelsReady?: boolean;
  readonly ocrDetModelLabel?: string;
  readonly ocrRecModelLabel?: string;
  readonly ocrDetModelPath?: string | null;
  readonly ocrRecModelPath?: string | null;
  readonly segmenterModelPath?: string | null;
  readonly segmenterModelLabel?: string;
  readonly defaultSegmenterModelPath?: string | null;
  readonly defaultSegmenterModelLabel?: string;
  readonly isCustomSegmenter?: boolean;
  readonly segmenterModelsReady?: boolean;
  readonly supportedRawCount: number;
  readonly backendLabel: string;
  readonly developGpuBackendLabel?: string;
  readonly developGpuAdapterName?: string | null;
  readonly appVersion?: string;
  readonly gpuAvailable?: boolean;
  readonly gpuAccelerationEnabled?: boolean;
}

export type ClearCacheTarget = 'previews' | 'face_cache' | 'all';

export interface StorageCacheSummary {
  readonly cacheDirPath: string;
  readonly isCustomCacheDir?: boolean;
  readonly previewCacheBytes: number;
  readonly previewFileCount: number;
  readonly faceCacheBytes: number;
  readonly faceCacheFileCount: number;
}

export interface ClearCacheResult {
  readonly removedFiles: number;
  readonly freedBytes: number;
}

export type ModelInstallStatus = 'ready' | 'missing' | 'outdated';

export interface ModelManifestEntryStatus {
  readonly id: string;
  readonly name: string;
  readonly category: string;
  readonly version: string;
  readonly installedVersion: string | null;
  readonly relativePath: string;
  readonly sizeBytes: number;
  readonly required: boolean;
  readonly status: ModelInstallStatus;
  readonly resolvedPath: string | null;
}

export interface ModelManifestStatus {
  readonly schemaVersion: number;
  readonly releaseTag: string;
  readonly modelsDir: string;
  readonly allReady: boolean;
  readonly missingCount: number;
  readonly outdatedCount: number;
  readonly pendingDownloadBytes: number;
  readonly entries: readonly ModelManifestEntryStatus[];
}

export interface ModelDownloadProgress {
  readonly phase: 'downloading' | 'completed';
  readonly modelId: string;
  readonly modelName: string;
  readonly index: number;
  readonly totalCount: number;
  readonly downloadedBytes: number;
  readonly totalBytes: number;
  readonly overallDownloadedBytes: number;
  readonly overallTotalBytes: number;
  readonly bytesPerSecond: number;
}

export interface ModelDownloadOutcome {
  readonly manifest: ModelManifestStatus;
  readonly runtime: RuntimeInfo;
}

export interface LiveRgbHistogram {
  readonly red: readonly number[];
  readonly green: readonly number[];
  readonly blue: readonly number[];
}

export interface ScanProgressUpdate {
  readonly phase: 'loading' | 'clustering' | 'ocr' | 'export';
  readonly completed: number;
  readonly total: number;
  readonly message: string;
}

export interface PreviewPatchItem {
  readonly photoId: string;
  readonly imageUrl: string;
  readonly capturedAtEpochSec: number | null;
  readonly capturedAtLabel: string | null;
}

export interface PreviewsReadyBatch {
  readonly updates: readonly PreviewPatchItem[];
  readonly completed: number;
  readonly total: number;
}

export interface PendingAmbiguousEntry {
  readonly photo: PhotoAsset;
  readonly candidateClusterId: string;
  readonly reason: string;
}

export interface OrganizationHistoryState {
  readonly canUndo: boolean;
  readonly canRedo: boolean;
  readonly undoLabel: string | null;
  readonly redoLabel: string | null;
}

export interface StudioSessionSnapshot {
  readonly isClustered: boolean;
  readonly folderPath?: string;
  readonly telemetry: EngineTelemetrySnapshot;
  readonly directories: readonly IngestedDirectorySummary[];
  readonly virtualFolders: readonly VirtualFolder[];
  readonly clusters: readonly SubjectCluster[];
  readonly ambiguousQueue?: readonly PendingAmbiguousEntry[];
  readonly outputRule: OutputNamingRule;
  readonly history?: OrganizationHistoryState;
}

export interface BatchExportRequest {
  readonly outputRoot?: string;
  readonly namingPattern: string;
  readonly excludeSlatePhoto: boolean;
  readonly archiveExcludedSlates: boolean;
  readonly mode: 'develop_and_raw' | 'develop' | 'copy' | 'move' | 'hardlink';
  readonly format: 'jpeg' | 'png' | 'tiff';
  readonly quality: number;
  readonly photoIds?: readonly string[];
}

export interface BatchExportResult {
  readonly organizedCount: number;
  readonly failedCount: number;
  readonly outputRoot: string | null;
  readonly manifestPath: string | null;
}

export interface ExportSizeEstimateRequest {
  readonly excludeSlatePhoto: boolean;
  readonly archiveExcludedSlates: boolean;
  readonly mode: 'develop_and_raw' | 'develop' | 'copy' | 'move' | 'hardlink';
  readonly format: 'jpeg' | 'png' | 'tiff';
  readonly quality: number;
  readonly photoIds?: readonly string[];
}

export interface MeasuredExportSize {
  readonly extraDiskBytes: number;
  readonly editedBytes: number;
  readonly originalsBytes: number;
  readonly totalSourceBytes: number;
  readonly fileCount: number;
  readonly sampledPhotoCount: number;
  readonly isExact: boolean;
}

/**
 * Coarse inward-owned gateway boundary (PoEAA Remote Facade).
 * Supports Edit-First (`openShootFolder`) followed by on-demand Face Organization (`runFaceClustering`),
 * and exports both Edited and original RAW files.
 */
export interface StudioGateway {
  isDesktopRuntime(): boolean;
  subscribeScanProgress(listener: (progress: ScanProgressUpdate) => void): () => void;
  subscribePreviewsReady?(listener: (batch: PreviewsReadyBatch) => void): () => void;
  subscribeModelDownloadProgress?(listener: (progress: ModelDownloadProgress) => void): () => void;
  checkRuntimeInfo(): Promise<RuntimeInfo | null>;
  checkModelsManifest?(): Promise<ModelManifestStatus | null>;
  downloadMissingModels?(): Promise<ModelDownloadOutcome | null>;
  cancelModelDownloads?(): Promise<void>;
  setGpuAcceleration?(enabled: boolean): Promise<RuntimeInfo | null>;
  getStorageCacheSummary?(): Promise<StorageCacheSummary | null>;
  pickCacheDirectory?(): Promise<string | null>;
  setStorageCacheDir?(cacheDir: string | null): Promise<StorageCacheSummary | null>;
  clearStorageCache?(target: ClearCacheTarget): Promise<ClearCacheResult | null>;
  pickOnnxModelFile?(): Promise<string | null>;
  inspectOnnxModel?(modelPath: string, role: AiModelRole): Promise<ModelInspectionResult | null>;
  setCustomAiModels?(
    yunetPath: string | null,
    recognizerPath: string | null,
    segmenterPath?: string | null,
  ): Promise<RuntimeInfo | null>;
  loadSession(): Promise<StudioSessionSnapshot | null>;
  openShootFolder(): Promise<StudioSessionSnapshot | null>;
  openShootFolderAtPath(folderPath: string): Promise<StudioSessionSnapshot | null>;
  runFaceClustering(
    sfaceThreshold: number,
    useTemporalAssist?: boolean,
  ): Promise<StudioSessionSnapshot | null>;
  scanProjectSlatesOcr?(options?: SlateOcrScanOptions): Promise<readonly ClusterSlateOcrResult[]>;
  cancelSlateOcrScan?(): Promise<void>;
  scanSinglePhotoOcr?(photoId: string): Promise<PhotoOcrResult | null>;
  applySlateOcrRenames?(
    items: readonly SlateRenameApplyItem[],
  ): Promise<StudioSessionSnapshot | null>;
  renameSlateSubject(clusterId: string, nextSubjectName: string): Promise<StudioSessionSnapshot | void>;
  mergeClusters(
    sourceClusterId: string,
    targetClusterId: string,
  ): Promise<StudioSessionSnapshot | null>;
  splitPhotosToNewCluster(photoIds: readonly string[]): Promise<StudioSessionSnapshot | null>;
  movePhotosToCluster(
    photoIds: readonly string[],
    targetClusterId: string,
  ): Promise<StudioSessionSnapshot | null>;
  createVirtualFolder?(
    name: string,
    parentId?: string | null,
  ): Promise<StudioSessionSnapshot | null>;
  renameVirtualFolder?(
    folderId: string,
    nextName: string,
  ): Promise<StudioSessionSnapshot | null>;
  moveVirtualFolder?(
    folderId: string,
    nextParentId: string | null,
  ): Promise<StudioSessionSnapshot | null>;
  deleteVirtualFolder?(folderId: string): Promise<StudioSessionSnapshot | null>;
  assignPhotosToVirtualFolder?(
    photoIds: readonly string[],
    folderId: string | null,
  ): Promise<StudioSessionSnapshot | null>;
  autoLevelClusterEyes(clusterId: string): Promise<StudioSessionSnapshot | null>;
  loadDevelopWorkingBuffer?(
    photoId: string,
    maxDim?: number,
    precision?: WorkingBufferPrecision,
  ): Promise<DevelopWorkingBuffer | null>;
  generateAiMask?(photoId: string, kind: AiMaskKind): Promise<GeneratedAiMaskResult | null>;
  savePhotoDevelopRecipe(photoId: string, recipe: DevelopRecipe): Promise<LiveRgbHistogram | null>;
  savePhotoDevelopRecipesBatch?(
    items: readonly { readonly photoId: string; readonly recipe: DevelopRecipe }[],
    activePhotoId?: string | null,
  ): Promise<LiveRgbHistogram | null>;
  syncDevelopRecipeToPhotos(
    sourcePhotoId: string,
    targetPhotoIds: readonly string[],
    sourceRecipe: DevelopRecipe,
    mask: DevelopSyncMask,
  ): Promise<void>;
  applyRelativeDevelopDeltaToPhotos(
    photoIds: readonly string[],
    delta: RelativeDevelopDelta,
  ): Promise<void>;
  calibrateWhiteBalanceFromSlate(
    slatePhotoId: string,
    targetPhotoIds: readonly string[],
  ): Promise<WhiteBalanceSettings | null>;
  resolveAmbiguousMatch(
    photoId: string,
    decision: 'approve' | 'split',
  ): Promise<StudioSessionSnapshot | null>;
  undoOrganizationAction(): Promise<StudioSessionSnapshot | null>;
  redoOrganizationAction(): Promise<StudioSessionSnapshot | null>;
  pickExportDirectory?(): Promise<string | null>;
  estimateExportSize?(request: ExportSizeEstimateRequest): Promise<MeasuredExportSize | null>;
  exportShoot(request: BatchExportRequest): Promise<BatchExportResult | null>;
  minimizeWindow?(): Promise<void>;
  toggleMaximizeWindow?(): Promise<boolean>;
  isWindowMaximized?(): Promise<boolean>;
  subscribeWindowMaximized?(listener: (maximized: boolean) => void): () => void;
  exitApp?(): Promise<void>;
}
