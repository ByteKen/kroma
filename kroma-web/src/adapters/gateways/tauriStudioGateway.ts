import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { open as openNativeDialog } from '@tauri-apps/plugin-dialog';
import type {
  AiModelRole,
  BatchExportRequest,
  BatchExportResult,
  ClearCacheResult,
  ClearCacheTarget,
  ExportSizeEstimateRequest,
  LiveRgbHistogram,
  MeasuredExportSize,
  ModelDownloadOutcome,
  ModelDownloadProgress,
  ModelInspectionResult,
  ModelManifestStatus,
  PendingAmbiguousEntry,
  PreviewsReadyBatch,
  RuntimeInfo,
  ScanProgressUpdate,
  StorageCacheSummary,
  StudioGateway,
  StudioSessionSnapshot,
  ThumbnailRenderProgress,
} from '../../application/ports/studioGateway';
import {
  createDefaultDevelopRecipe,
  hydrateDevelopRecipe,
  type DevelopRecipe,
  type DevelopSyncMask,
  type RelativeDevelopDelta,
  type WhiteBalanceSettings,
} from '../../domain/develop';
import {
  parseDevelopWorkingBuffer,
  type DevelopWorkingBuffer,
  type WorkingBufferPrecision,
} from '../../domain/developWorkingBuffer';
import type { SubjectCluster } from '../../domain/identity';
import type { AiMaskKind, GeneratedAiMaskResult } from '../../domain/masking';
import { DEFAULT_OUTPUT_NAMING_RULE } from '../../domain/organization';
import type { PhotoAsset, YuNetFaceOverlay } from '../../domain/photo';
import type { ThumbnailPreviewMode } from '../../domain/preferences';
import type {
  ClusterSlateOcrResult,
  OcrTextLine,
  PhotoOcrResult,
  SlateOcrScanOptions,
  SlateRenameApplyItem,
} from '../../domain/slateOcr';
import type { VirtualFolder } from '../../domain/virtualFolder';

// ── Wire DTOs mirroring kroma-tau/src/dto.rs ──────────────────────────────────

interface DirectorySummaryDto {
  readonly id: string;
  readonly name: string;
  readonly photoCount: number;
}

interface VirtualFolderDto {
  readonly id: string;
  readonly name: string;
  readonly parentId: string | null;
}

interface PhotoDto {
  readonly id: string;
  readonly fileName: string;
  readonly directoryId: string;
  readonly virtualFolderId?: string | null;
  readonly sourcePath: string;
  readonly previewPath: string;
  readonly renderedPreviewPath?: string | null;
  readonly fileSizeBytes?: number;
  readonly isSlateAnchor: boolean;
  readonly clusterId: string | null;
  readonly sfaceCosineSimilarity: number;
  readonly confidence: string;
  readonly capturedAtEpochSec: number | null;
  readonly capturedAtLabel: string | null;
  readonly faceOverlay: YuNetFaceOverlay | null;
  readonly developRecipe: DevelopRecipe;
}

interface ClusterDto {
  readonly id: string;
  readonly displayName: string;
  readonly subjectCode: string | null;
  readonly hasSlate: boolean;
  readonly representativePhotoId: string;
  readonly representativePreviewPath: string;
  readonly photos: readonly PhotoDto[];
}

interface AmbiguousCandidateDto {
  readonly photoId: string;
  readonly fileName: string;
  readonly sourcePath: string;
  readonly previewPath: string;
  readonly candidateClusterId: string;
  readonly cosineSimilarity: number;
  readonly faceOverlay: YuNetFaceOverlay;
}

interface ShootSessionDto {
  readonly folderPath: string;
  readonly projectName: string;
  readonly isClustered: boolean;
  readonly totalPhotos: number;
  readonly scanElapsedMs: number;
  readonly scanErrorCount: number;
  readonly directories: readonly DirectorySummaryDto[];
  readonly virtualFolders?: readonly VirtualFolderDto[];
  readonly clusters: readonly ClusterDto[];
  readonly ambiguousReview: readonly AmbiguousCandidateDto[];
  readonly groupPhotoCount: number;
  readonly noFacePhotoCount: number;
  readonly canUndo: boolean;
  readonly canRedo: boolean;
  readonly undoLabel: string | null;
  readonly redoLabel: string | null;
}

interface HistogramDto {
  readonly red: readonly number[];
  readonly green: readonly number[];
  readonly blue: readonly number[];
  readonly luminance: readonly number[];
  readonly totalPixels: number;
}

interface ExportShootResultDto {
  readonly organizedCount: number;
  readonly failedCount: number;
  readonly manifestPath: string | null;
}

interface PhotoOcrResultDto {
  readonly photoId: string;
  readonly fileName: string;
  readonly previewPath: string;
  readonly proposedName: string | null;
  readonly proposedCode: string | null;
  readonly confidence: number;
  readonly lines: readonly OcrTextLine[];
}

interface ClusterSlateOcrResultDto {
  readonly clusterId: string;
  readonly currentName: string;
  readonly slatePhotoId: string | null;
  readonly slateFileName: string | null;
  readonly slatePreviewPath: string | null;
  readonly proposedName: string | null;
  readonly proposedCode: string | null;
  readonly confidence: number;
  readonly scannedFrameCount: number;
  readonly lines: readonly OcrTextLine[];
}

export function isTauriDesktopEnvironment(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

function toWebViewAssetUrl(localPath: string): string {
  return convertFileSrc(localPath);
}

function downsampleHistogramChannel(raw256: readonly number[], targetBins = 48): number[] {
  if (raw256.length === 0) return Array.from({ length: targetBins }, () => 0.02);
  const buckets = new Array<number>(targetBins).fill(0);
  for (let i = 0; i < raw256.length; i++) {
    const idx = Math.min(targetBins - 1, Math.floor((i / raw256.length) * targetBins));
    buckets[idx] += raw256[i] ?? 0;
  }
  const peak = Math.max(1, ...buckets);
  return buckets.map((v) => Math.max(0.02, Math.min(1, v / peak)));
}

function mapHistogramDto(dto: HistogramDto): LiveRgbHistogram {
  return {
    red: downsampleHistogramChannel(dto.red),
    green: downsampleHistogramChannel(dto.green),
    blue: downsampleHistogramChannel(dto.blue),
  };
}

function mapPhotoDto(dto: PhotoDto): PhotoAsset {
  const ext = dto.fileName.split('.').pop()?.toUpperCase() ?? 'IMG';
  return {
    id: dto.id,
    fileName: dto.fileName,
    directoryId: dto.directoryId,
    virtualFolderId: dto.virtualFolderId ?? null,
    imageUrl: toWebViewAssetUrl(dto.previewPath),
    renderedThumbnailUrl: dto.renderedPreviewPath
      ? toWebViewAssetUrl(dto.renderedPreviewPath)
      : null,
    fileSizeBytes: dto.fileSizeBytes,
    isSlateAnchor: dto.isSlateAnchor,
    clusterId: dto.clusterId,
    sfaceCosineSimilarity: dto.sfaceCosineSimilarity,
    capturedAtEpochSec: dto.capturedAtEpochSec,
    capturedAtLabel: dto.capturedAtLabel,
    sharpnessPct: 95,
    eyeAfLocked: dto.faceOverlay !== null,
    heroSelect: false,
    exif: {
      shutterSpeed: ext,
      aperture: 'Original',
      iso: 100,
      focalLengthMm: 'Native',
      cameraModel: 'Local Project',
      lensModel: ext,
    },
    faceOverlay: dto.faceOverlay,
    developRecipe: hydrateDevelopRecipe(dto.developRecipe),
  };
}

function mapSessionDto(
  dto: ShootSessionDto,
  runtimeStatus: RuntimeInfo | null,
): StudioSessionSnapshot {
  const elapsedSec = Math.max(0.01, dto.scanElapsedMs / 1000);
  const fps = Math.round(dto.totalPhotos / elapsedSec);

  const clusters: SubjectCluster[] = dto.clusters.map((c, idx) => {
    const isSpecialBucket = c.id.startsWith('_');
    const backendCode =
      c.subjectCode && c.subjectCode.trim() !== '#000' && c.subjectCode.trim() !== '000'
        ? c.subjectCode.trim()
        : undefined;
    const subjectCode = isSpecialBucket
      ? undefined
      : (backendCode ?? `#${String(idx + 1).padStart(3, '0')}`);
    const slateCode = c.hasSlate ? (subjectCode ?? `#${String(idx + 1).padStart(3, '0')}`) : 'NO SLATE';
    const clusterAmbiguous = dto.ambiguousReview.filter((a) => a.candidateClusterId === c.id).length;
    return {
      anchor: {
        id: c.id,
        slateCode,
        subjectCode,
        subjectName: c.displayName,
        ocrExtractedText: c.hasSlate ? c.displayName : 'Face-Only Cluster',
        ocrConfidencePct: c.hasSlate ? 99.0 : 100,
        anchorPhotoId: c.representativePhotoId,
        thumbnailUrl: toWebViewAssetUrl(c.representativePreviewPath),
        capturedAtLabel: `${c.photos.length} photos`,
        sessionTimeRange: dto.projectName,
        cameraLabel: 'Local Project',
        lensLabel: dto.isClustered
          ? `YuNet + ${runtimeStatus?.recognitionModelLabel ?? 'ArcFace'}`
          : 'Unclustered',
        wbCalibrated: false,
        wbCalibrationLabel: c.hasSlate ? 'Slate Ready' : 'Auto / Manual WB',
        reportedTotalExposures: c.photos.length,
        ambiguousCount: clusterAmbiguous,
      },
      photos: c.photos.map(mapPhotoDto),
    };
  });

  const defaultDirId = dto.directories[0]?.id ?? 'dir-root';
  const ambiguousQueue: PendingAmbiguousEntry[] = dto.ambiguousReview.map((a) => ({
    candidateClusterId: a.candidateClusterId,
    reason: 'Borderline SFace Similarity',
    photo: {
      id: a.photoId,
      fileName: a.fileName,
      directoryId: defaultDirId,
      imageUrl: toWebViewAssetUrl(a.previewPath),
      isSlateAnchor: false,
      clusterId: a.candidateClusterId,
      sfaceCosineSimilarity: a.cosineSimilarity,
      sharpnessPct: 90,
      eyeAfLocked: false,
      heroSelect: false,
      exif: {
        shutterSpeed: 'RAW',
        aperture: 'Original',
        iso: 100,
        focalLengthMm: 'Native',
        cameraModel: 'Local Project',
        lensModel: 'Review',
      },
      faceOverlay: a.faceOverlay,
      developRecipe: createDefaultDevelopRecipe(),
    },
  }));

  const slatesReadCount = clusters.filter((c) => c.anchor.slateCode !== 'NO SLATE').length;
  const clusteredFacesCount = clusters.reduce((sum, c) => sum + c.photos.length, 0);

  return {
    isClustered: dto.isClustered,
    folderPath: dto.folderPath,
    telemetry: {
      sessionName: dto.projectName,
      projectName: dto.projectName,
      totalPhotos: dto.totalPhotos,
      indexedPhotos: dto.totalPhotos,
      indexElapsedSeconds: Number(elapsedSec.toFixed(2)),
      indexErrorCount: dto.scanErrorCount,
      slatesReadCount,
      clusteredFacesCount,
      needsReviewCount: dto.ambiguousReview.length,
      unassignedFacesCount: dto.noFacePhotoCount,
      focusDropsCount: 0,
      ocrMatchedCount: clusteredFacesCount,
      throughputFps: fps,
      modelSummary: `YuNet 2026 + ${runtimeStatus?.recognitionModelLabel ?? 'ArcFace R50 512-D'}`,
      runtimeBackendLabel: runtimeStatus
        ? `${runtimeStatus.backendLabel} • ${runtimeStatus.supportedRawCount} RAWs`
        : 'Rust + ONNX Runtime',
      vramUsedMb: 0,
      vramTotalMb: 0,
      engineLoadPct: 0,
    },
    directories: dto.directories.map((d) => ({
      id: d.id,
      name: d.name,
      rawCount: d.photoCount,
    })),
    virtualFolders: (dto.virtualFolders ?? []).map(
      (f): VirtualFolder => ({
        id: f.id,
        name: f.name,
        parentId: f.parentId,
      }),
    ),
    clusters,
    ambiguousQueue,
    outputRule: DEFAULT_OUTPUT_NAMING_RULE,
    history: {
      canUndo: dto.canUndo,
      canRedo: dto.canRedo,
      undoLabel: dto.undoLabel,
      redoLabel: dto.redoLabel,
    },
  };
}

/**
 * Tauri v2 Desktop Gateway adapter implementing the inward StudioGateway port.
 * Loads folders fast for Edit-First workflow (`load_shoot_folder`), then clusters on demand
 * via `run_face_clustering` without losing any saved `.kroma.json` edits.
 */
export class TauriStudioGateway implements StudioGateway {
  private static readonly MAX_DEVELOP_BUFFER_LRU = 12;
  private hasLoadedLocalShoot = false;
  private cachedRuntimeStatus: RuntimeInfo | null = null;
  private readonly developBufferLru = new Map<string, DevelopWorkingBuffer | null>();
  private readonly inFlightDevelopBuffers = new Map<string, Promise<DevelopWorkingBuffer | null>>();

  private clearDevelopBufferCache(): void {
    this.developBufferLru.clear();
    this.inFlightDevelopBuffers.clear();
  }

  isDesktopRuntime(): boolean {
    return true;
  }

  subscribeScanProgress(listener: (progress: ScanProgressUpdate) => void): () => void {
    let disposed = false;
    let unlistenFn: (() => void) | null = null;
    void listen<ScanProgressUpdate>('kroma://scan-progress', (event) => {
      if (!disposed) {
        listener(event.payload);
      }
    }).then((unlisten) => {
      if (disposed) {
        unlisten();
      } else {
        unlistenFn = unlisten;
      }
    });
    return () => {
      disposed = true;
      unlistenFn?.();
    };
  }

  subscribePreviewsReady(listener: (batch: PreviewsReadyBatch) => void): () => void {
    interface WirePreviewPatch {
      readonly photoId: string;
      readonly previewPath?: string | null;
      readonly renderedPreviewPath?: string | null;
      readonly clearRenderedPreview?: boolean;
      readonly capturedAtEpochSec: number | null;
      readonly capturedAtLabel: string | null;
    }
    interface WireBatch {
      readonly updates: readonly WirePreviewPatch[];
      readonly completed: number;
      readonly total: number;
    }
    let disposed = false;
    let unlistenFn: (() => void) | null = null;
    void listen<WireBatch>('kroma://previews-ready', (event) => {
      if (!disposed) {
        listener({
          completed: event.payload.completed,
          total: event.payload.total,
          updates: event.payload.updates.map((u) => ({
            photoId: u.photoId,
            imageUrl: u.previewPath ? toWebViewAssetUrl(u.previewPath) : null,
            renderedThumbnailUrl: u.renderedPreviewPath
              ? toWebViewAssetUrl(u.renderedPreviewPath)
              : null,
            clearRenderedThumbnail: Boolean(u.clearRenderedPreview),
            capturedAtEpochSec: u.capturedAtEpochSec,
            capturedAtLabel: u.capturedAtLabel,
          })),
        });
      }
    }).then((unlisten) => {
      if (disposed) {
        unlisten();
      } else {
        unlistenFn = unlisten;
      }
    });
    return () => {
      disposed = true;
      unlistenFn?.();
    };
  }

  subscribeThumbnailRenderProgress(
    listener: (progress: ThumbnailRenderProgress) => void,
  ): () => void {
    let disposed = false;
    let unlistenFn: (() => void) | null = null;
    void listen<ThumbnailRenderProgress>('kroma://thumbnail-render-progress', (event) => {
      if (!disposed) {
        listener(event.payload);
      }
    }).then((unlisten) => {
      if (disposed) {
        unlisten();
      } else {
        unlistenFn = unlisten;
      }
    });
    return () => {
      disposed = true;
      unlistenFn?.();
    };
  }

  subscribeModelDownloadProgress(listener: (progress: ModelDownloadProgress) => void): () => void {
    let disposed = false;
    let unlistenFn: (() => void) | null = null;
    void listen<ModelDownloadProgress>('kroma://model-download-progress', (event) => {
      if (!disposed) {
        listener(event.payload);
      }
    }).then((unlisten) => {
      if (disposed) {
        unlisten();
      } else {
        unlistenFn = unlisten;
      }
    });
    return () => {
      disposed = true;
      unlistenFn?.();
    };
  }

  async checkRuntimeInfo(): Promise<RuntimeInfo | null> {
    try {
      this.cachedRuntimeStatus = await invoke<RuntimeInfo>('check_runtime_status');
      return this.cachedRuntimeStatus;
    } catch {
      return null;
    }
  }

  async checkModelsManifest(): Promise<ModelManifestStatus | null> {
    return invoke<ModelManifestStatus>('check_models_manifest');
  }

  async downloadMissingModels(): Promise<ModelDownloadOutcome | null> {
    const outcome = await invoke<ModelDownloadOutcome>('download_missing_models');
    this.cachedRuntimeStatus = outcome.runtime;
    return outcome;
  }

  async cancelModelDownloads(): Promise<void> {
    await invoke<void>('cancel_model_downloads');
  }

  async setGpuAcceleration(enabled: boolean): Promise<RuntimeInfo | null> {
    const updated = await invoke<RuntimeInfo>('set_gpu_acceleration', { enabled });
    this.cachedRuntimeStatus = updated;
    return updated;
  }

  async getStorageCacheSummary(): Promise<StorageCacheSummary | null> {
    try {
      return await invoke<StorageCacheSummary>('get_storage_cache_summary');
    } catch {
      return null;
    }
  }

  async pickCacheDirectory(): Promise<string | null> {
    const selected = await openNativeDialog({
      directory: true,
      multiple: false,
      title: 'Choose Kroma Cache Folder',
    });
    if (!selected || Array.isArray(selected)) {
      return null;
    }
    return selected;
  }

  async setStorageCacheDir(cacheDir: string | null): Promise<StorageCacheSummary | null> {
    this.clearDevelopBufferCache();
    return invoke<StorageCacheSummary>('set_storage_cache_dir', { cacheDir });
  }

  async clearStorageCache(target: ClearCacheTarget): Promise<ClearCacheResult | null> {
    if (target === 'previews' || target === 'all') {
      this.clearDevelopBufferCache();
    }
    return invoke<ClearCacheResult>('clear_storage_cache', { target });
  }

  async pickOnnxModelFile(): Promise<string | null> {
    const selected = await openNativeDialog({
      directory: false,
      multiple: false,
      title: 'Select ONNX AI Model (.onnx)',
      filters: [{ name: 'ONNX Neural Network', extensions: ['onnx'] }],
    });
    if (!selected || Array.isArray(selected)) {
      return null;
    }
    return selected;
  }

  async inspectOnnxModel(
    modelPath: string,
    role: AiModelRole,
  ): Promise<ModelInspectionResult | null> {
    return invoke<ModelInspectionResult>('inspect_onnx_model', {
      modelPath,
      role,
    });
  }

  async setCustomAiModels(
    yunetPath: string | null,
    recognizerPath: string | null,
    segmenterPath: string | null = null,
  ): Promise<RuntimeInfo | null> {
    const updated = await invoke<RuntimeInfo>('set_custom_ai_models', {
      yunetPath,
      recognizerPath,
      segmenterPath,
    });
    this.cachedRuntimeStatus = updated;
    return updated;
  }

  async generateAiMask(
    photoId: string,
    kind: AiMaskKind,
  ): Promise<GeneratedAiMaskResult | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    return invoke<GeneratedAiMaskResult>('generate_ai_mask', {
      photoId,
      kind,
    });
  }

  async loadSession(): Promise<StudioSessionSnapshot | null> {
    await this.checkRuntimeInfo();
    return null;
  }

  async openShootFolder(): Promise<StudioSessionSnapshot | null> {
    const selected = await openNativeDialog({
      directory: true,
      multiple: false,
      title: 'Open Project Folder',
    });
    if (!selected || Array.isArray(selected)) {
      return null;
    }
    return this.openShootFolderAtPath(selected);
  }

  async openShootFolderAtPath(folderPath: string): Promise<StudioSessionSnapshot | null> {
    const trimmed = folderPath.trim();
    if (!trimmed) return null;
    this.clearDevelopBufferCache();
    const dto = await invoke<ShootSessionDto>('load_shoot_folder', {
      folderPath: trimmed,
    });
    this.hasLoadedLocalShoot = true;
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async runFaceClustering(
    sfaceThreshold: number,
    useTemporalAssist: boolean = true,
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('run_face_clustering', {
      sfaceThreshold,
      useTemporalAssist,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async scanProjectSlatesOcr(
    options?: SlateOcrScanOptions,
  ): Promise<readonly ClusterSlateOcrResult[]> {
    if (!this.hasLoadedLocalShoot) {
      return [];
    }
    const rows = await invoke<ClusterSlateOcrResultDto[]>('scan_project_slates_ocr', {
      scanAllFrames: options?.scanAllFrames ?? false,
      clusterIds:
        options?.clusterIds && options.clusterIds.length > 0 ? [...options.clusterIds] : null,
    });
    return rows.map((row) => ({
      clusterId: row.clusterId,
      currentName: row.currentName,
      slatePhotoId: row.slatePhotoId,
      slateFileName: row.slateFileName,
      slatePreviewUrl: row.slatePreviewPath ? toWebViewAssetUrl(row.slatePreviewPath) : null,
      proposedName: row.proposedName,
      proposedCode: row.proposedCode,
      confidence: row.confidence,
      scannedFrameCount: row.scannedFrameCount,
      lines: row.lines,
    }));
  }

  async cancelSlateOcrScan(): Promise<void> {
    if (!this.hasLoadedLocalShoot) return;
    await invoke('cancel_slate_ocr_scan');
  }

  async scanSinglePhotoOcr(photoId: string): Promise<PhotoOcrResult | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<PhotoOcrResultDto>('scan_single_photo_ocr', { photoId });
    return {
      photoId: dto.photoId,
      fileName: dto.fileName,
      previewUrl: toWebViewAssetUrl(dto.previewPath),
      proposedName: dto.proposedName,
      proposedCode: dto.proposedCode,
      confidence: dto.confidence,
      lines: dto.lines,
    };
  }

  async applySlateOcrRenames(
    items: readonly SlateRenameApplyItem[],
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('apply_slate_ocr_renames', {
      items: items.map((item) => ({
        clusterId: item.clusterId,
        nextName: item.nextName,
        subjectCode: item.subjectCode ?? null,
        slatePhotoId: item.slatePhotoId ?? null,
        ocrRawText: item.ocrRawText ?? null,
        ocrConfidence: item.ocrConfidence ?? null,
        markAsSlatePhoto: item.markAsSlatePhoto,
      })),
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async renameSlateSubject(
    clusterId: string,
    nextSubjectName: string,
  ): Promise<StudioSessionSnapshot | void> {
    if (!this.hasLoadedLocalShoot) {
      return undefined;
    }
    const dto = await invoke<ShootSessionDto>('rename_cluster', {
      clusterId,
      nextName: nextSubjectName,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async mergeClusters(
    sourceClusterId: string,
    targetClusterId: string,
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('merge_clusters', {
      sourceClusterId,
      targetClusterId,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async splitPhotosToNewCluster(
    photoIds: readonly string[],
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('split_photos_to_new_cluster', {
      photoIds: [...photoIds],
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async movePhotosToCluster(
    photoIds: readonly string[],
    targetClusterId: string,
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('move_photos_to_cluster', {
      photoIds: [...photoIds],
      targetClusterId,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async createVirtualFolder(
    name: string,
    parentId?: string | null,
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('create_virtual_folder', {
      name,
      parentId: parentId ?? null,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async renameVirtualFolder(
    folderId: string,
    nextName: string,
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('rename_virtual_folder', {
      folderId,
      nextName,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async moveVirtualFolder(
    folderId: string,
    nextParentId: string | null,
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('move_virtual_folder', {
      folderId,
      nextParentId,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async deleteVirtualFolder(folderId: string): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('delete_virtual_folder', { folderId });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async assignPhotosToVirtualFolder(
    photoIds: readonly string[],
    folderId: string | null,
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('assign_photos_to_virtual_folder', {
      photoIds: [...photoIds],
      folderId,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async autoLevelClusterEyes(clusterId: string): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('auto_level_cluster_eyes', { clusterId });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  peekDevelopWorkingBuffer(
    photoId: string,
    maxDim = 1600,
    precision: WorkingBufferPrecision = 'f16',
  ): DevelopWorkingBuffer | null {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const cacheKey = `${photoId}@${maxDim}@${precision}`;
    if (!this.developBufferLru.has(cacheKey)) {
      return null;
    }
    const hit = this.developBufferLru.get(cacheKey) ?? null;
    this.developBufferLru.delete(cacheKey);
    this.developBufferLru.set(cacheKey, hit);
    return hit;
  }

  async loadDevelopWorkingBuffer(
    photoId: string,
    maxDim = 1600,
    precision: WorkingBufferPrecision = 'f16',
  ): Promise<DevelopWorkingBuffer | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const cacheKey = `${photoId}@${maxDim}@${precision}`;
    if (this.developBufferLru.has(cacheKey)) {
      const hit = this.developBufferLru.get(cacheKey) ?? null;
      this.developBufferLru.delete(cacheKey);
      this.developBufferLru.set(cacheKey, hit);
      return hit;
    }
    const existing = this.inFlightDevelopBuffers.get(cacheKey);
    if (existing) {
      return existing;
    }

    const request = invoke<ArrayBuffer>('load_develop_working_buffer', {
      photoId,
      maxDim,
      precision,
    })
      .then((raw) => {
        const parsed = raw instanceof ArrayBuffer ? parseDevelopWorkingBuffer(raw) : null;
        if (this.developBufferLru.size >= TauriStudioGateway.MAX_DEVELOP_BUFFER_LRU) {
          const oldestKey = this.developBufferLru.keys().next().value;
          if (oldestKey !== undefined) {
            this.developBufferLru.delete(oldestKey);
          }
        }
        this.developBufferLru.set(cacheKey, parsed);
        return parsed;
      })
      .catch(() => null)
      .finally(() => {
        this.inFlightDevelopBuffers.delete(cacheKey);
      });

    this.inFlightDevelopBuffers.set(cacheKey, request);
    return request;
  }

  async render32BitThumbnails(
    mode: ThumbnailPreviewMode,
    photoIds?: readonly string[],
  ): Promise<void> {
    if (!this.hasLoadedLocalShoot) {
      return;
    }
    await invoke<void>('render_32bit_thumbnails', {
      mode,
      photoIds: photoIds && photoIds.length > 0 ? [...photoIds] : null,
    });
  }

  async savePhotoDevelopRecipe(
    photoId: string,
    recipe: DevelopRecipe,
  ): Promise<LiveRgbHistogram | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const histDto = await invoke<HistogramDto>('save_photo_develop_recipe', { photoId, recipe });
    return mapHistogramDto(histDto);
  }

  async savePhotoDevelopRecipesBatch(
    items: readonly { readonly photoId: string; readonly recipe: DevelopRecipe }[],
    activePhotoId?: string | null,
  ): Promise<LiveRgbHistogram | null> {
    if (items.length === 0 || !this.hasLoadedLocalShoot) return null;
    const histDto = await invoke<HistogramDto | null>('save_photo_develop_recipes_batch', {
      items: items.map((item) => ({ photoId: item.photoId, recipe: item.recipe })),
      activePhotoId: activePhotoId ?? null,
    });
    return histDto ? mapHistogramDto(histDto) : null;
  }

  async syncDevelopRecipeToPhotos(
    sourcePhotoId: string,
    targetPhotoIds: readonly string[],
    sourceRecipe: DevelopRecipe,
    mask: DevelopSyncMask,
  ): Promise<void> {
    if (!this.hasLoadedLocalShoot) {
      return;
    }
    await invoke('save_photo_develop_recipe', { photoId: sourcePhotoId, recipe: sourceRecipe });
    await invoke('sync_cluster_develop_recipe', {
      sourcePhotoId,
      targetPhotoIds: [...targetPhotoIds],
      mask,
    });
  }

  async applyRelativeDevelopDeltaToPhotos(
    photoIds: readonly string[],
    delta: RelativeDevelopDelta,
  ): Promise<void> {
    if (!this.hasLoadedLocalShoot) {
      return;
    }
    await invoke('apply_relative_develop_nudge', {
      photoIds: [...photoIds],
      delta,
    });
  }

  async calibrateWhiteBalanceFromSlate(
    slatePhotoId: string,
    targetPhotoIds: readonly string[],
  ): Promise<WhiteBalanceSettings | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    return invoke<WhiteBalanceSettings>('calibrate_wb_from_slate', {
      slatePhotoId,
      targetPhotoIds: [...targetPhotoIds],
    });
  }

  async resolveAmbiguousMatch(
    photoId: string,
    decision: 'approve' | 'split',
  ): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('resolve_ambiguous_match', {
      photoId,
      decision,
    });
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async undoOrganizationAction(): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('undo_organization_action');
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async redoOrganizationAction(): Promise<StudioSessionSnapshot | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const dto = await invoke<ShootSessionDto>('redo_organization_action');
    return mapSessionDto(dto, this.cachedRuntimeStatus);
  }

  async pickExportDirectory(): Promise<string | null> {
    const selected = await openNativeDialog({
      directory: true,
      multiple: false,
      title: 'Choose Export Destination Folder',
    });
    if (!selected || Array.isArray(selected)) {
      return null;
    }
    return selected;
  }

  async estimateExportSize(request: ExportSizeEstimateRequest): Promise<MeasuredExportSize | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    return invoke<MeasuredExportSize>('estimate_export_size', {
      request: {
        excludeSlatePhoto: request.excludeSlatePhoto,
        archiveExcludedSlates: request.archiveExcludedSlates,
        mode: request.mode,
        format: request.format,
        quality: request.quality,
        photoIds: request.photoIds && request.photoIds.length > 0 ? request.photoIds : undefined,
      },
    });
  }

  async exportShoot(request: BatchExportRequest): Promise<BatchExportResult | null> {
    if (!this.hasLoadedLocalShoot) {
      return null;
    }
    const providedRoot = request.outputRoot?.trim();
    const outputRoot = providedRoot && providedRoot.length > 0
      ? providedRoot
      : await this.pickExportDirectory();
    if (!outputRoot) {
      return null;
    }

    const result = await invoke<ExportShootResultDto>('organize_and_export_shoot', {
      request: {
        outputRoot,
        namingPattern: request.namingPattern,
        excludeSlatePhoto: request.excludeSlatePhoto,
        archiveExcludedSlates: request.archiveExcludedSlates,
        mode: request.mode,
        format: request.format,
        quality: request.quality,
        photoIds: request.photoIds && request.photoIds.length > 0 ? request.photoIds : undefined,
      },
    });

    return {
      organizedCount: result.organizedCount,
      failedCount: result.failedCount,
      outputRoot,
      manifestPath: result.manifestPath,
    };
  }

  async minimizeWindow(): Promise<void> {
    await getCurrentWindow().minimize();
  }

  async toggleMaximizeWindow(): Promise<boolean> {
    const win = getCurrentWindow();
    await win.toggleMaximize();
    return win.isMaximized();
  }

  async isWindowMaximized(): Promise<boolean> {
    return getCurrentWindow().isMaximized();
  }

  subscribeWindowMaximized(listener: (maximized: boolean) => void): () => void {
    let disposed = false;
    let unlisten: (() => void) | null = null;
    const win = getCurrentWindow();

    void win
      .isMaximized()
      .then((max) => {
        if (!disposed) listener(max);
      })
      .catch(() => {});

    void win
      .onResized(() => {
        void win
          .isMaximized()
          .then((max) => {
            if (!disposed) listener(max);
          })
          .catch(() => {});
      })
      .then((fn) => {
        if (disposed) {
          fn();
        } else {
          unlisten = fn;
        }
      })
      .catch(() => {});

    return () => {
      disposed = true;
      unlisten?.();
    };
  }

  async exitApp(): Promise<void> {
    await invoke('exit_app');
  }
}
