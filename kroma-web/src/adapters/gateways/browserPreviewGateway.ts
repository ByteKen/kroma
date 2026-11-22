import type {
  BatchExportRequest,
  BatchExportResult,
  LiveRgbHistogram,
  RuntimeInfo,
  ScanProgressUpdate,
  StudioGateway,
  StudioSessionSnapshot,
} from '../../application/ports/studioGateway';
import { KROMA_FALLBACK_VERSION } from '../../domain/appUpdates';
import type {
  DevelopRecipe,
  DevelopSyncMask,
  RelativeDevelopDelta,
  WhiteBalanceSettings,
} from '../../domain/develop';

/**
 * Minimal non-desktop fallback adapter used only when `kroma-web` is opened
 * directly in a web browser outside the Tauri desktop runtime.
 * Contains no mock photos or sample sessions; local folder workflows require Kroma Desktop.
 */
export class BrowserPreviewGateway implements StudioGateway {
  isDesktopRuntime(): boolean {
    return false;
  }

  subscribeScanProgress(_listener: (progress: ScanProgressUpdate) => void): () => void {
    return () => {};
  }

  async checkRuntimeInfo(): Promise<RuntimeInfo | null> {
    return {
      yunetModelPath: null,
      sfaceModelPath: null,
      modelsReady: false,
      ocrModelsReady: false,
      segmenterModelsReady: false,
      supportedRawCount: 29,
      backendLabel: 'Browser Preview',
      appVersion: KROMA_FALLBACK_VERSION,
      gpuAvailable: false,
      gpuAccelerationEnabled: false,
    };
  }

  async loadSession(): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async openShootFolder(): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async openShootFolderAtPath(_folderPath: string): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async runFaceClustering(
    _sfaceThreshold: number,
    _useTemporalAssist?: boolean,
  ): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async renameSlateSubject(
    _clusterId: string,
    _nextSubjectName: string,
  ): Promise<StudioSessionSnapshot | void> {
    return undefined;
  }

  async mergeClusters(
    _sourceClusterId: string,
    _targetClusterId: string,
  ): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async splitPhotosToNewCluster(
    _photoIds: readonly string[],
  ): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async movePhotosToCluster(
    _photoIds: readonly string[],
    _targetClusterId: string,
  ): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async autoLevelClusterEyes(_clusterId: string): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async savePhotoDevelopRecipe(
    _photoId: string,
    _recipe: DevelopRecipe,
  ): Promise<LiveRgbHistogram | null> {
    return null;
  }

  async syncDevelopRecipeToPhotos(
    _sourcePhotoId: string,
    _targetPhotoIds: readonly string[],
    _sourceRecipe: DevelopRecipe,
    _mask: DevelopSyncMask,
  ): Promise<void> {
    return undefined;
  }

  async applyRelativeDevelopDeltaToPhotos(
    _photoIds: readonly string[],
    _delta: RelativeDevelopDelta,
  ): Promise<void> {
    return undefined;
  }

  async calibrateWhiteBalanceFromSlate(
    _slatePhotoId: string,
    _targetPhotoIds: readonly string[],
  ): Promise<WhiteBalanceSettings | null> {
    return null;
  }

  async resolveAmbiguousMatch(
    _photoId: string,
    _decision: 'approve' | 'split',
  ): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async undoOrganizationAction(): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async redoOrganizationAction(): Promise<StudioSessionSnapshot | null> {
    return null;
  }

  async exportShoot(_request: BatchExportRequest): Promise<BatchExportResult | null> {
    return null;
  }
}
