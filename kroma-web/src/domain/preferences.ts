import { type ExportDeliveryMode, type ExportRenderFormat } from './exportEstimate';
import { DEFAULT_REVIEW_WARNING_THRESHOLD, DEFAULT_SFACE_THRESHOLD } from './identity';
import { DEFAULT_OUTPUT_NAMING_RULE } from './organization';

export type DefaultOpenWorkspace = 'develop' | 'culling';
export type StudioPreviewMaxDim = 1600 | 2048 | 2560;
export type ThumbnailPreviewMode = 'fast' | 'edited_32bit' | 'all_32bit';
export type DefaultSubjectSortMode = 'capture_time' | 'confidence' | 'exposures';
export type CanvasBackdropTone = 'obsidian' | 'charcoal' | 'neutral_gray';
export type InterfaceScalePct = 90 | 100 | 110;

export function resolveCanvasBackdropHex(tone: CanvasBackdropTone): string {
  switch (tone) {
    case 'obsidian':
      return '#0b0c0f';
    case 'neutral_gray':
      return '#2b2d31';
    case 'charcoal':
    default:
      return '#111214';
  }
}

export interface StudioPreferences {
  readonly sfaceThreshold: number;
  readonly reviewWarningThreshold: number;
  readonly useTemporalAssist: boolean;
  readonly enableOcrSlateRenaming: boolean;
  readonly showLandmarksOverlay: boolean;
  readonly openLastProjectOnStartup: boolean;
  readonly defaultOpenWorkspace: DefaultOpenWorkspace;
  readonly defaultSubjectSortMode: DefaultSubjectSortMode;
  readonly canvasBackdropTone: CanvasBackdropTone;
  readonly interfaceScale: InterfaceScalePct;
  readonly showThumbnailMetadata: boolean;
  readonly reduceMotion: boolean;
  readonly enableGpuAcceleration: boolean;
  readonly studioPreviewMaxDim: StudioPreviewMaxDim;
  readonly thumbnailPreviewMode: ThumbnailPreviewMode;
  readonly customCacheDir: string | null;
  readonly customDetectorModelPath: string | null;
  readonly customRecognizerModelPath: string | null;
  readonly customSegmenterModelPath: string | null;
  readonly defaultDeliveryMode: ExportDeliveryMode;
  readonly defaultRenderFormat: ExportRenderFormat;
  readonly defaultJpegQuality: number;
  readonly defaultExportBaseDir: string | null;
  readonly createExportProjectSubfolder: boolean;
  readonly defaultNamingPattern: string;
  readonly excludeSlateFromClientPack: boolean;
  readonly archiveSlatesSeparately: boolean;
}

export const DEFAULT_STUDIO_PREFERENCES: StudioPreferences = {
  sfaceThreshold: DEFAULT_SFACE_THRESHOLD,
  reviewWarningThreshold: DEFAULT_REVIEW_WARNING_THRESHOLD,
  useTemporalAssist: true,
  enableOcrSlateRenaming: true,
  showLandmarksOverlay: true,
  openLastProjectOnStartup: false,
  defaultOpenWorkspace: 'develop',
  defaultSubjectSortMode: 'capture_time',
  canvasBackdropTone: 'charcoal',
  interfaceScale: 100,
  showThumbnailMetadata: true,
  reduceMotion: false,
  enableGpuAcceleration: true,
  studioPreviewMaxDim: 2048,
  thumbnailPreviewMode: 'edited_32bit',
  customCacheDir: null,
  customDetectorModelPath: null,
  customRecognizerModelPath: null,
  customSegmenterModelPath: null,
  defaultDeliveryMode: 'develop',
  defaultRenderFormat: 'jpeg',
  defaultJpegQuality: 94,
  defaultExportBaseDir: null,
  createExportProjectSubfolder: true,
  defaultNamingPattern: DEFAULT_OUTPUT_NAMING_RULE.pattern,
  excludeSlateFromClientPack: DEFAULT_OUTPUT_NAMING_RULE.excludeSlateFromClientPack,
  archiveSlatesSeparately: true,
};

const VALID_DELIVERY_MODES = new Set<ExportDeliveryMode>([
  'develop',
  'develop_and_raw',
  'copy',
  'move',
]);
const VALID_RENDER_FORMATS = new Set<ExportRenderFormat>(['jpeg', 'png', 'tiff16']);

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

const LEGACY_DEFAULT_NAMING_PATTERNS = new Set([
  '/[Slate_Name]_[ID]/IMG_001.jpg',
  '/[Slate_Name]_[ID]/[Original_Name].jpg',
]);

export function normalizeStudioPreferences(raw: unknown): StudioPreferences {
  if (!raw || typeof raw !== 'object') {
    return { ...DEFAULT_STUDIO_PREFERENCES };
  }
  const obj = raw as Record<string, unknown>;

  const deliveryMode =
    typeof obj.defaultDeliveryMode === 'string' &&
      VALID_DELIVERY_MODES.has(obj.defaultDeliveryMode as ExportDeliveryMode)
      ? (obj.defaultDeliveryMode as ExportDeliveryMode)
      : DEFAULT_STUDIO_PREFERENCES.defaultDeliveryMode;

  const renderFormat =
    typeof obj.defaultRenderFormat === 'string' &&
      VALID_RENDER_FORMATS.has(obj.defaultRenderFormat as ExportRenderFormat)
      ? (obj.defaultRenderFormat as ExportRenderFormat)
      : DEFAULT_STUDIO_PREFERENCES.defaultRenderFormat;

  const defaultExportBaseDir =
    typeof obj.defaultExportBaseDir === 'string' && obj.defaultExportBaseDir.trim().length > 0
      ? obj.defaultExportBaseDir.trim()
      : null;

  const customCacheDir =
    typeof obj.customCacheDir === 'string' && obj.customCacheDir.trim().length > 0
      ? obj.customCacheDir.trim()
      : null;

  const customDetectorModelPath =
    typeof obj.customDetectorModelPath === 'string' && obj.customDetectorModelPath.trim().length > 0
      ? obj.customDetectorModelPath.trim()
      : null;

  const customRecognizerModelPath =
    typeof obj.customRecognizerModelPath === 'string' &&
      obj.customRecognizerModelPath.trim().length > 0
      ? obj.customRecognizerModelPath.trim()
      : null;

  const customSegmenterModelPath =
    typeof obj.customSegmenterModelPath === 'string' &&
      obj.customSegmenterModelPath.trim().length > 0
      ? obj.customSegmenterModelPath.trim()
      : null;

  const rawNamingPattern =
    typeof obj.defaultNamingPattern === 'string' ? obj.defaultNamingPattern.trim() : '';
  const defaultNamingPattern =
    rawNamingPattern.length > 0 && !LEGACY_DEFAULT_NAMING_PATTERNS.has(rawNamingPattern)
      ? rawNamingPattern
      : DEFAULT_STUDIO_PREFERENCES.defaultNamingPattern;

  return {
    sfaceThreshold: Number(
      clampNumber(obj.sfaceThreshold, 0.25, 0.9, DEFAULT_STUDIO_PREFERENCES.sfaceThreshold).toFixed(
        2,
      ),
    ),
    reviewWarningThreshold: Number(
      clampNumber(
        obj.reviewWarningThreshold,
        0.5,
        0.95,
        DEFAULT_STUDIO_PREFERENCES.reviewWarningThreshold,
      ).toFixed(2),
    ),
    useTemporalAssist: asBoolean(
      obj.useTemporalAssist,
      DEFAULT_STUDIO_PREFERENCES.useTemporalAssist,
    ),
    enableOcrSlateRenaming: asBoolean(
      obj.enableOcrSlateRenaming,
      DEFAULT_STUDIO_PREFERENCES.enableOcrSlateRenaming,
    ),
    showLandmarksOverlay: asBoolean(
      obj.showLandmarksOverlay,
      DEFAULT_STUDIO_PREFERENCES.showLandmarksOverlay,
    ),
    openLastProjectOnStartup: asBoolean(
      obj.openLastProjectOnStartup,
      DEFAULT_STUDIO_PREFERENCES.openLastProjectOnStartup,
    ),
    defaultOpenWorkspace:
      obj.defaultOpenWorkspace === 'culling' || obj.defaultOpenWorkspace === 'develop'
        ? obj.defaultOpenWorkspace
        : DEFAULT_STUDIO_PREFERENCES.defaultOpenWorkspace,
    defaultSubjectSortMode:
      obj.defaultSubjectSortMode === 'capture_time' ||
      obj.defaultSubjectSortMode === 'confidence' ||
      obj.defaultSubjectSortMode === 'exposures'
        ? obj.defaultSubjectSortMode
        : DEFAULT_STUDIO_PREFERENCES.defaultSubjectSortMode,
    canvasBackdropTone:
      obj.canvasBackdropTone === 'obsidian' ||
      obj.canvasBackdropTone === 'charcoal' ||
      obj.canvasBackdropTone === 'neutral_gray'
        ? obj.canvasBackdropTone
        : DEFAULT_STUDIO_PREFERENCES.canvasBackdropTone,
    interfaceScale:
      obj.interfaceScale === 90 || obj.interfaceScale === 100 || obj.interfaceScale === 110
        ? obj.interfaceScale
        : DEFAULT_STUDIO_PREFERENCES.interfaceScale,
    showThumbnailMetadata: asBoolean(
      obj.showThumbnailMetadata,
      DEFAULT_STUDIO_PREFERENCES.showThumbnailMetadata,
    ),
    reduceMotion: asBoolean(obj.reduceMotion, DEFAULT_STUDIO_PREFERENCES.reduceMotion),
    enableGpuAcceleration: asBoolean(
      obj.enableGpuAcceleration,
      DEFAULT_STUDIO_PREFERENCES.enableGpuAcceleration,
    ),
    studioPreviewMaxDim:
      obj.studioPreviewMaxDim === 1600 ||
      obj.studioPreviewMaxDim === 2048 ||
      obj.studioPreviewMaxDim === 2560
        ? obj.studioPreviewMaxDim
        : DEFAULT_STUDIO_PREFERENCES.studioPreviewMaxDim,
    thumbnailPreviewMode:
      obj.thumbnailPreviewMode === 'fast' ||
      obj.thumbnailPreviewMode === 'edited_32bit' ||
      obj.thumbnailPreviewMode === 'all_32bit'
        ? obj.thumbnailPreviewMode
        : DEFAULT_STUDIO_PREFERENCES.thumbnailPreviewMode,
    customCacheDir,
    customDetectorModelPath,
    customRecognizerModelPath,
    customSegmenterModelPath,
    defaultDeliveryMode: deliveryMode,
    defaultRenderFormat: renderFormat,
    defaultJpegQuality: Math.round(
      clampNumber(
        obj.defaultJpegQuality,
        70,
        100,
        DEFAULT_STUDIO_PREFERENCES.defaultJpegQuality,
      ),
    ),
    defaultExportBaseDir,
    createExportProjectSubfolder: asBoolean(
      obj.createExportProjectSubfolder,
      DEFAULT_STUDIO_PREFERENCES.createExportProjectSubfolder,
    ),
    defaultNamingPattern,
    excludeSlateFromClientPack: asBoolean(
      obj.excludeSlateFromClientPack,
      DEFAULT_STUDIO_PREFERENCES.excludeSlateFromClientPack,
    ),
    archiveSlatesSeparately: asBoolean(
      obj.archiveSlatesSeparately,
      DEFAULT_STUDIO_PREFERENCES.archiveSlatesSeparately,
    ),
  };
}

const PREFERENCE_KEYS: readonly (keyof StudioPreferences)[] = [
  'sfaceThreshold',
  'reviewWarningThreshold',
  'useTemporalAssist',
  'enableOcrSlateRenaming',
  'showLandmarksOverlay',
  'openLastProjectOnStartup',
  'defaultOpenWorkspace',
  'defaultSubjectSortMode',
  'canvasBackdropTone',
  'interfaceScale',
  'showThumbnailMetadata',
  'reduceMotion',
  'enableGpuAcceleration',
  'studioPreviewMaxDim',
  'thumbnailPreviewMode',
  'customCacheDir',
  'customDetectorModelPath',
  'customRecognizerModelPath',
  'customSegmenterModelPath',
  'defaultDeliveryMode',
  'defaultRenderFormat',
  'defaultJpegQuality',
  'defaultExportBaseDir',
  'createExportProjectSubfolder',
  'defaultNamingPattern',
  'excludeSlateFromClientPack',
  'archiveSlatesSeparately',
];

export function countStudioPreferenceChanges(
  a: StudioPreferences,
  b: StudioPreferences,
): number {
  let count = 0;
  for (const key of PREFERENCE_KEYS) {
    if (a[key] !== b[key]) {
      count += 1;
    }
  }
  return count;
}

export function areStudioPreferencesEqual(
  a: StudioPreferences,
  b: StudioPreferences,
): boolean {
  return countStudioPreferenceChanges(a, b) === 0;
}

export function isStudioPreferencesModified(prefs: StudioPreferences): boolean {
  return !areStudioPreferencesEqual(prefs, DEFAULT_STUDIO_PREFERENCES);
}
