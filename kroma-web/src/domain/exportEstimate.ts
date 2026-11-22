import type { PhotoAsset } from './photo';

export type ExportDeliveryMode = 'develop' | 'develop_and_raw' | 'copy' | 'move';
export type ExportRenderFormat = 'jpeg' | 'png' | 'tiff16';

export interface ExportSizeEstimate {
  readonly extraDiskBytes: number;
  readonly editedBytes: number;
  readonly originalsBytes: number;
  readonly totalSourceBytes: number;
  readonly isMoveRelocation: boolean;
  readonly isExact: boolean;
  readonly sampledPhotoCount: number;
  readonly formattedExtraSize: string;
  readonly formattedTotalPayload: string;
  readonly formattedEditedSize: string;
  readonly formattedOriginalsSize: string;
  readonly formattedTotalSourceSize: string;
}

export interface MeasuredExportSizePayload {
  readonly extraDiskBytes: number;
  readonly editedBytes: number;
  readonly originalsBytes: number;
  readonly totalSourceBytes: number;
  readonly fileCount: number;
  readonly sampledPhotoCount: number;
  readonly isExact: boolean;
}

const MB = 1024 * 1024;
const GB = 1024 * MB;

const WEB_IMAGE_EXTS = new Set(['jpg', 'jpeg', 'webp']);
const PNG_EXTS = new Set(['png']);
const TIFF_EXTS = new Set(['tif', 'tiff']);

type SourceKind = 'jpeg' | 'png' | 'tiff' | 'raw';

function classifySourceKind(fileName: string): SourceKind {
  const ext = fileName.split('.').pop()?.toLowerCase() ?? '';
  if (WEB_IMAGE_EXTS.has(ext)) return 'jpeg';
  if (PNG_EXTS.has(ext)) return 'png';
  if (TIFF_EXTS.has(ext)) return 'tiff';
  return 'raw';
}

function fallbackOriginalBytes(kind: SourceKind): number {
  if (kind === 'jpeg') return 4.5 * MB;
  if (kind === 'png') return 3.0 * MB;
  if (kind === 'tiff') return 36 * MB;
  return 28 * MB;
}

function jpegQualityMultiplier(jpegQuality: number): number {
  // Normalized factor centered at 1.0 for quality=94: ~0.42 at 70%, ~1.0 at 94%, ~1.55 at 100%
  const q = Math.min(100, Math.max(50, jpegQuality));
  const normalized = (q - 70) / 30;
  return 0.42 + 1.13 * Math.pow(Math.max(0, normalized), 1.75);
}

/**
 * Estimates rendered bytes per photo proportional to its real on-disk source size & format
 * when `fileSizeBytes` is known (so 1 MB PNGs never get overestimated like 28 MB camera RAWs).
 */
function estimateRenderedPhotoBytes(
  kind: SourceKind,
  sourceBytes: number,
  format: ExportRenderFormat,
  jpegQuality: number,
): number {
  const qMult = jpegQualityMultiplier(jpegQuality);
  switch (kind) {
    case 'png':
      if (format === 'jpeg') return sourceBytes * 0.28 * qMult;
      if (format === 'png') return sourceBytes * 1.02;
      return sourceBytes * 4.5;
    case 'jpeg':
      if (format === 'jpeg') return sourceBytes * 0.95 * qMult;
      if (format === 'png') return sourceBytes * 3.2;
      return sourceBytes * 12.0;
    case 'tiff':
      if (format === 'jpeg') return sourceBytes * 0.08 * qMult;
      if (format === 'png') return sourceBytes * 0.35;
      return sourceBytes * 1.0;
    case 'raw':
    default:
      if (format === 'jpeg') return sourceBytes * 0.16 * qMult;
      if (format === 'png') return sourceBytes * 0.65;
      return sourceBytes * 2.5;
  }
}

export function formatByteSize(bytes: number, exact = false): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB';
  const prefix = exact ? '' : '~';
  if (bytes >= GB) {
    const gb = bytes / GB;
    return `${prefix}${gb >= 100 ? Math.round(gb) : gb.toFixed(1)} GB`;
  }
  const mb = bytes / MB;
  if (mb >= 1) {
    return `${prefix}${mb >= 100 ? Math.round(mb) : mb.toFixed(1)} MB`;
  }
  return `${prefix}${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function formatApproxSize(bytes: number): string {
  return formatByteSize(bytes, false);
}

export function estimateExportSize(
  photos: readonly Pick<PhotoAsset, 'fileName' | 'fileSizeBytes'>[],
  mode: ExportDeliveryMode,
  format: ExportRenderFormat,
  jpegQuality: number,
): ExportSizeEstimate {
  const count = photos.length;
  if (count === 0) {
    return {
      extraDiskBytes: 0,
      editedBytes: 0,
      originalsBytes: 0,
      totalSourceBytes: 0,
      isMoveRelocation: mode === 'move',
      isExact: true,
      sampledPhotoCount: 0,
      formattedExtraSize: '0 MB',
      formattedTotalPayload: '0 MB',
      formattedEditedSize: '0 MB',
      formattedOriginalsSize: '0 MB',
      formattedTotalSourceSize: '0 MB',
    };
  }

  let totalSourceBytes = 0;
  let editedBytes = 0;
  let allOriginalSizesKnown = true;
  const rendersEdited = mode === 'develop' || mode === 'develop_and_raw';

  for (const photo of photos) {
    const kind = classifySourceKind(photo.fileName);
    const hasExactSize = typeof photo.fileSizeBytes === 'number' && photo.fileSizeBytes > 0;
    if (!hasExactSize) {
      allOriginalSizesKnown = false;
    }
    const sourceBytes = hasExactSize ? photo.fileSizeBytes! : fallbackOriginalBytes(kind);
    totalSourceBytes += sourceBytes;
    if (rendersEdited) {
      editedBytes += estimateRenderedPhotoBytes(kind, sourceBytes, format, jpegQuality);
    }
  }

  if (mode === 'move') {
    return {
      extraDiskBytes: 0,
      editedBytes: 0,
      originalsBytes: totalSourceBytes,
      totalSourceBytes,
      isMoveRelocation: true,
      isExact: allOriginalSizesKnown,
      sampledPhotoCount: 0,
      formattedExtraSize: '0 MB extra',
      formattedTotalPayload: formatByteSize(totalSourceBytes, allOriginalSizesKnown),
      formattedEditedSize: '0 MB',
      formattedOriginalsSize: formatByteSize(totalSourceBytes, allOriginalSizesKnown),
      formattedTotalSourceSize: formatByteSize(totalSourceBytes, allOriginalSizesKnown),
    };
  }

  const originalsIncludedBytes = mode === 'develop' ? 0 : totalSourceBytes;
  const extraDiskBytes =
    mode === 'develop'
      ? editedBytes
      : mode === 'develop_and_raw'
        ? editedBytes + totalSourceBytes
        : totalSourceBytes;
  const isExact = mode === 'copy' && allOriginalSizesKnown;

  return {
    extraDiskBytes,
    editedBytes,
    originalsBytes: originalsIncludedBytes,
    totalSourceBytes,
    isMoveRelocation: false,
    isExact,
    sampledPhotoCount: 0,
    formattedExtraSize: formatByteSize(extraDiskBytes, isExact),
    formattedTotalPayload: formatByteSize(extraDiskBytes, isExact),
    formattedEditedSize: formatByteSize(editedBytes, false),
    formattedOriginalsSize: formatByteSize(originalsIncludedBytes, allOriginalSizesKnown),
    formattedTotalSourceSize: formatByteSize(totalSourceBytes, allOriginalSizesKnown),
  };
}

export function fromMeasuredExportSize(
  measured: MeasuredExportSizePayload,
  mode: ExportDeliveryMode,
): ExportSizeEstimate {
  const isMove = mode === 'move';
  const totalSourceFormatted = formatByteSize(measured.totalSourceBytes, true);
  if (isMove) {
    return {
      extraDiskBytes: 0,
      editedBytes: 0,
      originalsBytes: measured.originalsBytes,
      totalSourceBytes: measured.totalSourceBytes,
      isMoveRelocation: true,
      isExact: true,
      sampledPhotoCount: measured.sampledPhotoCount,
      formattedExtraSize: '0 MB extra',
      formattedTotalPayload: totalSourceFormatted,
      formattedEditedSize: '0 MB',
      formattedOriginalsSize: totalSourceFormatted,
      formattedTotalSourceSize: totalSourceFormatted,
    };
  }

  return {
    extraDiskBytes: measured.extraDiskBytes,
    editedBytes: measured.editedBytes,
    originalsBytes: measured.originalsBytes,
    totalSourceBytes: measured.totalSourceBytes,
    isMoveRelocation: false,
    isExact: measured.isExact,
    sampledPhotoCount: measured.sampledPhotoCount,
    formattedExtraSize: formatByteSize(measured.extraDiskBytes, measured.isExact),
    formattedTotalPayload: formatByteSize(measured.extraDiskBytes, measured.isExact),
    formattedEditedSize: formatByteSize(measured.editedBytes, measured.isExact),
    formattedOriginalsSize: formatByteSize(measured.originalsBytes, true),
    formattedTotalSourceSize: totalSourceFormatted,
  };
}

/**
 * Builds a clean sibling default export folder path beside the source project
 * so exported client deliveries never mix into the untouched source RAW folder.
 */
export function buildDefaultExportLocation(
  projectFolderPath: string | undefined,
  projectName: string,
): string {
  const safeName = `${projectName.trim() || 'Kroma_Project'}_Export`;
  if (!projectFolderPath || !projectFolderPath.trim()) {
    return safeName;
  }

  const trimmed = projectFolderPath.trim().replace(/[\\/]+$/, '');
  const usesBackslash = trimmed.includes('\\') && !trimmed.includes('/');
  const sep = usesBackslash ? '\\' : '/';
  const lastSep = Math.max(trimmed.lastIndexOf('\\'), trimmed.lastIndexOf('/'));
  if (lastSep <= 0) {
    return `${trimmed}${sep}${safeName}`;
  }
  const parent = trimmed.slice(0, lastSep);
  return `${parent}${sep}${safeName}`;
}

export function joinExportPath(baseDir: string, folderName: string): string {
  const cleanBase = baseDir.trim().replace(/[\\/]+$/, '');
  const cleanSub = folderName.trim().replace(/^[\\/]+|[\\/]+$/g, '');
  if (!cleanBase) return cleanSub;
  if (!cleanSub) return cleanBase;
  const sep = cleanBase.includes('\\') && !cleanBase.includes('/') ? '\\' : '/';
  return `${cleanBase}${sep}${cleanSub}`;
}
