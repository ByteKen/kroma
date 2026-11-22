export interface OcrTextLine {
  readonly text: string;
  readonly confidence: number;
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

export interface PhotoOcrResult {
  readonly photoId: string;
  readonly fileName: string;
  readonly previewUrl: string;
  readonly proposedName: string | null;
  readonly proposedCode: string | null;
  readonly confidence: number;
  readonly lines: readonly OcrTextLine[];
}

export interface ClusterSlateOcrResult {
  readonly clusterId: string;
  readonly currentName: string;
  readonly slatePhotoId: string | null;
  readonly slateFileName: string | null;
  readonly slatePreviewUrl: string | null;
  readonly proposedName: string | null;
  readonly proposedCode: string | null;
  readonly confidence: number;
  readonly scannedFrameCount: number;
  readonly lines: readonly OcrTextLine[];
}

export interface SlateRenameApplyItem {
  readonly clusterId: string;
  readonly nextName: string;
  readonly subjectCode?: string | null;
  readonly slatePhotoId?: string | null;
  readonly ocrRawText?: string | null;
  readonly ocrConfidence?: number | null;
  readonly markAsSlatePhoto: boolean;
}

export type SlateOcrConfidenceTier = 'ready' | 'check' | 'none';
export type SlateOcrFilter = 'all' | 'ready' | 'check' | 'none' | 'applied';

export const HIGH_CONFIDENCE_OCR_THRESHOLD = 0.85;
export const CHECK_CONFIDENCE_OCR_THRESHOLD = 0.55;

export function classifySlateOcrConfidence(
  proposedName: string | null,
  confidence: number,
): SlateOcrConfidenceTier {
  if (!proposedName || !proposedName.trim() || confidence < CHECK_CONFIDENCE_OCR_THRESHOLD) {
    return 'none';
  }
  return confidence >= HIGH_CONFIDENCE_OCR_THRESHOLD ? 'ready' : 'check';
}

export function isSlateOcrAlreadyApplied(
  currentName: string,
  draftName: string,
): boolean {
  const curr = currentName.trim().toLowerCase();
  const draft = draftName.trim().toLowerCase();
  return draft.length > 0 && curr === draft;
}

export interface SlateOcrScanOptions {
  readonly scanAllFrames?: boolean;
  readonly clusterIds?: readonly string[];
}

/**
 * Merges newly scanned subject OCR results into previously scanned results so targeted
 * or incremental scans never discard earlier whiteboard reads.
 */
export function mergeSlateOcrResults(
  existing: readonly ClusterSlateOcrResult[],
  incoming: readonly ClusterSlateOcrResult[],
): readonly ClusterSlateOcrResult[] {
  if (incoming.length === 0) return existing;
  const byId = new Map<string, ClusterSlateOcrResult>();
  for (const item of existing) {
    byId.set(item.clusterId, item);
  }
  for (const item of incoming) {
    byId.set(item.clusterId, item);
  }
  return Array.from(byId.values());
}

export interface SlateOcrRowEntry {
  readonly clusterId: string;
  readonly currentName: string;
  readonly draftName: string;
  readonly draftCode: string;
  readonly confidence: number;
  readonly proposedName: string | null;
  readonly isApplied: boolean;
}

export function filterSlateOcrRows<T extends SlateOcrRowEntry>(
  rows: readonly T[],
  filter: SlateOcrFilter,
  searchQuery: string,
): readonly T[] {
  const q = searchQuery.trim().toLowerCase();
  return rows.filter((row) => {
    if (filter !== 'all') {
      if (filter === 'applied') {
        if (!row.isApplied) return false;
      } else {
        const tier = classifySlateOcrConfidence(row.proposedName, row.confidence);
        if (tier !== filter) return false;
      }
    }
    if (!q) return true;
    return (
      row.currentName.toLowerCase().includes(q) ||
      row.draftName.toLowerCase().includes(q) ||
      row.draftCode.toLowerCase().includes(q)
    );
  });
}
