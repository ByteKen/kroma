import type { PhotoAsset } from './photo';

export type MatchConfidence = 'verified' | 'needs_review' | 'unmatched';

export const DEFAULT_SFACE_THRESHOLD = 0.44;
export const DEFAULT_REVIEW_BAND_DELTA = 0.09;
/**
 * Visual confidence floor for already-grouped face matches.
 * ArcFace 512-D same-person studio shots typically score 75%–92%; look-alike cluster merges
 * pull group-average scores into ~60%–72%, so flagging < 75% surfaces collisions immediately.
 */
export const DEFAULT_REVIEW_WARNING_THRESHOLD = 0.75;

/**
 * Classifies an SFace cosine similarity score against the active darkroom threshold.
 * Note: UI telemetry percentages display `similarity * 100` (e.g. 0.991 -> 99.1%), while
 * raw SFace thresholds typically sit in [0.30, 0.65].
 */
export function classifyMatchConfidence(
  cosineSimilarity: number,
  verifiedThreshold: number,
  reviewBandDelta: number = DEFAULT_REVIEW_BAND_DELTA,
): MatchConfidence {
  const clampedVerified = Math.min(1, Math.max(0, verifiedThreshold));
  const reviewFloor = Math.max(0, clampedVerified - Math.max(0, reviewBandDelta));

  if (cosineSimilarity >= clampedVerified) {
    return 'verified';
  }
  if (cosineSimilarity >= reviewFloor) {
    return 'needs_review';
  }
  return 'unmatched';
}

export function formatSimilarityPercent(cosineSimilarity: number): string {
  const pct = Math.min(100, Math.max(0, cosineSimilarity * 100));
  return `${pct.toFixed(1)}%`;
}

/**
 * Returns true when a grouped, face-detected non-anchor photo falls below the visual
 * review threshold (default 75%) and should be highlighted orange for quick human verification.
 */
export function needsVisualMatchReview(
  photo: Pick<PhotoAsset, 'isSlateAnchor' | 'faceOverlay' | 'sfaceCosineSimilarity'>,
  reviewWarningThreshold: number = DEFAULT_REVIEW_WARNING_THRESHOLD,
): boolean {
  if (photo.isSlateAnchor || photo.faceOverlay === null) {
    return false;
  }
  return photo.sfaceCosineSimilarity < reviewWarningThreshold;
}

export function countClusterReviewWarnings(
  cluster: SubjectCluster,
  reviewWarningThreshold: number = DEFAULT_REVIEW_WARNING_THRESHOLD,
  approvedPhotoIds?: ReadonlySet<string>,
): number {
  if (cluster.anchor.id.startsWith('_')) return 0;
  let count = 0;
  for (const photo of cluster.photos) {
    if (approvedPhotoIds?.has(photo.id)) continue;
    if (needsVisualMatchReview(photo, reviewWarningThreshold)) {
      count += 1;
    }
  }
  return count;
}

export function isFaceOnlySlateCode(slateCode: string): boolean {
  const normalized = slateCode.trim().toUpperCase();
  return normalized.length === 0 || normalized === 'NO SLATE';
}

export interface SlateAnchor {
  readonly id: string;
  readonly slateCode: string;
  /** Auto-generated sequential subject/slate number (`#001`, `#002`, ...) available for all subjects. */
  readonly subjectCode?: string;
  readonly subjectName: string;
  readonly ocrExtractedText: string;
  readonly ocrConfidencePct: number;
  readonly anchorPhotoId: string;
  readonly thumbnailUrl: string;
  readonly capturedAtLabel: string;
  readonly sessionTimeRange: string;
  readonly cameraLabel: string;
  readonly lensLabel: string;
  readonly wbCalibrated: boolean;
  readonly wbCalibrationLabel: string;
  readonly reportedTotalExposures: number;
  readonly ambiguousCount: number;
}

export interface SubjectCluster {
  readonly anchor: SlateAnchor;
  readonly photos: readonly PhotoAsset[];
}

export interface ManualMatchSuggestion {
  readonly cluster: SubjectCluster;
  readonly proximityLabel: string | null;
  readonly isRecommended: boolean;
}

function normalizeFileStem(fileName: string): string {
  const dot = fileName.lastIndexOf('.');
  let stem = (dot > 0 ? fileName.slice(0, dot) : fileName).trim().toLowerCase();
  stem = stem.replace(/\s*\(\d+\)$/, '');
  return stem.replace(/([_-](edited|edit|retouched|graded|final|export|web))+$/g, '');
}

function parseCameraSeq(fileName: string): { prefix: string; seq: number } | null {
  const stem = normalizeFileStem(fileName);
  const match = /^(.*?)(\d{3,7})$/.exec(stem);
  if (!match) return null;
  return {
    prefix: match[1].replace(/[_-\s]+$/, ''),
    seq: Number.parseInt(match[2], 10),
  };
}

/**
 * Pure domain ranker for Manual Match suggestions: surfaces subjects that share the exact same
 * camera stem (`DSC03440.ARW` <-> `DSC03440.jpg`), tight EXIF shutter time, or adjacent camera
 * sequence numbers at the top so fixing 1-photo singletons takes a single click.
 */
export function rankManualMatchSuggestions(
  sourcePhotos: readonly PhotoAsset[],
  candidateClusters: readonly SubjectCluster[],
): readonly ManualMatchSuggestion[] {
  const srcStems = new Set(sourcePhotos.map((p) => normalizeFileStem(p.fileName)));
  const srcTimes = sourcePhotos
    .map((p) => p.capturedAtEpochSec)
    .filter((t): t is number => typeof t === 'number' && Number.isFinite(t));
  const srcSeqs = sourcePhotos
    .map((p) => parseCameraSeq(p.fileName))
    .filter((s): s is { prefix: string; seq: number } => s !== null);

  const scored = candidateClusters.map((cluster, idx) => {
    let bestRank = 10_000 + idx;
    let proximityLabel: string | null = null;
    let isRecommended = false;

    for (const p of cluster.photos) {
      const stem = normalizeFileStem(p.fileName);
      if (srcStems.has(stem)) {
        bestRank = 0;
        proximityLabel = 'Same frame';
        isRecommended = true;
        break;
      }

      if (typeof p.capturedAtEpochSec === 'number' && srcTimes.length > 0) {
        for (const t of srcTimes) {
          const dt = Math.abs(p.capturedAtEpochSec - t);
          if (dt <= 90 && dt < bestRank) {
            bestRank = 10 + dt;
            proximityLabel = dt === 0 ? 'Same second' : `${dt}s apart`;
            isRecommended = true;
          }
        }
      }

      const seq = parseCameraSeq(p.fileName);
      if (seq && srcSeqs.length > 0) {
        for (const s of srcSeqs) {
          if (s.prefix === seq.prefix) {
            const ds = Math.abs(seq.seq - s.seq);
            const seqRank = 200 + ds;
            if (ds <= 8 && seqRank < bestRank) {
              bestRank = seqRank;
              proximityLabel = ds === 1 ? '1 frame apart' : `${ds} frames apart`;
              isRecommended = true;
            }
          }
        }
      }
    }

    return { cluster, proximityLabel, isRecommended, bestRank, idx };
  });

  scored.sort((a, b) => a.bestRank - b.bestRank || a.idx - b.idx);
  return scored.map(({ cluster, proximityLabel, isRecommended }) => ({
    cluster,
    proximityLabel,
    isRecommended,
  }));
}

export type ShootTimelineSortMode = 'filename' | 'timestamp' | 'nearest';

export interface ShootTimelinePhoto {
  readonly photo: PhotoAsset;
  readonly clusterId: string;
  readonly clusterName: string;
  readonly clusterPhotoCount: number;
  readonly isInActiveCluster: boolean;
  readonly proximityLabel: string | null;
  readonly proximityRank: number;
}

/**
 * Natural, case-insensitive A–Z comparator (`"Folder 2"` before `"Folder 10"`, `"a"` == `"A"`).
 */
export function compareNaturalAlphabetical(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

/**
 * Sorts subject clusters alphabetically A–Z by display name (case-insensitive, natural numeric),
 * keeping special system buckets (`_group_photos`, `_no_face_photos`) pinned at the end.
 */
export function compareSubjectClustersAlphabetically(
  a: SubjectCluster,
  b: SubjectCluster,
): number {
  const aSpecial = a.anchor.id.startsWith('_');
  const bSpecial = b.anchor.id.startsWith('_');
  if (aSpecial !== bSpecial) {
    return aSpecial ? 1 : -1;
  }
  return (
    compareNaturalAlphabetical(a.anchor.subjectName, b.anchor.subjectName) ||
    compareNaturalAlphabetical(a.anchor.id, b.anchor.id)
  );
}

/**
 * Sorts photos naturally A–Z by filename (with deterministic ID tie-breaker).
 */
export function comparePhotosAlphabetically(a: PhotoAsset, b: PhotoAsset): number {
  return (
    compareNaturalAlphabetical(a.fileName, b.fileName) ||
    compareNaturalAlphabetical(a.id, b.id)
  );
}

const naturalFileCompare = compareNaturalAlphabetical;

/**
 * Flattens all photos across the shoot (clusters + review queue) into a single unorganized
 * chronological / filename-ordered browser for side-by-side manual matching against `activeClusterId`.
 */
export function buildUnorganizedShootTimeline(
  clusters: readonly SubjectCluster[],
  activeClusterId: string,
  sortMode: ShootTimelineSortMode,
  pendingReviewPhotos: readonly PhotoAsset[] = [],
): readonly ShootTimelinePhoto[] {
  const activeCluster = clusters.find((c) => c.anchor.id === activeClusterId);
  const activePhotos = activeCluster?.photos ?? [];

  const activeStems = new Set(activePhotos.map((p) => normalizeFileStem(p.fileName)));
  const activeTimes = activePhotos
    .map((p) => p.capturedAtEpochSec)
    .filter((t): t is number => typeof t === 'number' && Number.isFinite(t));
  const activeSeqs = activePhotos
    .map((p) => parseCameraSeq(p.fileName))
    .filter((s): s is { prefix: string; seq: number } => s !== null);

  const scoreProximity = (
    photo: PhotoAsset,
    isInActive: boolean,
  ): { label: string | null; rank: number } => {
    if (isInActive) {
      return { label: 'Matched', rank: -1 };
    }
    const stem = normalizeFileStem(photo.fileName);
    if (activeStems.has(stem)) {
      return { label: 'Same frame', rank: 0 };
    }

    let bestRank = 10_000;
    let label: string | null = null;

    if (typeof photo.capturedAtEpochSec === 'number' && activeTimes.length > 0) {
      for (const t of activeTimes) {
        const dt = Math.abs(photo.capturedAtEpochSec - t);
        if (dt <= 90 && 10 + dt < bestRank) {
          bestRank = 10 + dt;
          label = dt === 0 ? 'Same sec' : `${dt}s apart`;
        }
      }
    }

    const seq = parseCameraSeq(photo.fileName);
    if (seq && activeSeqs.length > 0) {
      for (const s of activeSeqs) {
        if (s.prefix === seq.prefix) {
          const ds = Math.abs(seq.seq - s.seq);
          if (ds <= 8 && 200 + ds < bestRank) {
            bestRank = 200 + ds;
            label = ds === 1 ? '1 frame apart' : `${ds} frames apart`;
          }
        }
      }
    }

    return { label, rank: bestRank };
  };

  const seenIds = new Set<string>();
  const items: ShootTimelinePhoto[] = [];

  for (const cluster of clusters) {
    const isInActive = cluster.anchor.id === activeClusterId;
    for (const photo of cluster.photos) {
      if (seenIds.has(photo.id)) continue;
      seenIds.add(photo.id);
      const prox = scoreProximity(photo, isInActive);
      items.push({
        photo,
        clusterId: cluster.anchor.id,
        clusterName: cluster.anchor.subjectName,
        clusterPhotoCount: cluster.photos.length,
        isInActiveCluster: isInActive,
        proximityLabel: prox.label,
        proximityRank: prox.rank,
      });
    }
  }

  for (const photo of pendingReviewPhotos) {
    if (seenIds.has(photo.id)) continue;
    seenIds.add(photo.id);
    const prox = scoreProximity(photo, false);
    items.push({
      photo,
      clusterId: '_review_queue',
      clusterName: 'Needs Review',
      clusterPhotoCount: 1,
      isInActiveCluster: false,
      proximityLabel: prox.label,
      proximityRank: prox.rank,
    });
  }

  items.sort((a, b) => {
    if (sortMode === 'nearest') {
      if (a.proximityRank !== b.proximityRank) {
        return a.proximityRank - b.proximityRank;
      }
      return naturalFileCompare(a.photo.fileName, b.photo.fileName);
    }
    if (sortMode === 'timestamp') {
      const ta = a.photo.capturedAtEpochSec ?? Number.MAX_SAFE_INTEGER;
      const tb = b.photo.capturedAtEpochSec ?? Number.MAX_SAFE_INTEGER;
      if (ta !== tb) return ta - tb;
      return naturalFileCompare(a.photo.fileName, b.photo.fileName);
    }
    // Default 'filename': natural alphanumeric order pairs RAW/Edited frames & camera counters side-by-side
    const byName = naturalFileCompare(a.photo.fileName, b.photo.fileName);
    if (byName !== 0) return byName;
    return a.photo.id.localeCompare(b.photo.id);
  });

  return items;
}
