import {
  classifyMatchConfidence,
  DEFAULT_REVIEW_WARNING_THRESHOLD,
  needsVisualMatchReview,
} from '../../domain/identity';
import type { PhotoAsset } from '../../domain/photo';
import {
  matchesVirtualFolderFilter,
  type VirtualFolderFilter,
} from '../../domain/virtualFolder';

export type SmartBinFilter = 'all' | 'unassigned' | 'focus_drops' | 'ocr_matched';

export interface ClusterFilterCriteria {
  readonly sfaceThreshold: number;
  readonly reviewWarningThreshold?: number;
  readonly activeDirectoryId: string | null;
  readonly activeVirtualFolderId?: VirtualFolderFilter;
  readonly allowedVirtualFolderIds?: ReadonlySet<string> | null;
  readonly smartBin: SmartBinFilter;
  readonly searchQuery: string;
}

export function filterClusterPhotos(
  photos: readonly PhotoAsset[],
  criteria: ClusterFilterCriteria,
): readonly PhotoAsset[] {
  const normalizedQuery = criteria.searchQuery.trim().toLowerCase();
  const reviewThreshold = criteria.reviewWarningThreshold ?? DEFAULT_REVIEW_WARNING_THRESHOLD;
  const virtualFolderFilter = criteria.activeVirtualFolderId ?? 'all';

  return photos.filter((photo) => {
    if (criteria.activeDirectoryId && photo.directoryId !== criteria.activeDirectoryId) {
      return false;
    }

    if (
      !matchesVirtualFolderFilter(
        photo.virtualFolderId,
        virtualFolderFilter,
        criteria.allowedVirtualFolderIds ?? null,
      )
    ) {
      return false;
    }

    if (criteria.smartBin === 'focus_drops' && photo.sharpnessPct >= 65) {
      return false;
    }

    if (
      criteria.smartBin === 'unassigned' &&
      !needsVisualMatchReview(photo, reviewThreshold)
    ) {
      return false;
    }

    if (
      criteria.smartBin === 'ocr_matched' &&
      !photo.isSlateAnchor &&
      photo.sfaceCosineSimilarity < reviewThreshold
    ) {
      return false;
    }

    if (!photo.isSlateAnchor) {
      const confidence = classifyMatchConfidence(
        photo.sfaceCosineSimilarity,
        criteria.sfaceThreshold,
      );
      if (confidence === 'unmatched') {
        return false;
      }
    }

    if (normalizedQuery.length > 0) {
      const haystack = [
        photo.fileName,
        photo.exif.cameraModel,
        photo.exif.lensModel,
        photo.isSlateAnchor ? 'slate anchor whiteboard' : '',
        photo.heroSelect ? 'hero select' : '',
        photo.eyeAfLocked ? 'eye af locked' : '',
      ]
        .join(' ')
        .toLowerCase();

      if (!haystack.includes(normalizedQuery)) {
        return false;
      }
    }

    return true;
  });
}
