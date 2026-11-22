import type { SubjectCluster } from '../../domain/identity';

export type SubjectFilterMode = 'all' | 'review' | 'single' | 'multi' | 'edited' | 'unedited';

export interface SubjectFilterCriteria {
  readonly query: string;
  readonly filter: SubjectFilterMode;
  readonly editedCountByClusterId?: ReadonlyMap<string, number>;
}

/**
 * Filters subject folders by quick status bucket and free-text query
 * (matching subject display name, slate ID, or any photo filename inside the subject).
 */
export function filterSubjectClusters(
  clusters: readonly SubjectCluster[],
  criteria: SubjectFilterCriteria,
): readonly SubjectCluster[] {
  const normalizedQuery = criteria.query.trim().toLowerCase();
  const { filter, editedCountByClusterId } = criteria;

  if (normalizedQuery.length === 0 && filter === 'all') {
    return clusters;
  }

  return clusters.filter((cluster) => {
    if (filter === 'review' && cluster.anchor.ambiguousCount === 0) {
      return false;
    }
    if (filter === 'single' && cluster.photos.length !== 1) {
      return false;
    }
    if (filter === 'multi' && cluster.photos.length < 2) {
      return false;
    }
    if (filter === 'edited' && (editedCountByClusterId?.get(cluster.anchor.id) ?? 0) === 0) {
      return false;
    }
    if (filter === 'unedited' && (editedCountByClusterId?.get(cluster.anchor.id) ?? 0) > 0) {
      return false;
    }

    if (normalizedQuery.length === 0) {
      return true;
    }

    if (cluster.anchor.subjectName.toLowerCase().includes(normalizedQuery)) {
      return true;
    }
    if (cluster.anchor.slateCode.toLowerCase().includes(normalizedQuery)) {
      return true;
    }
    if (cluster.anchor.subjectCode?.toLowerCase().includes(normalizedQuery)) {
      return true;
    }
    return cluster.photos.some((photo) =>
      photo.fileName.toLowerCase().includes(normalizedQuery),
    );
  });
}
