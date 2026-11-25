import type { PhotoAsset } from '../../domain/photo';
import type { IngestedDirectorySummary } from '../ports/studioGateway';

export interface FilteredDevelopFolderEntry {
  readonly directory: IngestedDirectorySummary;
  readonly totalPhotos: number;
  readonly editedPhotos: number;
  readonly visiblePhotos: readonly PhotoAsset[];
  /** `true` when the query matched photo filename(s) inside this folder rather than its folder name. */
  readonly autoExpand: boolean;
}

/**
 * Filters physical disk folders in Studio's Folders sidebar by folder path or contained photo
 * filename. When a search matches filenames inside a folder, that folder auto-expands to surface
 * the matching photos immediately without expanding unrelated folder-name matches.
 */
export function filterDevelopFolders(
  directories: readonly IngestedDirectorySummary[],
  photosByDirId: ReadonlyMap<string, readonly PhotoAsset[]>,
  editedCountByDirId: ReadonlyMap<string, number>,
  query: string,
): readonly FilteredDevelopFolderEntry[] {
  const normalizedQuery = query.trim().toLowerCase();
  const results: FilteredDevelopFolderEntry[] = [];

  for (const directory of directories) {
    const folderPhotos = photosByDirId.get(directory.id) ?? [];
    const totalPhotos = folderPhotos.length;
    const editedPhotos = editedCountByDirId.get(directory.id) ?? 0;

    if (normalizedQuery.length === 0) {
      results.push({
        directory,
        totalPhotos,
        editedPhotos,
        visiblePhotos: folderPhotos,
        autoExpand: false,
      });
      continue;
    }

    const nameMatches = directory.name.toLowerCase().includes(normalizedQuery);
    const matchingPhotos = folderPhotos.filter((photo) =>
      photo.fileName.toLowerCase().includes(normalizedQuery),
    );

    if (!nameMatches && matchingPhotos.length === 0) {
      continue;
    }

    if (!nameMatches && matchingPhotos.length > 0) {
      results.push({
        directory,
        totalPhotos,
        editedPhotos,
        visiblePhotos: matchingPhotos,
        autoExpand: true,
      });
    } else {
      results.push({
        directory,
        totalPhotos,
        editedPhotos,
        visiblePhotos: folderPhotos,
        autoExpand: false,
      });
    }
  }

  return results;
}
