import { sortRecentProjects, type RecentProjectEntry } from '../../domain/recentProjects';

const STORAGE_KEY = 'kroma.recentProjects.v1';

function isValidRecentEntry(raw: unknown): raw is RecentProjectEntry {
  if (!raw || typeof raw !== 'object') return false;
  const candidate = raw as Record<string, unknown>;
  return (
    typeof candidate.id === 'string' &&
    typeof candidate.projectName === 'string' &&
    typeof candidate.folderPath === 'string' &&
    typeof candidate.photoCount === 'number' &&
    typeof candidate.clusterCount === 'number' &&
    typeof candidate.isClustered === 'boolean' &&
    Array.isArray(candidate.coverThumbnails) &&
    typeof candidate.lastOpenedAtIso === 'string'
  );
}

export function loadRecentProjectsFromStorage(): readonly RecentProjectEntry[] {
  if (typeof window === 'undefined' || !window.localStorage) return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const valid = parsed.filter(isValidRecentEntry).map((entry) => ({
      ...entry,
      isPinned: Boolean(entry.isPinned),
      coverThumbnails: entry.coverThumbnails
        .filter((u): u is string => typeof u === 'string' && u.length > 0)
        .slice(0, 3),
    }));
    return sortRecentProjects(valid);
  } catch {
    return [];
  }
}

export function saveRecentProjectsToStorage(projects: readonly RecentProjectEntry[]): void {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(projects));
  } catch {
    // Ignore storage quota or private-mode restrictions cleanly
  }
}
