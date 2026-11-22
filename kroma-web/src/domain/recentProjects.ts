export interface RecentProjectEntry {
  readonly id: string;
  readonly projectName: string;
  readonly folderPath: string;
  readonly photoCount: number;
  readonly clusterCount: number;
  readonly isClustered: boolean;
  readonly coverThumbnails: readonly string[];
  readonly lastOpenedAtIso: string;
  readonly isPinned: boolean;
}

export const MAX_RECENT_PROJECTS = 12;

export function normalizeProjectPathKey(folderPath: string): string {
  return folderPath.trim().replace(/[\\/]+/g, '/').replace(/\/$/, '').toLowerCase();
}

export function sortRecentProjects(
  projects: readonly RecentProjectEntry[],
): readonly RecentProjectEntry[] {
  return [...projects].sort((a, b) => {
    if (a.isPinned !== b.isPinned) {
      return a.isPinned ? -1 : 1;
    }
    const ta = Date.parse(a.lastOpenedAtIso) || 0;
    const tb = Date.parse(b.lastOpenedAtIso) || 0;
    return tb - ta;
  });
}

export function upsertRecentProject(
  projects: readonly RecentProjectEntry[],
  incoming: Omit<RecentProjectEntry, 'id' | 'isPinned'> & { readonly isPinned?: boolean },
  maxItems: number = MAX_RECENT_PROJECTS,
): readonly RecentProjectEntry[] {
  const key = normalizeProjectPathKey(incoming.folderPath);
  if (!key) return projects;

  const existing = projects.find((p) => p.id === key);
  const merged: RecentProjectEntry = {
    id: key,
    projectName: incoming.projectName.trim() || existing?.projectName || 'Untitled Project',
    folderPath: incoming.folderPath.trim(),
    photoCount: incoming.photoCount,
    clusterCount: incoming.clusterCount,
    isClustered: incoming.isClustered,
    coverThumbnails:
      incoming.coverThumbnails.length > 0
        ? incoming.coverThumbnails.slice(0, 3)
        : (existing?.coverThumbnails ?? []),
    lastOpenedAtIso: incoming.lastOpenedAtIso,
    isPinned: incoming.isPinned ?? existing?.isPinned ?? false,
  };

  const withoutOld = projects.filter((p) => p.id !== key);
  const sorted = sortRecentProjects([merged, ...withoutOld]);
  return sorted.slice(0, Math.max(1, maxItems));
}

export function removeRecentProject(
  projects: readonly RecentProjectEntry[],
  folderPathOrId: string,
): readonly RecentProjectEntry[] {
  const key = normalizeProjectPathKey(folderPathOrId);
  return projects.filter((p) => p.id !== key && p.folderPath !== folderPathOrId);
}

export function togglePinRecentProject(
  projects: readonly RecentProjectEntry[],
  folderPathOrId: string,
): readonly RecentProjectEntry[] {
  const key = normalizeProjectPathKey(folderPathOrId);
  const toggled = projects.map((p) =>
    p.id === key || p.folderPath === folderPathOrId ? { ...p, isPinned: !p.isPinned } : p,
  );
  return sortRecentProjects(toggled);
}

export type RecentProjectsFilter = 'all' | 'pinned' | 'clustered' | 'unclustered';
export type RecentProjectsSort = 'recent' | 'name' | 'photos';

export function filterRecentProjects(
  projects: readonly RecentProjectEntry[],
  query: string,
  filter: RecentProjectsFilter = 'all',
  sort: RecentProjectsSort = 'recent',
): readonly RecentProjectEntry[] {
  const q = query.trim().toLowerCase();
  const filtered = projects.filter((p) => {
    if (filter === 'pinned' && !p.isPinned) return false;
    if (filter === 'clustered' && !p.isClustered) return false;
    if (filter === 'unclustered' && p.isClustered) return false;
    if (q.length > 0) {
      const hay = `${p.projectName} ${p.folderPath}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  if (sort === 'name') {
    return [...filtered].sort(
      (a, b) =>
        Number(b.isPinned) - Number(a.isPinned) ||
        a.projectName.localeCompare(b.projectName, undefined, { numeric: true }),
    );
  }
  if (sort === 'photos') {
    return [...filtered].sort(
      (a, b) => Number(b.isPinned) - Number(a.isPinned) || b.photoCount - a.photoCount,
    );
  }
  return sortRecentProjects(filtered);
}

export function formatRelativeOpenedAt(isoString: string, nowMs: number = Date.now()): string {
  const ts = Date.parse(isoString);
  if (!Number.isFinite(ts)) return 'Recently';
  const diffSec = Math.max(0, Math.floor((nowMs - ts) / 1000));
  if (diffSec < 60) return 'Just now';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay === 1) return 'Yesterday';
  if (diffDay < 7) return `${diffDay}d ago`;
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
