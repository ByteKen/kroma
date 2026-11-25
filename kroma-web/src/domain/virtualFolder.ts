export interface VirtualFolder {
  readonly id: string;
  readonly name: string;
  readonly parentId: string | null;
}

/**
 * Virtual folder filter selection in the Folder Tree:
 * - `'all'` shows every photo/subject in the project
 * - `'unassigned'` shows only photos without any virtual folder
 * - any other string filters to that folder ID and all of its nested subfolders
 */
export type VirtualFolderFilter = 'all' | 'unassigned' | (string & {});

export interface VirtualFolderTreeNode {
  readonly folder: VirtualFolder;
  readonly depth: number;
  readonly children: readonly VirtualFolderTreeNode[];
}

export interface FolderPhotoCounts {
  readonly all: number;
  readonly unassigned: number;
  readonly directByFolderId: ReadonlyMap<string, number>;
  readonly rolledUpByFolderId: ReadonlyMap<string, number>;
}

export function buildVirtualFolderTree(
  folders: readonly VirtualFolder[],
): readonly VirtualFolderTreeNode[] {
  const byId = new Map(folders.map((f) => [f.id, f]));
  const childrenByParent = new Map<string | null, VirtualFolder[]>();

  for (const folder of folders) {
    const validParent =
      folder.parentId && folder.parentId !== folder.id && byId.has(folder.parentId)
        ? folder.parentId
        : null;
    const list = childrenByParent.get(validParent) ?? [];
    list.push(folder);
    childrenByParent.set(validParent, list);
  }

  for (const list of childrenByParent.values()) {
    list.sort((a, b) =>
      a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }),
    );
  }

  const visited = new Set<string>();
  const buildLevel = (parentId: string | null, depth: number): VirtualFolderTreeNode[] => {
    const items = childrenByParent.get(parentId) ?? [];
    const nodes: VirtualFolderTreeNode[] = [];
    for (const folder of items) {
      if (visited.has(folder.id)) continue;
      visited.add(folder.id);
      nodes.push({
        folder,
        depth,
        children: buildLevel(folder.id, depth + 1),
      });
    }
    return nodes;
  };

  return buildLevel(null, 0);
}

/** Returns the folder itself plus every nested descendant folder ID. */
export function collectDescendantFolderIds(
  folders: readonly VirtualFolder[],
  rootFolderId: string,
): ReadonlySet<string> {
  const result = new Set<string>([rootFolderId]);
  let expanded = true;
  while (expanded) {
    expanded = false;
    for (const folder of folders) {
      if (folder.parentId && result.has(folder.parentId) && !result.has(folder.id)) {
        result.add(folder.id);
        expanded = true;
      }
    }
  }
  return result;
}

/** Returns `true` if `candidateChildId` sits inside `ancestorId`'s subtree. */
export function isDescendantFolder(
  folders: readonly VirtualFolder[],
  candidateChildId: string,
  ancestorId: string,
): boolean {
  if (candidateChildId === ancestorId) return true;
  const byId = new Map(folders.map((f) => [f.id, f]));
  const seen = new Set<string>();
  let cursor: string | null = candidateChildId;
  while (cursor && !seen.has(cursor)) {
    seen.add(cursor);
    const current = byId.get(cursor);
    if (!current) return false;
    if (current.parentId === ancestorId) return true;
    cursor = current.parentId;
  }
  return false;
}

/** Resolves ordered folder names from root down to `folderId`, e.g. `['Engineering', 'BSARC']`. */
export function resolveVirtualFolderPath(
  folders: readonly VirtualFolder[],
  folderId: string | null | undefined,
): readonly string[] {
  if (!folderId) return [];
  const byId = new Map(folders.map((f) => [f.id, f]));
  const segments: string[] = [];
  const seen = new Set<string>();
  let cursor: string | null = folderId;
  while (cursor && !seen.has(cursor)) {
    seen.add(cursor);
    const folder = byId.get(cursor);
    if (!folder) break;
    segments.push(folder.name);
    cursor = folder.parentId;
  }
  segments.reverse();
  return segments;
}

export function formatVirtualFolderBreadcrumb(
  folders: readonly VirtualFolder[],
  folderId: string | null | undefined,
): string {
  return resolveVirtualFolderPath(folders, folderId).join(' / ');
}

export function matchesVirtualFolderFilter(
  photoFolderId: string | null | undefined,
  filter: VirtualFolderFilter,
  allowedFolderIds: ReadonlySet<string> | null,
): boolean {
  if (filter === 'all') return true;
  if (filter === 'unassigned') return !photoFolderId;
  if (!photoFolderId) return false;
  return allowedFolderIds ? allowedFolderIds.has(photoFolderId) : photoFolderId === filter;
}

export function computeFolderPhotoCounts(
  folders: readonly VirtualFolder[],
  photos: readonly { readonly virtualFolderId?: string | null }[],
): FolderPhotoCounts {
  const validIds = new Set(folders.map((f) => f.id));
  const direct = new Map<string, number>();
  let unassigned = 0;

  for (const photo of photos) {
    const fid = photo.virtualFolderId ?? null;
    if (fid && validIds.has(fid)) {
      direct.set(fid, (direct.get(fid) ?? 0) + 1);
    } else {
      unassigned += 1;
    }
  }

  const rolledUp = new Map<string, number>();
  for (const folder of folders) {
    const descendants = collectDescendantFolderIds(folders, folder.id);
    let sum = 0;
    for (const id of descendants) {
      sum += direct.get(id) ?? 0;
    }
    rolledUp.set(folder.id, sum);
  }

  return {
    all: photos.length,
    unassigned,
    directByFolderId: direct,
    rolledUpByFolderId: rolledUp,
  };
}
