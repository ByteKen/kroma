/**
 * Lightweight in-memory drag payload tracker that mirrors HTML5 DataTransfer payloads.
 *
 * WebView2 / Chromium restricts reading custom MIME payloads during `dragover` and can
 * normalize `dataTransfer.types`, so keeping the active drag payload in memory guarantees
 * instant drop-target validation and zero-failure drops across Tauri v2 windows.
 */
export type KromaDragPayload =
  | { readonly kind: 'photos'; readonly photoIds: readonly string[] }
  | { readonly kind: 'clusters'; readonly clusterIds: readonly string[] }
  | { readonly kind: 'vfolder'; readonly folderId: string };

let activeDragPayload: KromaDragPayload | null = null;

export function beginKromaDrag(dataTransfer: DataTransfer, payload: KromaDragPayload): void {
  activeDragPayload = payload;
  dataTransfer.effectAllowed = 'move';

  try {
    if (payload.kind === 'photos') {
      const json = JSON.stringify(payload.photoIds);
      dataTransfer.setData('application/x-kroma-photos', json);
      dataTransfer.setData('text/plain', `kroma:photos:${json}`);
    } else if (payload.kind === 'clusters') {
      const json = JSON.stringify(payload.clusterIds);
      dataTransfer.setData('application/x-kroma-clusters', json);
      if (payload.clusterIds.length === 1) {
        dataTransfer.setData('application/x-kroma-cluster', payload.clusterIds[0]);
      }
      dataTransfer.setData('text/plain', `kroma:clusters:${json}`);
    } else {
      dataTransfer.setData('application/x-kroma-vfolder', payload.folderId);
      dataTransfer.setData('text/plain', `kroma:vfolder:${payload.folderId}`);
    }
  } catch {
    // Some synthetic test DataTransfer objects don't implement all MIME types
  }
}

export function getActiveKromaDrag(): KromaDragPayload | null {
  return activeDragPayload;
}

export function clearActiveKromaDrag(): void {
  activeDragPayload = null;
}

export function readKromaDragFromDrop(dataTransfer: DataTransfer): KromaDragPayload | null {
  try {
    const photosRaw = dataTransfer.getData('application/x-kroma-photos');
    if (photosRaw) {
      const ids = JSON.parse(photosRaw) as string[];
      if (Array.isArray(ids) && ids.length > 0) {
        return { kind: 'photos', photoIds: ids };
      }
    }

    const clustersRaw = dataTransfer.getData('application/x-kroma-clusters');
    if (clustersRaw) {
      const ids = JSON.parse(clustersRaw) as string[];
      if (Array.isArray(ids) && ids.length > 0) {
        return { kind: 'clusters', clusterIds: ids };
      }
    }

    const singleCluster = dataTransfer.getData('application/x-kroma-cluster');
    if (singleCluster) {
      return { kind: 'clusters', clusterIds: [singleCluster] };
    }

    const folderId = dataTransfer.getData('application/x-kroma-vfolder');
    if (folderId) {
      return { kind: 'vfolder', folderId };
    }

    const plain = dataTransfer.getData('text/plain');
    if (plain.startsWith('kroma:photos:')) {
      const ids = JSON.parse(plain.slice('kroma:photos:'.length)) as string[];
      if (Array.isArray(ids) && ids.length > 0) return { kind: 'photos', photoIds: ids };
    } else if (plain.startsWith('kroma:clusters:')) {
      const ids = JSON.parse(plain.slice('kroma:clusters:'.length)) as string[];
      if (Array.isArray(ids) && ids.length > 0) return { kind: 'clusters', clusterIds: ids };
    } else if (plain.startsWith('kroma:vfolder:')) {
      const id = plain.slice('kroma:vfolder:'.length);
      if (id) return { kind: 'vfolder', folderId: id };
    }
  } catch {
    // Fall back to in-memory payload
  }

  return activeDragPayload;
}
