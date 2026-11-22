import { cacheParsedCubeLut3D, parseCubeLut3D } from '../../domain/cubeLut3d';

const STORAGE_KEY = 'kroma.studio.customLuts.v1';

export interface CustomCubeLutEntry {
  readonly id: string;
  readonly name: string;
  readonly cubePath: string;
  readonly gridSize: number;
  /** Optional inline `.cube` text persisted for browser preview or quick warmup. */
  readonly inlineContent?: string;
}

export function loadCustomLutsFromStorage(): readonly CustomCubeLutEntry[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    const result: CustomCubeLutEntry[] = [];
    for (const item of parsed) {
      if (
        item &&
        typeof item === 'object' &&
        typeof (item as CustomCubeLutEntry).id === 'string' &&
        typeof (item as CustomCubeLutEntry).name === 'string' &&
        typeof (item as CustomCubeLutEntry).cubePath === 'string'
      ) {
        const rec = item as CustomCubeLutEntry;
        if (typeof rec.inlineContent === 'string' && rec.inlineContent.length > 0) {
          try {
            cacheParsedCubeLut3D(rec.cubePath, parseCubeLut3D(rec.inlineContent));
          } catch {
            // Ignore malformed cached inline content
          }
        }
        result.push({
          id: rec.id,
          name: rec.name.trim() || 'Custom.cube',
          cubePath: rec.cubePath,
          gridSize: typeof rec.gridSize === 'number' && rec.gridSize >= 2 ? rec.gridSize : 33,
          inlineContent: rec.inlineContent,
        });
      }
    }
    return result;
  } catch {
    return [];
  }
}

export function saveCustomLutsToStorage(luts: readonly CustomCubeLutEntry[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(luts));
  } catch {
    // Ignore localStorage quota errors gracefully
  }
}
