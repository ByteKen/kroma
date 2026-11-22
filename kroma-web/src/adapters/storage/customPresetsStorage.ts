import { hydrateDevelopRecipe, type DevelopPreset, type DevelopRecipe } from '../../domain/develop';

const STORAGE_KEY = 'kroma.studio.customPresets.v1';

interface StoredPresetRecord {
  readonly id: string;
  readonly name: string;
  readonly recipe: Partial<DevelopRecipe>;
}

/**
 * Outer-layer localStorage adapter persisting user-saved Develop Presets across sessions.
 * Hydrates every preset through `hydrateDevelopRecipe` so older presets automatically inherit
 * defaults for any newly added Develop modules.
 */
export function loadCustomPresetsFromStorage(): readonly DevelopPreset[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    const result: DevelopPreset[] = [];
    for (const item of parsed) {
      if (
        item &&
        typeof item === 'object' &&
        typeof (item as StoredPresetRecord).id === 'string' &&
        typeof (item as StoredPresetRecord).name === 'string'
      ) {
        const rec = item as StoredPresetRecord;
        result.push({
          id: rec.id,
          name: rec.name.trim() || 'Untitled Preset',
          category: 'Custom',
          recipe: hydrateDevelopRecipe(rec.recipe),
          isCustom: true,
        });
      }
    }
    return result;
  } catch {
    return [];
  }
}

export function saveCustomPresetsToStorage(presets: readonly DevelopPreset[]): void {
  try {
    const records: StoredPresetRecord[] = presets.map((p) => ({
      id: p.id,
      name: p.name,
      recipe: p.recipe,
    }));
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Ignore quota / private mode storage failures gracefully
  }
}
