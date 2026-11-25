import {
  DEFAULT_STUDIO_PREFERENCES,
  normalizeStudioPreferences,
  type StudioPreferences,
} from '../../domain/preferences';

const STORAGE_KEY = 'kroma.studioPreferences.v1';
const ALPHABETICAL_SORT_MIGRATED_KEY = 'kroma.migrated.alphabeticalDefault.v1';

export function loadStudioPreferencesFromStorage(): StudioPreferences {
  if (typeof window === 'undefined' || !window.localStorage) {
    return { ...DEFAULT_STUDIO_PREFERENCES };
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      window.localStorage.setItem(ALPHABETICAL_SORT_MIGRATED_KEY, '1');
      return { ...DEFAULT_STUDIO_PREFERENCES };
    }
    const parsed: unknown = JSON.parse(raw);
    let normalized = normalizeStudioPreferences(parsed);
    // One-time upgrade for installs that persisted the previous factory default ('capture_time')
    // before 'alphabetical' existed, while preserving any explicit choice made afterwards.
    if (!window.localStorage.getItem(ALPHABETICAL_SORT_MIGRATED_KEY)) {
      window.localStorage.setItem(ALPHABETICAL_SORT_MIGRATED_KEY, '1');
      if (normalized.defaultSubjectSortMode === 'capture_time') {
        normalized = { ...normalized, defaultSubjectSortMode: 'alphabetical' };
        saveStudioPreferencesToStorage(normalized);
      }
    }
    return normalized;
  } catch {
    return { ...DEFAULT_STUDIO_PREFERENCES };
  }
}

export function saveStudioPreferencesToStorage(preferences: StudioPreferences): void {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  } catch {
    // Ignore storage quota or private-mode restrictions cleanly
  }
}
