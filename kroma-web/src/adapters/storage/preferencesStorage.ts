import {
  DEFAULT_STUDIO_PREFERENCES,
  normalizeStudioPreferences,
  type StudioPreferences,
} from '../../domain/preferences';

const STORAGE_KEY = 'kroma.studioPreferences.v1';

export function loadStudioPreferencesFromStorage(): StudioPreferences {
  if (typeof window === 'undefined' || !window.localStorage) {
    return { ...DEFAULT_STUDIO_PREFERENCES };
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_STUDIO_PREFERENCES };
    const parsed: unknown = JSON.parse(raw);
    return normalizeStudioPreferences(parsed);
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
