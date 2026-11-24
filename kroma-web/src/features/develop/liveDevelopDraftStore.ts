import { useSyncExternalStore } from 'react';
import type { DevelopRecipe } from '../../domain/develop';

export interface LiveDevelopDraftEntry {
  readonly photoId: string;
  readonly beforeRecipe: DevelopRecipe;
  readonly recipe: DevelopRecipe;
  readonly label: string;
}

type Listener = () => void;

/**
 * Lightweight external store for continuous 60–120 Hz Studio Develop drags (sliders, tone curves,
 * color wheels, crop handles, mask brush strokes).
 *
 * Why this lives outside React Context state:
 * - Mutating `StudioContext.session` on every pointer-move tick forces the entire app tree
 *   (including hidden Organizer grids, WorkspaceRail's O(N) edited-count scan, TopCommandBar, and
 *   StatusFilmstripBar) to re-render at 120 Hz.
 * - Storing the in-flight drag draft here keeps `StudioProvider` completely quiet during drags;
 *   only subscribers watching the active photo re-render, and the single coalesced delta commits
 *   into `StudioContext.session` + `developHistory` once the drag settles or before any discrete
 *   action (Undo/Redo, Copy/Paste, photo switch, tab switch, export).
 */
export interface LiveDevelopDraftStore {
  getDraft: () => LiveDevelopDraftEntry | null;
  getRecipeForPhoto: (photoId: string, fallback: DevelopRecipe) => DevelopRecipe;
  setDraft: (entry: {
    readonly photoId: string;
    readonly beforeRecipe: DevelopRecipe;
    readonly recipe: DevelopRecipe;
    readonly label: string;
  }) => LiveDevelopDraftEntry | null;
  takePendingCommit: () => LiveDevelopDraftEntry | null;
  clear: () => void;
  subscribe: (listener: Listener) => () => void;
}

export function createLiveDevelopDraftStore(): LiveDevelopDraftStore {
  let current: LiveDevelopDraftEntry | null = null;
  const listeners = new Set<Listener>();

  const emit = () => {
    for (const listener of listeners) {
      listener();
    }
  };

  return {
    getDraft() {
      return current;
    },

    getRecipeForPhoto(photoId: string, fallback: DevelopRecipe): DevelopRecipe {
      return current && current.photoId === photoId ? current.recipe : fallback;
    },

    setDraft(next) {
      let displaced: LiveDevelopDraftEntry | null = null;
      let preservedBefore = next.beforeRecipe;

      if (current) {
        if (current.photoId === next.photoId) {
          // Preserve the original pre-drag recipe across all frames of the same drag burst
          // so Undo steps cleanly back to the state before the drag started.
          preservedBefore = current.beforeRecipe;
        } else {
          // Switching photos mid-drag: hand the previous photo's draft back to the caller to flush
          displaced = current;
        }
      }

      current = {
        photoId: next.photoId,
        beforeRecipe: preservedBefore,
        recipe: next.recipe,
        label: next.label,
      };
      emit();
      return displaced;
    },

    takePendingCommit() {
      if (!current) return null;
      const pending = current;
      current = null;
      emit();
      return pending;
    },

    clear() {
      if (!current) return;
      current = null;
      emit();
    },

    subscribe(listener: Listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

export const sharedLiveDevelopDraftStore = createLiveDevelopDraftStore();

/**
 * Subscribes a Studio component to the live 120 Hz draft recipe for `photoId` without triggering
 * re-renders in `StudioProvider` or unrelated workspaces.
 */
export function useLiveDevelopRecipe(
  photoId: string | null | undefined,
  committedRecipe: DevelopRecipe | null | undefined,
): DevelopRecipe | null {
  return useSyncExternalStore(
    sharedLiveDevelopDraftStore.subscribe,
    () => {
      if (!photoId || !committedRecipe) return committedRecipe ?? null;
      return sharedLiveDevelopDraftStore.getRecipeForPhoto(photoId, committedRecipe);
    },
    () => committedRecipe ?? null,
  );
}
