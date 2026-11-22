import type { DevelopRecipe } from './develop';

export interface PhotoRecipeDiff {
  readonly photoId: string;
  readonly before: DevelopRecipe;
  readonly after: DevelopRecipe;
}

export interface DevelopHistoryEntry {
  readonly label: string;
  readonly coalesceKey?: string;
  readonly timestampMs: number;
  readonly changes: readonly PhotoRecipeDiff[];
}

export interface DevelopHistoryStack {
  readonly undoStack: readonly DevelopHistoryEntry[];
  readonly redoStack: readonly DevelopHistoryEntry[];
}

export const MAX_DEVELOP_HISTORY = 50;
export const DEFAULT_COALESCE_WINDOW_MS = 650;

export const EMPTY_DEVELOP_HISTORY: DevelopHistoryStack = {
  undoStack: [],
  redoStack: [],
};

/**
 * Records a non-destructive Studio develop change onto a bounded undo/redo stack.
 * Continuous slider/curve drags with the same `coalesceKey` within `coalesceWindowMs`
 * merge into a single clean undo step preserving the original `before` recipe.
 */
export function recordDevelopHistory(
  stack: DevelopHistoryStack,
  entry: DevelopHistoryEntry,
  maxDepth: number = MAX_DEVELOP_HISTORY,
  coalesceWindowMs: number = DEFAULT_COALESCE_WINDOW_MS,
): DevelopHistoryStack {
  if (entry.changes.length === 0) {
    return stack;
  }

  const prevUndo = stack.undoStack;
  const top = prevUndo[prevUndo.length - 1];

  if (
    top &&
    entry.coalesceKey &&
    top.coalesceKey === entry.coalesceKey &&
    entry.timestampMs - top.timestampMs <= coalesceWindowMs &&
    top.changes.length === 1 &&
    entry.changes.length === 1 &&
    top.changes[0].photoId === entry.changes[0].photoId
  ) {
    const merged: DevelopHistoryEntry = {
      label: entry.label,
      coalesceKey: entry.coalesceKey,
      timestampMs: entry.timestampMs,
      changes: [
        {
          photoId: top.changes[0].photoId,
          before: top.changes[0].before,
          after: entry.changes[0].after,
        },
      ],
    };
    return {
      undoStack: [...prevUndo.slice(0, -1), merged],
      redoStack: [],
    };
  }

  const nextUndo = [...prevUndo, entry];
  const boundedUndo =
    nextUndo.length > maxDepth ? nextUndo.slice(nextUndo.length - maxDepth) : nextUndo;

  return {
    undoStack: boundedUndo,
    redoStack: [],
  };
}

export function undoDevelopHistory(stack: DevelopHistoryStack): {
  readonly nextStack: DevelopHistoryStack;
  readonly entry: DevelopHistoryEntry | null;
} {
  if (stack.undoStack.length === 0) {
    return { nextStack: stack, entry: null };
  }
  const entry = stack.undoStack[stack.undoStack.length - 1];
  return {
    nextStack: {
      undoStack: stack.undoStack.slice(0, -1),
      redoStack: [...stack.redoStack, entry],
    },
    entry,
  };
}

export function redoDevelopHistory(stack: DevelopHistoryStack): {
  readonly nextStack: DevelopHistoryStack;
  readonly entry: DevelopHistoryEntry | null;
} {
  if (stack.redoStack.length === 0) {
    return { nextStack: stack, entry: null };
  }
  const entry = stack.redoStack[stack.redoStack.length - 1];
  return {
    nextStack: {
      undoStack: [...stack.undoStack, entry],
      redoStack: stack.redoStack.slice(0, -1),
    },
    entry,
  };
}
