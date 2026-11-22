import type { WorkspaceView } from '../../app/StudioContext';

export type SkeletonWorkspaceView = WorkspaceView;

/**
 * Determines whether transitioning from `previous` to `next` should display
 * a brief layout-matched skeleton overlay.
 */
export function shouldShowWorkspaceSkeleton(
  previous: WorkspaceView | null,
  next: WorkspaceView,
): next is SkeletonWorkspaceView {
  return previous !== null && previous !== next;
}
