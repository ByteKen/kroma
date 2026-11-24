import type { WorkspaceView } from '../../app/StudioContext';

export type SkeletonWorkspaceView = WorkspaceView;

/**
 * Determines whether transitioning from `previous` to `next` should display
 * a brief layout-matched skeleton overlay. Already-mounted workspaces switch
 * instantaneously (`alreadyMounted = true`) with zero artificial delay.
 */
export function shouldShowWorkspaceSkeleton(
  previous: WorkspaceView | null,
  next: WorkspaceView,
  alreadyMounted = false,
): next is SkeletonWorkspaceView {
  if (alreadyMounted) return false;
  return previous !== null && previous !== next;
}
