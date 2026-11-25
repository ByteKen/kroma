import { MotionConfig } from 'framer-motion';
import {
  lazy,
  Suspense,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { BrowserPreviewGateway } from '../adapters/gateways/browserPreviewGateway';
import {
  getSharedTauriStudioGateway,
  isTauriDesktopEnvironment,
} from '../adapters/gateways/tauriStudioGateway';
import { DarkroomTooltipProvider } from '../features/darkroom-kit/DarkroomTooltip';
import { SplashScreen } from '../features/shell/SplashScreen';
import { StatusFilmstripBar } from '../features/shell/StatusFilmstripBar';
import { TopCommandBar } from '../features/shell/TopCommandBar';
import { WelcomeStartScreen } from '../features/shell/WelcomeStartScreen';
import { WorkspaceRail } from '../features/shell/WorkspaceRail';
import { WorkspaceTransitionSkeleton } from '../features/shell/WorkspaceTransitionSkeleton';
import {
  shouldShowWorkspaceSkeleton,
  type SkeletonWorkspaceView,
} from '../features/shell/workspaceTransition';
import { StudioProvider, useStudio, type WorkspaceView } from './StudioContext';

// Code-split heavy workspaces so initial boot parses only the shell + splash screen
const loadClusterGridWorkspace = () =>
  import('../features/culling/ClusterGridWorkspace').then((m) => ({
    default: m.ClusterGridWorkspace,
  }));
const loadSlateAnchorColumn = () =>
  import('../features/slates/SlateAnchorColumn').then((m) => ({
    default: m.SlateAnchorColumn,
  }));
const loadBatchDevelopStudio = () =>
  import('../features/develop/BatchDevelopStudio').then((m) => ({
    default: m.BatchDevelopStudio,
  }));
const loadBatchExportWorkspace = () =>
  import('../features/export/BatchExportWorkspace').then((m) => ({
    default: m.BatchExportWorkspace,
  }));
const loadSlateOcrWorkspace = () =>
  import('../features/ocr/SlateOcrWorkspace').then((m) => ({
    default: m.SlateOcrWorkspace,
  }));
const loadAmbiguousReviewWorkspace = () =>
  import('../features/review/AmbiguousReviewWorkspace').then((m) => ({
    default: m.AmbiguousReviewWorkspace,
  }));
const loadSettingsWorkspace = () =>
  import('../features/settings/SettingsWorkspace').then((m) => ({
    default: m.SettingsWorkspace,
  }));

const ClusterGridWorkspace = lazy(loadClusterGridWorkspace);
const SlateAnchorColumn = lazy(loadSlateAnchorColumn);
const BatchDevelopStudio = lazy(loadBatchDevelopStudio);
const BatchExportWorkspace = lazy(loadBatchExportWorkspace);
const SlateOcrWorkspace = lazy(loadSlateOcrWorkspace);
const AmbiguousReviewWorkspace = lazy(loadAmbiguousReviewWorkspace);
const SettingsWorkspace = lazy(loadSettingsWorkspace);

const WORKSPACE_SKELETON_MS = 90;

function StudioShellLayout() {
  const { isBootReady, session, activeWorkspace, settingsOpen, studioFocusMode } = useStudio();
  const sessionKey = session ? (session.folderPath ?? '__session__') : null;

  // Keep Settings mounted after first open so close exit animations stay smooth
  const [hasMountedSettings, setHasMountedSettings] = useState(false);
  useEffect(() => {
    if (settingsOpen) {
      setHasMountedSettings(true);
    }
  }, [settingsOpen]);

  // Pre-warm primary Studio & Face Organizer chunks during idle time after boot finishes
  useEffect(() => {
    if (!isBootReady) return;
    const warm = () => {
      void loadBatchDevelopStudio();
      void loadClusterGridWorkspace();
      void loadSlateAnchorColumn();
    };
    const idleApi = window as Window & {
      requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number;
      cancelIdleCallback?: (id: number) => void;
    };
    if (idleApi.requestIdleCallback) {
      const id = idleApi.requestIdleCallback(warm, { timeout: 1500 });
      return () => idleApi.cancelIdleCallback?.(id);
    }
    const timer = window.setTimeout(warm, 600);
    return () => window.clearTimeout(timer);
  }, [isBootReady]);

  // Keep heavy workspaces (Studio & Face Organizer) cached in memory once visited for the active
  // project so switching tabs is instantaneous with zero DOM teardown or texture re-decoding.
  const [cachedWorkspaces, setCachedWorkspaces] = useState<{
    readonly sessionKey: string | null;
    readonly develop: boolean;
    readonly culling: boolean;
  }>({ sessionKey: null, develop: false, culling: false });

  useEffect(() => {
    setCachedWorkspaces((prev) => {
      if (prev.sessionKey !== sessionKey) {
        return {
          sessionKey,
          develop: sessionKey !== null && activeWorkspace === 'develop',
          culling: sessionKey !== null && activeWorkspace === 'culling',
        };
      }
      const nextDevelop = prev.develop || (sessionKey !== null && activeWorkspace === 'develop');
      const nextCulling = prev.culling || (sessionKey !== null && activeWorkspace === 'culling');
      if (nextDevelop === prev.develop && nextCulling === prev.culling) {
        return prev;
      }
      return { sessionKey, develop: nextDevelop, culling: nextCulling };
    });
  }, [sessionKey, activeWorkspace]);

  const showHome = !session || activeWorkspace === 'home';
  const effectiveView: WorkspaceView = showHome ? 'home' : activeWorkspace;

  const previousViewRef = useRef<WorkspaceView | null>(null);
  const [transitionSkeleton, setTransitionSkeleton] = useState<SkeletonWorkspaceView | null>(null);

  const isTargetAlreadyMounted =
    effectiveView === 'home' ||
    (effectiveView === 'develop' &&
      cachedWorkspaces.sessionKey === sessionKey &&
      cachedWorkspaces.develop) ||
    (effectiveView === 'culling' &&
      cachedWorkspaces.sessionKey === sessionKey &&
      cachedWorkspaces.culling);

  // Trigger layout-matched skeleton synchronously before paint ONLY for unmounted lazy views;
  // already-cached workspaces (Home, Studio, Face Organizer) switch instantaneously in 0 ms.
  useLayoutEffect(() => {
    const prev = previousViewRef.current;
    previousViewRef.current = effectiveView;

    if (!shouldShowWorkspaceSkeleton(prev, effectiveView, isTargetAlreadyMounted)) {
      setTransitionSkeleton(null);
      return;
    }

    setTransitionSkeleton(effectiveView);
    const timer = window.setTimeout(() => {
      setTransitionSkeleton(null);
    }, WORKSPACE_SKELETON_MS);

    return () => window.clearTimeout(timer);
  }, [effectiveView, isTargetAlreadyMounted]);

  const mountDevelop =
    Boolean(session) &&
    (activeWorkspace === 'develop' ||
      (cachedWorkspaces.sessionKey === sessionKey && cachedWorkspaces.develop));
  const mountCulling =
    Boolean(session) &&
    (activeWorkspace === 'culling' ||
      (cachedWorkspaces.sessionKey === sessionKey && cachedWorkspaces.culling));

  const hideShellChrome =
    studioFocusMode && !showHome && !settingsOpen && activeWorkspace === 'develop';

  return (
    <div className="h-full w-full flex flex-col overflow-hidden bg-canvas text-ink-primary">
      {!hideShellChrome && <TopCommandBar />}

      <div className="flex-1 flex min-h-0 overflow-hidden">
        {!hideShellChrome && <WorkspaceRail />}

        <main className="flex-1 flex min-w-0 min-h-0 overflow-hidden relative">
          {showHome && (
            <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden">
              <WelcomeStartScreen />
            </div>
          )}

          {mountCulling && (
            <div
              key="culling-workspace"
              className={
                !showHome && activeWorkspace === 'culling'
                  ? 'flex-1 flex min-w-0 min-h-0 overflow-hidden'
                  : 'hidden'
              }
            >
              <Suspense fallback={<WorkspaceTransitionSkeleton workspace="culling" />}>
                <SlateAnchorColumn />
                <ClusterGridWorkspace />
              </Suspense>
            </div>
          )}

          {mountDevelop && (
            <div
              key="develop-workspace"
              className={
                !showHome && activeWorkspace === 'develop'
                  ? 'flex-1 flex min-w-0 min-h-0 overflow-hidden'
                  : 'hidden'
              }
            >
              <Suspense fallback={<WorkspaceTransitionSkeleton workspace="develop" />}>
                <BatchDevelopStudio
                  isActive={!showHome && !settingsOpen && activeWorkspace === 'develop'}
                />
              </Suspense>
            </div>
          )}

          {!showHome && activeWorkspace === 'slate_ocr' && (
            <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden">
              <Suspense fallback={<WorkspaceTransitionSkeleton workspace="slate_ocr" />}>
                <SlateOcrWorkspace />
              </Suspense>
            </div>
          )}

          {!showHome && activeWorkspace === 'review_queue' && (
            <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden">
              <Suspense fallback={<WorkspaceTransitionSkeleton workspace="review_queue" />}>
                <AmbiguousReviewWorkspace />
              </Suspense>
            </div>
          )}

          {!showHome && activeWorkspace === 'export' && (
            <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden">
              <Suspense fallback={<WorkspaceTransitionSkeleton workspace="export" />}>
                <BatchExportWorkspace />
              </Suspense>
            </div>
          )}

          <WorkspaceTransitionSkeleton workspace={transitionSkeleton} />
        </main>
      </div>

      {!hideShellChrome && <StatusFilmstripBar />}

      {hasMountedSettings && (
        <Suspense fallback={null}>
          <SettingsWorkspace />
        </Suspense>
      )}
    </div>
  );
}

function MotionPreferenceBoundary({ children }: { readonly children: ReactNode }) {
  const { preferences } = useStudio();

  useEffect(() => {
    const scale = preferences.interfaceScale;
    document.documentElement.style.zoom = scale === 100 ? '' : `${scale}%`;
    return () => {
      document.documentElement.style.zoom = '';
    };
  }, [preferences.interfaceScale]);

  return (
    <MotionConfig reducedMotion={preferences.reduceMotion ? 'always' : 'user'}>
      {children}
    </MotionConfig>
  );
}

export function App() {
  // Composition root: uses a shared TauriStudioGateway inside the desktop app (surviving dev HMR
  // reloads without dropping loaded-shoot state or hot RAW buffers), and falls back to a read-only
  // BrowserPreviewGateway when opened in a regular browser.
  const gateway = useMemo(
    () =>
      isTauriDesktopEnvironment()
        ? getSharedTauriStudioGateway()
        : new BrowserPreviewGateway(),
    [],
  );

  return (
    <DarkroomTooltipProvider>
      <StudioProvider gateway={gateway}>
        <MotionPreferenceBoundary>
          <SplashScreen />
          <StudioShellLayout />
        </MotionPreferenceBoundary>
      </StudioProvider>
    </DarkroomTooltipProvider>
  );
}
