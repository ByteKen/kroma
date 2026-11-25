import { AnimatePresence, motion, type Variants } from 'framer-motion';
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useStudio } from '../../app/StudioContext';
import type {
  AiModelRole,
  ClearCacheTarget,
  ModelInspectionResult,
  StorageCacheSummary,
} from '../../application/ports/studioGateway';
import kromaLogo from '../../assets/kroma-logo.png';
import {
  checkGithubForAppUpdate,
  KROMA_FALLBACK_VERSION,
  KROMA_GITHUB_REPO,
  KROMA_GITHUB_URL,
  type AppUpdateCheckResult,
} from '../../domain/appUpdates';
import { formatByteSize } from '../../domain/exportEstimate';
import {
  formatShortcutCombo,
  formatShortcutKey,
  isMacPlatform,
  openExternalUrl,
} from '../../domain/platform';
import {
  countStudioPreferenceChanges,
  DEFAULT_STUDIO_PREFERENCES,
  isStudioPreferencesModified,
  normalizeStudioPreferences,
  type CanvasBackdropTone,
  type DefaultOpenWorkspace,
  type DefaultSubjectSortMode,
  type InterfaceScalePct,
  type StudioPreferences,
  type StudioPreviewMaxDim,
  type ThumbnailPreviewMode,
} from '../../domain/preferences';
import {
  DarkroomButton,
  DarkroomScrubber,
  DarkroomToggle,
  FAST_FADE,
  SOFT_SPRING,
  SegmentedControl,
  TACTILE_SPRING,
  type SegmentedOption,
} from '../darkroom-kit';
import { AppUpdateModal } from '../shell/AppUpdateModal';

type SettingsSectionId = 'general' | 'performance' | 'matching' | 'storage' | 'shortcuts' | 'about';

const SETTINGS_NAV: readonly {
  readonly id: SettingsSectionId;
  readonly label: string;
  readonly icon: string;
}[] = [
  { id: 'general', label: 'General', icon: 'tune' },
  { id: 'performance', label: 'Performance', icon: 'speed' },
  { id: 'matching', label: 'Models', icon: 'auto_awesome' },
  { id: 'storage', label: 'Storage', icon: 'hard_drive' },
  { id: 'shortcuts', label: 'Shortcuts', icon: 'keyboard' },
  { id: 'about', label: 'About', icon: 'info' },
];

const OPEN_WORKSPACE_OPTIONS: readonly SegmentedOption<DefaultOpenWorkspace>[] = [
  { id: 'develop', label: 'Studio', title: 'Start in Studio for immediate color & masking edits' },
  { id: 'culling', label: 'Face Organizer', title: 'Start in Face Organizer for already-grouped shoots' },
];

const SUBJECT_SORT_OPTIONS: readonly SegmentedOption<DefaultSubjectSortMode>[] = [
  { id: 'alphabetical', label: 'A–Z', title: 'Sort subjects alphabetically by name' },
  { id: 'capture_time', label: 'Time', title: 'Sort subjects by first shutter capture time' },
  { id: 'confidence', label: 'Confidence', title: 'Sort subjects needing review first' },
  { id: 'exposures', label: 'Photos', title: 'Sort subjects by largest photo count first' },
];

const CANVAS_BACKDROP_OPTIONS: readonly SegmentedOption<CanvasBackdropTone>[] = [
  { id: 'obsidian', label: 'Obsidian', title: 'Deep black surround (#0B0C0F)' },
  { id: 'charcoal', label: 'Charcoal', title: 'Kroma default darkroom surround (#111214)' },
  { id: 'neutral_gray', label: 'Neutral Gray', title: 'Color-neutral gray surround for accurate exposure & skin judgment (#2B2D31)' },
];

const INTERFACE_SCALE_OPTIONS: readonly SegmentedOption<'90' | '100' | '110'>[] = [
  { id: '90', label: '90%', title: 'Compact UI density' },
  { id: '100', label: '100%', title: 'Default UI scale' },
  { id: '110', label: '110%', title: 'Comfortable larger UI scale' },
];

type GpuMode = 'auto' | 'cpu';

const GPU_MODE_OPTIONS: readonly SegmentedOption<GpuMode>[] = [
  { id: 'auto', label: 'Auto (GPU)', title: 'Use GPU acceleration (DirectML on Windows, CoreML on macOS) with automatic CPU fallback' },
  { id: 'cpu', label: 'CPU Only', title: 'Disable GPU acceleration for AI models (useful on low-VRAM systems)' },
];

const PREVIEW_QUALITY_OPTIONS: readonly SegmentedOption<`${StudioPreviewMaxDim}`>[] = [
  { id: '1600', label: '1600px', title: 'Fastest RAW loading on laptops and large shoots' },
  { id: '2048', label: '2048px', title: 'Balanced sharpness and speed (Recommended)' },
  { id: '2560', label: '2560px', title: 'Extra detail for 4K / HiDPI studio monitors' },
];

const THUMBNAIL_PREVIEW_MODE_OPTIONS: readonly SegmentedOption<ThumbnailPreviewMode>[] = [
  {
    id: 'fast',
    label: 'Fast',
    title: 'Instant camera embedded previews with lightweight CSS approximation (lowest CPU/GPU usage)',
  },
  {
    id: 'edited_32bit',
    label: 'Edited 32-Bit',
    title: 'Render exact 32-bit Studio color, AI backdrops, masks & LUTs for edited cards in the background (Recommended)',
  },
  {
    id: 'all_32bit',
    label: 'All 32-Bit',
    title: 'Render all RAW & edited thumbnails through the 32-bit scene-linear pipeline in the background',
  },
];

interface ShortcutEntry {
  readonly action: string;
  readonly keys: readonly string[];
}

interface ShortcutGroup {
  readonly title: string;
  readonly items: readonly ShortcutEntry[];
}

const SHORTCUT_GROUPS: readonly ShortcutGroup[] = [
  {
    title: 'General',
    items: [
      { action: 'Open project folder', keys: ['Ctrl', 'O'] },
      { action: 'Open / close Settings', keys: ['Ctrl', ','] },
      { action: 'Undo edit or organization change', keys: ['Ctrl', 'Z'] },
      { action: 'Redo change', keys: ['Ctrl', 'Shift', 'Z'] },
      { action: 'Focus search / filter bar', keys: ['Ctrl', 'K'] },
    ],
  },
  {
    title: 'Studio',
    items: [
      { action: 'Toggle Focus Mode (hide toolbars)', keys: ['Tab'] },
      { action: 'Edit / Crop / Retouch / Mask / Presets tools', keys: ['E', 'R', 'Q', 'M', 'P'] },
      { action: 'White Balance eyedropper', keys: ['W'] },
      { action: 'Toggle retouch spot pins', keys: ['H'] },
      { action: 'Toggle shadow & highlight clipping', keys: ['J'] },
      { action: 'Toggle mask overlay', keys: ['O'] },
      { action: 'Decrease / increase brush size', keys: ['[', ']'] },
      { action: 'Copy / Paste edits', keys: ['Ctrl', 'Shift', 'C / V'] },
      { action: 'Open Sync Settings', keys: ['Ctrl', 'Shift', 'S'] },
      { action: 'Previous / Next photo', keys: ['←', '→'] },
      { action: 'Finish crop / exit active tool', keys: ['Enter', 'Esc'] },
    ],
  },
  {
    title: 'Lightbox & Face Organizer',
    items: [
      { action: 'Previous / Next photo', keys: ['←', '→'] },
      { action: 'Toggle 100% zoom', keys: ['Z'] },
      { action: 'Select / unselect photo', keys: ['Space'] },
      { action: 'Toggle face landmarks overlay', keys: ['F'] },
      { action: 'Open current photo in Studio', keys: ['E'] },
      { action: 'Close lightbox', keys: ['Esc'] },
    ],
  },
  {
    title: 'Needs Review',
    items: [
      { action: 'Approve match', keys: ['A'] },
      { action: 'Split into new person', keys: ['S'] },
      { action: 'Clear multi-selection', keys: ['Esc'] },
    ],
  },
];

const SECTION_CONTAINER_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 8 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.2,
      ease: [0.22, 1, 0.36, 1],
      staggerChildren: 0.04,
    },
  },
  exit: {
    opacity: 0,
    y: -5,
    transition: { duration: 0.12, ease: [0.4, 0, 1, 1] },
  },
};

const SECTION_ITEM_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 6 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.2, ease: [0.22, 1, 0.36, 1] },
  },
};

interface PendingModelCheck {
  readonly role: AiModelRole;
  readonly filePath: string;
  readonly fileName: string;
  readonly status: 'inspecting' | 'done';
  readonly report: ModelInspectionResult | null;
}

function basenameOfPath(path: string | null | undefined): string {
  if (!path) return 'Not installed';
  return path.split(/[\\/]/).pop() || path;
}

function detectWebGlGpuName(): string | null {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    if (!gl) return null;
    const debugExt = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugExt) {
      const unmasked = gl.getParameter(debugExt.UNMASKED_RENDERER_WEBGL);
      if (typeof unmasked === 'string' && unmasked.trim().length > 0) {
        const cleaned = unmasked
          .replace(/^ANGLE\s*\([^,]+,\s*/i, '')
          .replace(/\s*Direct3D.*$/i, '')
          .replace(/\)\s*$/, '')
          .trim();
        return cleaned || unmasked.trim();
      }
    }
    const renderer = gl.getParameter(gl.RENDERER);
    return typeof renderer === 'string' && renderer.trim().length > 0 ? renderer.trim() : null;
  } catch {
    return null;
  }
}

function SectionHeading({ children }: { readonly children: string }) {
  return (
    <h3 className="tabular-nums-mono text-[10px] uppercase tracking-[0.14em] text-ink-telemetry font-medium mb-1">
      {children}
    </h3>
  );
}

function SettingRow({
  title,
  description,
  control,
  last = false,
}: {
  readonly title: string;
  readonly description?: string;
  readonly control: ReactNode;
  readonly last?: boolean;
}) {
  return (
    <div
      className={`py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-6 ${
        last ? '' : 'border-b border-hairline/70'
      }`}
    >
      <div className="min-w-0 flex-1">
        <div className="font-display text-[13px] font-medium text-ink-primary">{title}</div>
        {description && (
          <p className="text-[11.5px] text-ink-secondary mt-0.5 leading-snug">{description}</p>
        )}
      </div>
      <div className="shrink-0 flex items-center">{control}</div>
    </div>
  );
}

export function SettingsWorkspace() {
  const {
    isDesktopRuntime,
    session,
    runtimeInfo,
    modelDownloadProgress,
    downloadingOptionalModelId,
    preferences,
    recentProjects,
    settingsOpen,
    lastClusteredThreshold,
    lastClusteredTemporalAssist,
    isScanningFolder,
    savePreferences,
    downloadOptionalModel,
    cancelOptionalModelDownload,
    clearRecentProjects,
    getStorageCacheSummary,
    pickCacheDirectory,
    clearStorageCache,
    pickOnnxModelFile,
    inspectOnnxModel,
    runFaceOrganizer,
    closeSettings,
  } = useStudio();

  const [activeSection, setActiveSection] = useState<SettingsSectionId>('general');
  const [draft, setDraft] = useState<StudioPreferences>(preferences);
  const [showAdvancedModels, setShowAdvancedModels] = useState(false);
  const [pendingModelCheck, setPendingModelCheck] = useState<PendingModelCheck | null>(null);
  const [isApplying, setIsApplying] = useState(false);
  const [confirmClosePrompt, setConfirmClosePrompt] = useState(false);

  // Storage tab live cache telemetry
  const [cacheSummary, setCacheSummary] = useState<StorageCacheSummary | null>(null);
  const [isLoadingCache, setIsLoadingCache] = useState(false);
  const [clearingCacheTarget, setClearingCacheTarget] = useState<ClearCacheTarget | null>(null);

  // Shortcuts filter
  const [shortcutQuery, setShortcutQuery] = useState('');

  // About / Updates state
  const [isCheckingUpdates, setIsCheckingUpdates] = useState(false);
  const [updateCheckResult, setUpdateCheckResult] = useState<AppUpdateCheckResult | null>(null);
  const [showReleaseNotesModal, setShowReleaseNotesModal] = useState(false);

  const isMac = useMemo(() => isMacPlatform(), []);
  const detectedGpuName = useMemo(() => (settingsOpen ? detectWebGlGpuName() : null), [settingsOpen]);
  const appVersion = runtimeInfo?.appVersion ?? KROMA_FALLBACK_VERSION;

  const refreshCacheSummary = useCallback(async () => {
    setIsLoadingCache(true);
    const summary = await getStorageCacheSummary();
    setCacheSummary(summary);
    setIsLoadingCache(false);
  }, [getStorageCacheSummary]);

  // Sync local draft cleanly each time Settings opens
  useEffect(() => {
    if (settingsOpen) {
      setDraft(preferences);
      setPendingModelCheck(null);
      setConfirmClosePrompt(false);
    } else {
      setShowReleaseNotesModal(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settingsOpen]);

  // Refresh disk cache summary whenever viewing the Storage tab
  useEffect(() => {
    if (settingsOpen && activeSection === 'storage') {
      void refreshCacheSummary();
    }
  }, [settingsOpen, activeSection, refreshCacheSummary]);

  const changedCount = useMemo(
    () => countStudioPreferenceChanges(draft, preferences),
    [draft, preferences],
  );
  const hasUnsavedChanges = changedCount > 0;
  const isDraftModified = isStudioPreferencesModified(draft);

  const patchDraft = useCallback((patch: Partial<StudioPreferences>) => {
    setDraft((prev) => normalizeStudioPreferences({ ...prev, ...patch }));
    setConfirmClosePrompt(false);
  }, []);

  const handleStageFactoryDefaults = useCallback(() => {
    setDraft({ ...DEFAULT_STUDIO_PREFERENCES });
    setPendingModelCheck(null);
    setConfirmClosePrompt(false);
  }, []);

  const handleDiscardChanges = useCallback(() => {
    setDraft(preferences);
    setPendingModelCheck(null);
    setConfirmClosePrompt(false);
  }, [preferences]);

  const handleApplyChanges = useCallback(
    async (closeAfter = false): Promise<boolean> => {
      if (isApplying) return false;
      setIsApplying(true);
      const ok = await savePreferences(draft);
      setIsApplying(false);
      if (ok) {
        setPendingModelCheck(null);
        setConfirmClosePrompt(false);
        if (activeSection === 'storage') {
          void refreshCacheSummary();
        }
        if (closeAfter) {
          closeSettings();
        }
      }
      return ok;
    },
    [isApplying, savePreferences, draft, activeSection, refreshCacheSummary, closeSettings],
  );

  const requestClose = useCallback(() => {
    if (hasUnsavedChanges) {
      setConfirmClosePrompt(true);
      return;
    }
    closeSettings();
  }, [hasUnsavedChanges, closeSettings]);

  const handleDiscardAndClose = useCallback(() => {
    setDraft(preferences);
    setPendingModelCheck(null);
    setConfirmClosePrompt(false);
    closeSettings();
  }, [preferences, closeSettings]);

  // Capture Escape so closing Settings never leaks to underlying workspaces
  useEffect(() => {
    if (!settingsOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        requestClose();
      }
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [settingsOpen, requestClose]);

  const clusteringParamsDirty =
    Boolean(session?.isClustered) &&
    (Math.abs(draft.sfaceThreshold - lastClusteredThreshold) >= 0.005 ||
      draft.useTemporalAssist !== lastClusteredTemporalAssist);

  const stageCustomModelPath = (role: AiModelRole, modelPath: string | null) => {
    if (role === 'detector') {
      patchDraft({ customDetectorModelPath: modelPath });
    } else if (role === 'recognizer') {
      patchDraft({ customRecognizerModelPath: modelPath });
    } else if (role === 'segmenter') {
      patchDraft({ customSegmenterModelPath: modelPath });
    } else if (role === 'face_parser') {
      patchDraft({ customFaceParserModelPath: modelPath });
    } else {
      patchDraft({ customLamaModelPath: modelPath });
    }
  };

  const handlePickCustomModel = async (role: AiModelRole) => {
    const filePath = await pickOnnxModelFile();
    if (!filePath) return;

    const fileName = basenameOfPath(filePath);
    setPendingModelCheck({
      role,
      filePath,
      fileName,
      status: 'inspecting',
      report: null,
    });

    const report = await inspectOnnxModel(filePath, role);
    if (!report) {
      setPendingModelCheck(null);
      return;
    }

    if (report.compatibility === 'compatible') {
      stageCustomModelPath(role, filePath);
    }

    setPendingModelCheck({
      role,
      filePath,
      fileName,
      status: 'done',
      report,
    });
  };

  const handleConfirmWarningModel = () => {
    if (!pendingModelCheck) return;
    stageCustomModelPath(pendingModelCheck.role, pendingModelCheck.filePath);
    setPendingModelCheck(null);
  };

  const handleResetModelToDefault = (role: AiModelRole) => {
    stageCustomModelPath(role, null);
    if (pendingModelCheck?.role === role) {
      setPendingModelCheck(null);
    }
  };

  const handleClearCache = async (target: ClearCacheTarget) => {
    setClearingCacheTarget(target);
    await clearStorageCache(target);
    await refreshCacheSummary();
    setClearingCacheTarget(null);
  };

  const handleCheckForUpdates = async () => {
    if (isCheckingUpdates) return;
    setIsCheckingUpdates(true);
    const result = await checkGithubForAppUpdate(appVersion);
    setUpdateCheckResult(result);
    setIsCheckingUpdates(false);
  };

  const filteredShortcutGroups = useMemo(() => {
    const q = shortcutQuery.trim().toLowerCase();
    if (!q) return SHORTCUT_GROUPS;
    return SHORTCUT_GROUPS.map((group) => ({
      ...group,
      items: group.items.filter(
        (item) =>
          item.action.toLowerCase().includes(q) ||
          item.keys.some((k) => {
            const formatted = formatShortcutKey(k, isMac).toLowerCase();
            return (
              k.toLowerCase().includes(q) ||
              formatted.includes(q) ||
              (isMac && k === 'Ctrl' && ('cmd'.includes(q) || 'command'.includes(q)))
            );
          }) ||
          group.title.toLowerCase().includes(q),
      ),
    })).filter((group) => group.items.length > 0);
  }, [shortcutQuery, isMac]);

  const defaultRecognizerLabel =
    runtimeInfo?.defaultRecognitionModelLabel ?? 'ArcFace R50 512-D';
  const activeRecognizerLabel =
    draft.customRecognizerModelPath &&
    draft.customRecognizerModelPath !== preferences.customRecognizerModelPath
      ? basenameOfPath(draft.customRecognizerModelPath)
      : draft.customRecognizerModelPath === null
        ? defaultRecognizerLabel
        : (runtimeInfo?.recognitionModelLabel ?? defaultRecognizerLabel);
  const activeRecognizerFile = basenameOfPath(
    draft.customRecognizerModelPath ??
      runtimeInfo?.defaultSfaceModelPath ??
      runtimeInfo?.sfaceModelPath,
  );
  const isCustomRecognizer = Boolean(draft.customRecognizerModelPath);

  const defaultDetectorLabel = runtimeInfo?.defaultDetectorModelLabel ?? 'YuNet 2026';
  const activeDetectorLabel =
    draft.customDetectorModelPath &&
    draft.customDetectorModelPath !== preferences.customDetectorModelPath
      ? basenameOfPath(draft.customDetectorModelPath)
      : draft.customDetectorModelPath === null
        ? defaultDetectorLabel
        : (runtimeInfo?.detectorModelLabel ?? defaultDetectorLabel);
  const activeDetectorFile = basenameOfPath(
    draft.customDetectorModelPath ??
      runtimeInfo?.defaultYunetModelPath ??
      runtimeInfo?.yunetModelPath,
  );
  const isCustomDetector = Boolean(draft.customDetectorModelPath);

  const defaultSegmenterLabel =
    runtimeInfo?.defaultSegmenterModelLabel ?? 'Built-in Portrait AI';
  const activeSegmenterLabel =
    draft.customSegmenterModelPath &&
    draft.customSegmenterModelPath !== preferences.customSegmenterModelPath
      ? basenameOfPath(draft.customSegmenterModelPath)
      : draft.customSegmenterModelPath === null
        ? defaultSegmenterLabel
        : (runtimeInfo?.segmenterModelLabel ?? defaultSegmenterLabel);
  const activeSegmenterFile = basenameOfPath(
    draft.customSegmenterModelPath ??
      runtimeInfo?.defaultSegmenterModelPath ??
      runtimeInfo?.segmenterModelPath,
  );
  const isCustomSegmenter = Boolean(draft.customSegmenterModelPath);
  const hasOnnxSegmenter = Boolean(
    draft.customSegmenterModelPath ||
      runtimeInfo?.defaultSegmenterModelPath ||
      runtimeInfo?.segmenterModelsReady,
  );
  const defaultFaceParserLabel =
    runtimeInfo?.defaultFaceParserModelLabel ?? 'YuNet 5-Point (Built-in)';
  const activeFaceParserLabel =
    draft.customFaceParserModelPath &&
    draft.customFaceParserModelPath !== preferences.customFaceParserModelPath
      ? basenameOfPath(draft.customFaceParserModelPath)
      : draft.customFaceParserModelPath === null
        ? defaultFaceParserLabel
        : (runtimeInfo?.faceParserModelLabel ?? defaultFaceParserLabel);
  const activeFaceParserFile = basenameOfPath(
    draft.customFaceParserModelPath ??
      runtimeInfo?.defaultFaceParserModelPath ??
      runtimeInfo?.faceParserModelPath,
  );
  const isCustomFaceParser = Boolean(draft.customFaceParserModelPath);
  const hasFaceParser = Boolean(
    draft.customFaceParserModelPath ||
      runtimeInfo?.defaultFaceParserModelPath ||
      runtimeInfo?.faceParserReady,
  );

  const defaultLamaLabel = runtimeInfo?.defaultLamaModelLabel ?? 'Harmonic Fill (Built-in)';
  const activeLamaLabel =
    draft.customLamaModelPath &&
    draft.customLamaModelPath !== preferences.customLamaModelPath
      ? basenameOfPath(draft.customLamaModelPath)
      : draft.customLamaModelPath === null
        ? defaultLamaLabel
        : (runtimeInfo?.lamaModelLabel ?? defaultLamaLabel);
  const activeLamaFile = basenameOfPath(
    draft.customLamaModelPath ??
      runtimeInfo?.defaultLamaModelPath ??
      runtimeInfo?.lamaModelPath,
  );
  const isCustomLama = Boolean(draft.customLamaModelPath);
  const hasLamaModel = Boolean(
    draft.customLamaModelPath ||
      runtimeInfo?.defaultLamaModelPath ||
      runtimeInfo?.lamaReady,
  );

  const customModelCount =
    Number(isCustomRecognizer) +
    Number(isCustomDetector) +
    Number(isCustomSegmenter) +
    Number(isCustomFaceParser) +
    Number(isCustomLama);

  const optionalDownloadPct =
    modelDownloadProgress && modelDownloadProgress.totalBytes > 0
      ? Math.min(
          100,
          Math.max(
            0,
            Math.round(
              (modelDownloadProgress.downloadedBytes / modelDownloadProgress.totalBytes) * 100,
            ),
          ),
        )
      : 0;

  const aiGpuActive = Boolean(runtimeInfo?.gpuAvailable && draft.enableGpuAcceleration);

  return (
    <>
      <AnimatePresence>
        {settingsOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Settings"
          className="fixed inset-0 z-[60] flex items-center justify-center p-4 sm:p-6"
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={FAST_FADE}
            onClick={requestClose}
            className="absolute inset-0 bg-canvas/75 backdrop-blur-sm"
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.965, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 8 }}
            transition={SOFT_SPRING}
            className="relative z-10 w-full max-w-[880px] h-[min(82vh,680px)] rounded-xl bg-canvas border border-hairline-strong shadow-[0_24px_64px_rgba(0,0,0,0.78)] flex min-w-0 overflow-hidden"
          >
            {/* Left Preferences Navigation */}
            <aside
              aria-label="Settings Navigation"
              className="w-[208px] shrink-0 bg-surface-lowest border-r border-hairline flex flex-col min-h-0 overflow-hidden"
            >
              <div className="px-4 pt-5 pb-3.5 flex items-center justify-between gap-2 shrink-0">
                <h1 className="font-display text-[15px] font-semibold tracking-tight text-ink-primary">
                  Settings
                </h1>
                <span className="tabular-nums-mono text-[9.5px] px-1.5 py-0.5 rounded bg-surface border border-hairline text-ink-telemetry">
                  {formatShortcutCombo('Ctrl+,', isMac)}
                </span>
              </div>

              <nav className="flex-1 overflow-y-auto px-2.5 flex flex-col gap-0.5">
                {SETTINGS_NAV.map((item) => {
                  const active = activeSection === item.id;
                  return (
                    <motion.button
                      key={item.id}
                      type="button"
                      whileTap={{ scale: 0.985 }}
                      onClick={() => setActiveSection(item.id)}
                      className={`relative w-full h-9 px-2.5 rounded-md text-left transition-colors cursor-pointer flex items-center gap-2.5 ${
                        active
                          ? 'text-ink-primary font-medium'
                          : 'text-ink-secondary hover:bg-surface/60 hover:text-ink-primary'
                      }`}
                    >
                      {active && (
                        <motion.span
                          layoutId="settings-nav-bg"
                          transition={TACTILE_SPRING}
                          className="absolute inset-0 rounded-md bg-surface-active border border-hairline-strong/60"
                        />
                      )}
                      {active && (
                        <motion.span
                          layoutId="settings-nav-accent"
                          transition={TACTILE_SPRING}
                          className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-amber-core"
                        />
                      )}
                      <span
                        className={`relative z-10 material-symbols-outlined text-[17px] shrink-0 transition-colors ${
                          active ? 'text-amber-soft' : 'text-ink-telemetry'
                        }`}
                      >
                        {item.icon}
                      </span>
                      <span className="relative z-10 font-display text-[12.5px] truncate">
                        {item.label}
                      </span>
                    </motion.button>
                  );
                })}
              </nav>

              <div className="px-4 py-3 border-t border-hairline/70 flex items-center justify-between gap-2 shrink-0">
                <span className="tabular-nums-mono text-[10px] flex items-center gap-1.5">
                  {hasUnsavedChanges ? (
                    <>
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-core shrink-0" />
                      <span className="text-amber-soft">{changedCount} unsaved</span>
                    </>
                  ) : (
                    <span className="text-ink-telemetry">v{appVersion}</span>
                  )}
                </span>
                <AnimatePresence>
                  {isDraftModified && (
                    <motion.button
                      type="button"
                      initial={{ opacity: 0, x: 4 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 4 }}
                      transition={FAST_FADE}
                      onClick={handleStageFactoryDefaults}
                      className="tabular-nums-mono text-[10px] text-ink-secondary hover:text-amber-soft transition-colors cursor-pointer"
                    >
                      Reset all
                    </motion.button>
                  )}
                </AnimatePresence>
              </div>
            </aside>

            {/* Right Preferences Content Pane */}
            <section
              aria-label="Settings Content"
              className="relative flex-1 min-w-0 flex flex-col overflow-hidden bg-canvas"
            >
              <button
                type="button"
                onClick={requestClose}
                title="Close Settings (Esc)"
                aria-label="Close Settings"
                className="absolute top-3.5 right-4 z-20 w-7 h-7 rounded-md bg-surface/85 hover:bg-surface-active border border-hairline text-ink-telemetry hover:text-ink-primary flex items-center justify-center transition-colors cursor-pointer backdrop-blur-sm"
              >
                <span className="material-symbols-outlined text-[16px]">close</span>
              </button>

              <div className="flex-1 overflow-y-auto">
                <div className="max-w-[640px] mx-auto px-7 py-6">
                  <AnimatePresence mode="wait">
                    {/* ── 1. General ── */}
                    {activeSection === 'general' && (
                      <motion.div
                        key="section-general"
                        variants={SECTION_CONTAINER_VARIANTS}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        className="flex flex-col gap-7"
                      >
                        <motion.header variants={SECTION_ITEM_VARIANTS} className="flex flex-col gap-1 pr-8">
                          <h2 className="font-display text-[19px] font-semibold tracking-tight text-ink-primary">
                            General
                          </h2>
                          <p className="text-[12px] text-ink-secondary">
                            Startup workflow, darkroom canvas appearance, and interface ergonomics.
                          </p>
                        </motion.header>

                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>Startup &amp; Workflow</SectionHeading>

                          <SettingRow
                            title="Re-open last project on launch"
                            description="Automatically reopen your most recent shoot when Kroma starts instead of showing Home."
                            control={
                              <DarkroomToggle
                                checked={draft.openLastProjectOnStartup}
                                onChange={(checked) =>
                                  patchDraft({ openLastProjectOnStartup: checked })
                                }
                                ariaLabel="Toggle Re-open last project on launch"
                              />
                            }
                          />

                          <SettingRow
                            title="Default workspace for organized projects"
                            description="Choose where to land when reopening a shoot that has already been grouped by face."
                            control={
                              <SegmentedControl
                                layoutId="settings-default-open-workspace"
                                options={OPEN_WORKSPACE_OPTIONS}
                                value={draft.defaultOpenWorkspace}
                                onChange={(view) => patchDraft({ defaultOpenWorkspace: view })}
                              />
                            }
                          />

                          <SettingRow
                            title="Default subject sort order"
                            description="Controls how subjects are ordered in the left column when opening a shoot."
                            last
                            control={
                              <SegmentedControl
                                layoutId="settings-default-subject-sort"
                                options={SUBJECT_SORT_OPTIONS}
                                value={draft.defaultSubjectSortMode}
                                onChange={(mode) => patchDraft({ defaultSubjectSortMode: mode })}
                              />
                            }
                          />
                        </motion.div>

                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>Interface &amp; Darkroom</SectionHeading>

                          <SettingRow
                            title="Darkroom canvas backdrop"
                            description="Background surround behind the active photo in Studio. Neutral Gray helps judge exposure and skin tones accurately."
                            control={
                              <SegmentedControl
                                layoutId="settings-canvas-backdrop"
                                options={CANVAS_BACKDROP_OPTIONS}
                                value={draft.canvasBackdropTone}
                                onChange={(tone) => patchDraft({ canvasBackdropTone: tone })}
                              />
                            }
                          />

                          <SettingRow
                            title="Interface scale"
                            description="Adjust overall UI sizing for compact laptop displays or high-DPI studio monitors."
                            control={
                              <SegmentedControl
                                layoutId="settings-interface-scale"
                                options={INTERFACE_SCALE_OPTIONS}
                                value={String(draft.interfaceScale) as '90' | '100' | '110'}
                                onChange={(scale) =>
                                  patchDraft({
                                    interfaceScale: Number(scale) as InterfaceScalePct,
                                  })
                                }
                              />
                            }
                          />

                          <SettingRow
                            title="Show metadata on thumbnails"
                            description="Display filenames, match badges, and capture info on Face Organizer photo cards."
                            control={
                              <DarkroomToggle
                                checked={draft.showThumbnailMetadata}
                                onChange={(checked) =>
                                  patchDraft({ showThumbnailMetadata: checked })
                                }
                                ariaLabel="Toggle Show metadata on thumbnails"
                              />
                            }
                          />

                          <SettingRow
                            title="Reduce Motion"
                            description="Disables spring transitions and panel slide animations across the app."
                            last
                            control={
                              <DarkroomToggle
                                checked={draft.reduceMotion}
                                onChange={(checked) => patchDraft({ reduceMotion: checked })}
                                ariaLabel="Toggle Reduce Motion"
                              />
                            }
                          />
                        </motion.div>
                      </motion.div>
                    )}

                    {/* ── 2. Performance ── */}
                    {activeSection === 'performance' && (
                      <motion.div
                        key="section-performance"
                        variants={SECTION_CONTAINER_VARIANTS}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        className="flex flex-col gap-7"
                      >
                        <motion.header variants={SECTION_ITEM_VARIANTS} className="flex flex-col gap-1 pr-8">
                          <h2 className="font-display text-[19px] font-semibold tracking-tight text-ink-primary">
                            Performance
                          </h2>
                          <p className="text-[12px] text-ink-secondary">
                            GPU acceleration and Studio live RAW preview resolution.
                          </p>
                        </motion.header>

                        {/* Live Hardware Status Banner */}
                        <motion.div
                          variants={SECTION_ITEM_VARIANTS}
                          className="p-3.5 rounded-xl bg-surface border border-hairline flex items-center justify-between gap-4"
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <div
                              className={`w-9 h-9 rounded-lg border flex items-center justify-center shrink-0 ${
                                aiGpuActive || detectedGpuName
                                  ? 'bg-emerald-focus/12 border-emerald-focus/35 text-emerald-bright'
                                  : 'bg-surface-elevated border-hairline text-ink-telemetry'
                              }`}
                            >
                              <span className="material-symbols-outlined text-[19px]">memory</span>
                            </div>
                            <div className="min-w-0">
                              <div className="font-display text-[12.5px] font-semibold text-ink-primary truncate">
                                {runtimeInfo?.developGpuAdapterName ??
                                  detectedGpuName ??
                                  (isDesktopRuntime ? 'System GPU' : 'WebGPU / WebGL2 Renderer')}
                              </div>
                              <div className="tabular-nums-mono text-[10.5px] text-ink-secondary truncate mt-0.5">
                                Studio &amp; Export:{' '}
                                {draft.enableGpuAcceleration
                                  ? `${typeof navigator !== 'undefined' && 'gpu' in navigator ? 'WebGPU' : 'WebGL2'} + ${
                                      isDesktopRuntime
                                        ? (runtimeInfo?.developGpuBackendLabel &&
                                          runtimeInfo.developGpuBackendLabel !== 'CPU'
                                            ? runtimeInfo.developGpuBackendLabel
                                            : isMac
                                              ? 'wgpu Metal'
                                              : 'wgpu DX12')
                                        : '32-bit Float'
                                    }`
                                  : 'WebGL2 + CPU'}{' '}
                                • AI Vision:{' '}
                                {isDesktopRuntime
                                  ? aiGpuActive
                                    ? (runtimeInfo?.backendLabel && runtimeInfo.backendLabel !== 'CPU'
                                        ? runtimeInfo.backendLabel
                                        : isMac
                                          ? 'CoreML GPU'
                                          : 'DirectML GPU')
                                    : draft.enableGpuAcceleration
                                      ? 'CPU Fallback'
                                      : 'CPU Only'
                                  : 'Browser Preview'}
                              </div>
                            </div>
                          </div>
                          <span
                            className={`tabular-nums-mono text-[9.5px] px-2 py-0.5 rounded-full border shrink-0 ${
                              aiGpuActive
                                ? 'bg-emerald-focus/12 border-emerald-focus/35 text-emerald-bright'
                                : 'bg-canvas border-hairline text-ink-telemetry'
                            }`}
                          >
                            {aiGpuActive ? 'GPU Ready' : 'CPU Mode'}
                          </span>
                        </motion.div>

                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>Graphics &amp; Preview</SectionHeading>

                          <SettingRow
                            title="AI Hardware Acceleration"
                            description="Accelerates Face Grouping, Whiteboard OCR, and Subject Masking on your GPU. Switch to CPU Only if you see driver or VRAM issues."
                            control={
                              <SegmentedControl
                                layoutId="settings-gpu-mode"
                                options={GPU_MODE_OPTIONS}
                                value={draft.enableGpuAcceleration ? 'auto' : 'cpu'}
                                onChange={(mode) =>
                                  patchDraft({ enableGpuAcceleration: mode === 'auto' })
                                }
                              />
                            }
                          />

                          <SettingRow
                            title="Studio RAW Preview Resolution"
                            description="Controls the working resolution when editing RAW photos in Studio. Final exports always render at full camera resolution."
                            control={
                              <SegmentedControl
                                layoutId="settings-preview-quality"
                                options={PREVIEW_QUALITY_OPTIONS}
                                value={`${draft.studioPreviewMaxDim}`}
                                onChange={(val) =>
                                  patchDraft({
                                    studioPreviewMaxDim: Number(val) as StudioPreviewMaxDim,
                                  })
                                }
                              />
                            }
                          />

                          <SettingRow
                            title="Grid & Filmstrip Previews"
                            description="Choose whether Face Organizer, filmstrip, and lightbox cards use fast camera previews or background 32-bit rendered thumbnails matching Studio."
                            last
                            control={
                              <SegmentedControl
                                layoutId="settings-thumbnail-preview-mode"
                                options={THUMBNAIL_PREVIEW_MODE_OPTIONS}
                                value={draft.thumbnailPreviewMode}
                                onChange={(val) =>
                                  patchDraft({
                                    thumbnailPreviewMode: val,
                                  })
                                }
                              />
                            }
                          />
                        </motion.div>
                      </motion.div>
                    )}

                    {/* ── 3. Models ── */}
                    {activeSection === 'matching' && (
                      <motion.div
                        key="section-matching"
                        variants={SECTION_CONTAINER_VARIANTS}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        className="flex flex-col gap-7"
                      >
                        <motion.header variants={SECTION_ITEM_VARIANTS} className="flex flex-col gap-1 pr-8">
                          <h2 className="font-display text-[19px] font-semibold tracking-tight text-ink-primary">
                            Models
                          </h2>
                          <p className="text-[12px] text-ink-secondary">
                            Configure face grouping sensitivity and local ONNX AI models.
                          </p>
                        </motion.header>

                        <AnimatePresence initial={false}>
                          {clusteringParamsDirty && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              transition={FAST_FADE}
                              className="overflow-hidden"
                            >
                              <div className="px-4 py-3 rounded-lg bg-amber-core/10 border border-amber-core/35 flex items-center justify-between gap-4">
                                <span className="text-[12px] text-ink-primary">
                                  Matching rules changed for the open project. Re-group from cached
                                  face data to apply.
                                </span>
                                <DarkroomButton
                                  variant="primary"
                                  size="xs"
                                  icon="refresh"
                                  disabled={isScanningFolder || isApplying}
                                  onClick={async () => {
                                    if (hasUnsavedChanges) {
                                      const ok = await handleApplyChanges(false);
                                      if (!ok) return;
                                    }
                                    closeSettings();
                                    void runFaceOrganizer();
                                  }}
                                  className="shrink-0"
                                >
                                  {hasUnsavedChanges ? 'Apply & Re-group' : 'Re-group'}
                                </DarkroomButton>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>

                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>Face Grouping</SectionHeading>

                          <SettingRow
                            title="Match Cutoff"
                            description="Higher values split look-alikes more strictly; lower values merge side angles."
                            control={
                              <div className="w-[220px]">
                                <DarkroomScrubber
                                  label="Cutoff"
                                  value={draft.sfaceThreshold}
                                  min={0.25}
                                  max={0.9}
                                  step={0.01}
                                  defaultValue={DEFAULT_STUDIO_PREFERENCES.sfaceThreshold}
                                  formatValue={(v) => v.toFixed(2)}
                                  onChange={(value) =>
                                    patchDraft({ sfaceThreshold: Number(value.toFixed(2)) })
                                  }
                                />
                              </div>
                            }
                          />

                          <SettingRow
                            title="Review Warning Threshold"
                            description="Photos matched below this confidence are flagged in Needs Review."
                            control={
                              <div className="w-[220px]">
                                <DarkroomScrubber
                                  label="Threshold"
                                  value={Math.round(draft.reviewWarningThreshold * 100)}
                                  min={50}
                                  max={95}
                                  step={1}
                                  defaultValue={Math.round(
                                    DEFAULT_STUDIO_PREFERENCES.reviewWarningThreshold * 100,
                                  )}
                                  formatValue={(v) => `${Math.round(v)}%`}
                                  onChange={(pct) =>
                                    patchDraft({
                                      reviewWarningThreshold: Number((pct / 100).toFixed(2)),
                                    })
                                  }
                                />
                              </div>
                            }
                          />

                          <SettingRow
                            title="Time Assist"
                            description="Uses shutter timestamps and sequence order to bridge burst frames of the same person."
                            control={
                              <DarkroomToggle
                                checked={draft.useTemporalAssist}
                                onChange={(checked) => patchDraft({ useTemporalAssist: checked })}
                                ariaLabel="Toggle Time Assist"
                              />
                            }
                          />

                          <SettingRow
                            title="Face Landmarks Overlay"
                            description="Shows face detection boxes and alignment markers in Face Organizer."
                            last
                            control={
                              <DarkroomToggle
                                checked={draft.showLandmarksOverlay}
                                onChange={(checked) =>
                                  patchDraft({ showLandmarksOverlay: checked })
                                }
                                ariaLabel="Toggle Face Landmarks Overlay"
                              />
                            }
                          />
                        </motion.div>

                        {/* Optional Phase 3 AI Retouch Packs (On-Demand Opt-In) */}
                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>Optional AI Retouch Packs</SectionHeading>

                          <SettingRow
                            title="BiSeNet Neural Face Parsing"
                            description="Pixel-exact 19-class skin, eye, teeth & lip masks that protect glasses and hairlines during Portrait Retouch and upgrade AI Face & Skin masks (~51 MB)."
                            control={
                              downloadingOptionalModelId === 'bisenet-face-parsing' ? (
                                <div className="flex items-center gap-2 min-w-[180px]">
                                  <div className="flex-1 flex flex-col gap-1">
                                    <div className="flex items-center justify-between tabular-nums-mono text-[10px] text-ink-secondary">
                                      <span>Downloading…</span>
                                      <span>{optionalDownloadPct}%</span>
                                    </div>
                                    <div className="h-1.5 w-full rounded-full bg-surface overflow-hidden">
                                      <div
                                        style={{ width: `${optionalDownloadPct}%` }}
                                        className="h-full bg-emerald-400 transition-[width] duration-150"
                                      />
                                    </div>
                                  </div>
                                  <DarkroomButton
                                    variant="ghost"
                                    size="xs"
                                    icon="close"
                                    onClick={() => void cancelOptionalModelDownload()}
                                    title="Cancel download"
                                    aria-label="Cancel BiSeNet download"
                                  />
                                </div>
                              ) : hasFaceParser ? (
                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-focus/12 border border-emerald-focus/35 text-emerald-bright tabular-nums-mono text-[10.5px]">
                                  <span className="material-symbols-outlined text-[14px]">
                                    check_circle
                                  </span>
                                  Installed
                                </span>
                              ) : (
                                <DarkroomButton
                                  variant="secondary"
                                  size="xs"
                                  icon="download"
                                  disabled={!isDesktopRuntime || downloadingOptionalModelId !== null}
                                  onClick={() => void downloadOptionalModel('bisenet-face-parsing')}
                                >
                                  Install (51 MB)
                                </DarkroomButton>
                              )
                            }
                          />

                          <SettingRow
                            title="LaMa Generative Object Removal"
                            description="Neural inpainting model for erasing large distractions, microphone stands, and photobombers. Falls back to built-in Harmonic Fill when not installed (~198 MB)."
                            last
                            control={
                              downloadingOptionalModelId === 'lama-inpainting' ? (
                                <div className="flex items-center gap-2 min-w-[180px]">
                                  <div className="flex-1 flex flex-col gap-1">
                                    <div className="flex items-center justify-between tabular-nums-mono text-[10px] text-ink-secondary">
                                      <span>Downloading…</span>
                                      <span>{optionalDownloadPct}%</span>
                                    </div>
                                    <div className="h-1.5 w-full rounded-full bg-surface overflow-hidden">
                                      <div
                                        style={{ width: `${optionalDownloadPct}%` }}
                                        className="h-full bg-amber-core transition-[width] duration-150"
                                      />
                                    </div>
                                  </div>
                                  <DarkroomButton
                                    variant="ghost"
                                    size="xs"
                                    icon="close"
                                    onClick={() => void cancelOptionalModelDownload()}
                                    title="Cancel download"
                                    aria-label="Cancel LaMa download"
                                  />
                                </div>
                              ) : hasLamaModel ? (
                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-focus/12 border border-emerald-focus/35 text-emerald-bright tabular-nums-mono text-[10.5px]">
                                  <span className="material-symbols-outlined text-[14px]">
                                    check_circle
                                  </span>
                                  Installed
                                </span>
                              ) : (
                                <DarkroomButton
                                  variant="secondary"
                                  size="xs"
                                  icon="download"
                                  disabled={!isDesktopRuntime || downloadingOptionalModelId !== null}
                                  onClick={() => void downloadOptionalModel('lama-inpainting')}
                                >
                                  Install (198 MB)
                                </DarkroomButton>
                              )
                            }
                          />
                        </motion.div>

                        {/* Collapsed Advanced: Custom ONNX Models */}
                        <motion.div
                          variants={SECTION_ITEM_VARIANTS}
                          className="rounded-xl bg-surface border border-hairline overflow-hidden"
                        >
                          <button
                            type="button"
                            onClick={() => setShowAdvancedModels((prev) => !prev)}
                            aria-expanded={showAdvancedModels}
                            className="w-full px-4 py-3 flex items-center justify-between gap-2 text-left hover:bg-surface-elevated/60 transition-colors cursor-pointer"
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry font-semibold">
                                Advanced: Custom AI Models
                              </span>
                              {customModelCount > 0 && (
                                <span className="tabular-nums-mono text-[9.5px] px-1.5 py-0.5 rounded bg-amber-core/15 border border-amber-core/35 text-amber-soft">
                                  {customModelCount} custom
                                </span>
                              )}
                            </div>
                            <span
                              className={`material-symbols-outlined text-[16px] text-ink-telemetry transition-transform ${
                                showAdvancedModels ? 'rotate-180' : ''
                              }`}
                            >
                              expand_more
                            </span>
                          </button>

                          <AnimatePresence initial={false}>
                            {showAdvancedModels && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={FAST_FADE}
                                className="overflow-hidden"
                              >
                                <div className="px-4 pb-4 pt-1 border-t border-hairline flex flex-col">
                                  {/* Face Recognition */}
                                  <div className="py-3 border-b border-hairline/70 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                    <div className="min-w-0 flex-1">
                                      <div className="flex items-center gap-2">
                                        <span className="font-display text-[12.5px] font-medium text-ink-primary">
                                          Face Recognition
                                        </span>
                                        <span
                                          className={`tabular-nums-mono text-[9px] px-1.5 py-0.5 rounded border ${
                                            isCustomRecognizer
                                              ? 'bg-amber-core/12 border-amber-core/40 text-amber-soft'
                                              : 'bg-canvas border-hairline text-ink-telemetry'
                                          }`}
                                        >
                                          {isCustomRecognizer ? 'Custom' : 'Default'}
                                        </span>
                                      </div>
                                      <div className="mt-0.5 tabular-nums-mono text-[10.5px] text-ink-telemetry truncate">
                                        {activeRecognizerLabel} • {activeRecognizerFile}
                                      </div>
                                    </div>
                                    <div className="shrink-0 flex items-center gap-2">
                                      {isCustomRecognizer && (
                                        <button
                                          type="button"
                                          disabled={isApplying}
                                          onClick={() => handleResetModelToDefault('recognizer')}
                                          className="text-[11px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                                        >
                                          Use Default
                                        </button>
                                      )}
                                      <DarkroomButton
                                        variant="secondary"
                                        size="xs"
                                        icon="upload_file"
                                        disabled={isApplying}
                                        onClick={() => void handlePickCustomModel('recognizer')}
                                      >
                                        Choose .onnx…
                                      </DarkroomButton>
                                    </div>
                                  </div>

                                  {/* Face Detector */}
                                  <div className="py-3 border-b border-hairline/70 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                    <div className="min-w-0 flex-1">
                                      <div className="flex items-center gap-2">
                                        <span className="font-display text-[12.5px] font-medium text-ink-primary">
                                          Face Detector
                                        </span>
                                        <span
                                          className={`tabular-nums-mono text-[9px] px-1.5 py-0.5 rounded border ${
                                            isCustomDetector
                                              ? 'bg-amber-core/12 border-amber-core/40 text-amber-soft'
                                              : 'bg-canvas border-hairline text-ink-telemetry'
                                          }`}
                                        >
                                          {isCustomDetector ? 'Custom' : 'Default'}
                                        </span>
                                      </div>
                                      <div className="mt-0.5 tabular-nums-mono text-[10.5px] text-ink-telemetry truncate">
                                        {activeDetectorLabel} • {activeDetectorFile}
                                      </div>
                                    </div>
                                    <div className="shrink-0 flex items-center gap-2">
                                      {isCustomDetector && (
                                        <button
                                          type="button"
                                          disabled={isApplying}
                                          onClick={() => handleResetModelToDefault('detector')}
                                          className="text-[11px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                                        >
                                          Use Default
                                        </button>
                                      )}
                                      <DarkroomButton
                                        variant="secondary"
                                        size="xs"
                                        icon="upload_file"
                                        disabled={isApplying}
                                        onClick={() => void handlePickCustomModel('detector')}
                                      >
                                        Choose .onnx…
                                      </DarkroomButton>
                                    </div>
                                  </div>

                                  {/* Subject Masking */}
                                  <div className="py-3 border-b border-hairline/70 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                    <div className="min-w-0 flex-1">
                                      <div className="flex items-center gap-2">
                                        <span className="font-display text-[12.5px] font-medium text-ink-primary">
                                          Subject Masking
                                        </span>
                                        <span
                                          className={`tabular-nums-mono text-[9px] px-1.5 py-0.5 rounded border ${
                                            isCustomSegmenter
                                              ? 'bg-amber-core/12 border-amber-core/40 text-amber-soft'
                                              : 'bg-canvas border-hairline text-ink-telemetry'
                                          }`}
                                        >
                                          {isCustomSegmenter
                                            ? 'Custom'
                                            : hasOnnxSegmenter
                                              ? 'Default'
                                              : 'Built-in'}
                                        </span>
                                      </div>
                                      <div className="mt-0.5 tabular-nums-mono text-[10.5px] text-ink-telemetry truncate">
                                        {activeSegmenterLabel}
                                        {hasOnnxSegmenter ? ` • ${activeSegmenterFile}` : ''}
                                      </div>
                                    </div>
                                    <div className="shrink-0 flex items-center gap-2">
                                      {isCustomSegmenter && (
                                        <button
                                          type="button"
                                          disabled={isApplying}
                                          onClick={() => handleResetModelToDefault('segmenter')}
                                          className="text-[11px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                                        >
                                          Use Default
                                        </button>
                                      )}
                                      <DarkroomButton
                                        variant="secondary"
                                        size="xs"
                                        icon="upload_file"
                                        disabled={isApplying}
                                        onClick={() => void handlePickCustomModel('segmenter')}
                                      >
                                        Choose .onnx…
                                      </DarkroomButton>
                                    </div>
                                  </div>

                                  {/* Face Parsing (BiSeNet) */}
                                  <div className="py-3 border-b border-hairline/70 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                    <div className="min-w-0 flex-1">
                                      <div className="flex items-center gap-2">
                                        <span className="font-display text-[12.5px] font-medium text-ink-primary">
                                          Face Parsing (BiSeNet)
                                        </span>
                                        <span
                                          className={`tabular-nums-mono text-[9px] px-1.5 py-0.5 rounded border ${
                                            isCustomFaceParser
                                              ? 'bg-amber-core/12 border-amber-core/40 text-amber-soft'
                                              : 'bg-canvas border-hairline text-ink-telemetry'
                                          }`}
                                        >
                                          {isCustomFaceParser
                                            ? 'Custom'
                                            : hasFaceParser
                                              ? 'Installed'
                                              : 'Optional'}
                                        </span>
                                      </div>
                                      <div className="mt-0.5 tabular-nums-mono text-[10.5px] text-ink-telemetry truncate">
                                        {activeFaceParserLabel}
                                        {hasFaceParser ? ` • ${activeFaceParserFile}` : ''}
                                      </div>
                                    </div>
                                    <div className="shrink-0 flex items-center gap-2">
                                      {isCustomFaceParser && (
                                        <button
                                          type="button"
                                          disabled={isApplying}
                                          onClick={() => handleResetModelToDefault('face_parser')}
                                          className="text-[11px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                                        >
                                          Use Default
                                        </button>
                                      )}
                                      <DarkroomButton
                                        variant="secondary"
                                        size="xs"
                                        icon="upload_file"
                                        disabled={isApplying}
                                        onClick={() => void handlePickCustomModel('face_parser')}
                                      >
                                        Choose .onnx…
                                      </DarkroomButton>
                                    </div>
                                  </div>

                                  {/* Generative Object Removal (LaMa) */}
                                  <div className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                    <div className="min-w-0 flex-1">
                                      <div className="flex items-center gap-2">
                                        <span className="font-display text-[12.5px] font-medium text-ink-primary">
                                          Object Removal (LaMa)
                                        </span>
                                        <span
                                          className={`tabular-nums-mono text-[9px] px-1.5 py-0.5 rounded border ${
                                            isCustomLama
                                              ? 'bg-amber-core/12 border-amber-core/40 text-amber-soft'
                                              : 'bg-canvas border-hairline text-ink-telemetry'
                                          }`}
                                        >
                                          {isCustomLama
                                            ? 'Custom'
                                            : hasLamaModel
                                              ? 'Installed'
                                              : 'Optional'}
                                        </span>
                                      </div>
                                      <div className="mt-0.5 tabular-nums-mono text-[10.5px] text-ink-telemetry truncate">
                                        {activeLamaLabel}
                                        {hasLamaModel ? ` • ${activeLamaFile}` : ''}
                                      </div>
                                    </div>
                                    <div className="shrink-0 flex items-center gap-2">
                                      {isCustomLama && (
                                        <button
                                          type="button"
                                          disabled={isApplying}
                                          onClick={() => handleResetModelToDefault('inpainter')}
                                          className="text-[11px] text-ink-telemetry hover:text-amber-soft cursor-pointer"
                                        >
                                          Use Default
                                        </button>
                                      )}
                                      <DarkroomButton
                                        variant="secondary"
                                        size="xs"
                                        icon="upload_file"
                                        disabled={isApplying}
                                        onClick={() => void handlePickCustomModel('inpainter')}
                                      >
                                        Choose .onnx…
                                      </DarkroomButton>
                                    </div>
                                  </div>

                                  {/* Model Compatibility Verification Drawer */}
                                  <AnimatePresence initial={false}>
                                    {pendingModelCheck && (
                                      <motion.div
                                        key={`${pendingModelCheck.role}-${pendingModelCheck.filePath}`}
                                        initial={{ opacity: 0, height: 0, y: -4 }}
                                        animate={{ opacity: 1, height: 'auto', y: 0 }}
                                        exit={{ opacity: 0, height: 0, y: -4 }}
                                        transition={SOFT_SPRING}
                                        className="overflow-hidden"
                                      >
                                        <div
                                          className={`mt-2 p-3.5 rounded-lg border flex flex-col gap-2 ${
                                            pendingModelCheck.status === 'inspecting'
                                              ? 'bg-canvas border-hairline'
                                              : pendingModelCheck.report?.compatibility === 'compatible'
                                                ? 'bg-emerald-focus/8 border-emerald-focus/35'
                                                : pendingModelCheck.report?.compatibility === 'warning'
                                                  ? 'bg-amber-core/10 border-amber-core/40'
                                                  : 'bg-crimson-clip/10 border-crimson-clip/40'
                                          }`}
                                        >
                                          <div className="flex items-start justify-between gap-2">
                                            <div className="flex items-start gap-2 min-w-0">
                                              <span
                                                className={`material-symbols-outlined text-[17px] shrink-0 mt-0.5 ${
                                                  pendingModelCheck.status === 'inspecting'
                                                    ? 'text-amber-soft animate-spin'
                                                    : pendingModelCheck.report?.compatibility ===
                                                        'compatible'
                                                      ? 'text-emerald-bright'
                                                      : pendingModelCheck.report?.compatibility ===
                                                          'warning'
                                                        ? 'text-amber-soft'
                                                        : 'text-crimson-clip'
                                                }`}
                                              >
                                                {pendingModelCheck.status === 'inspecting'
                                                  ? 'progress_activity'
                                                  : pendingModelCheck.report?.compatibility ===
                                                      'compatible'
                                                    ? 'verified'
                                                    : pendingModelCheck.report?.compatibility ===
                                                        'warning'
                                                      ? 'warning'
                                                      : 'block'}
                                              </span>
                                              <div className="min-w-0">
                                                <div className="font-display text-[12px] font-semibold text-ink-primary">
                                                  {pendingModelCheck.status === 'inspecting'
                                                    ? `Inspecting ${pendingModelCheck.fileName}…`
                                                    : pendingModelCheck.report?.compatibility ===
                                                        'compatible'
                                                      ? `Verified — ${pendingModelCheck.report.detectedLabel} (click Apply Changes)`
                                                      : pendingModelCheck.report?.compatibility ===
                                                          'warning'
                                                        ? `Unrecognized model — ${pendingModelCheck.fileName}`
                                                        : `Incompatible model — ${pendingModelCheck.report?.summary ?? 'Blocked'}`}
                                                </div>
                                                {pendingModelCheck.report && (
                                                  <div className="mt-0.5 tabular-nums-mono text-[10px] text-ink-secondary flex flex-wrap items-center gap-x-2">
                                                    <span>In: {pendingModelCheck.report.inputShape}</span>
                                                    <span>→</span>
                                                    <span>Out: {pendingModelCheck.report.outputShape}</span>
                                                    {pendingModelCheck.report.fileSizeBytes > 0 && (
                                                      <span>
                                                        •{' '}
                                                        {formatByteSize(
                                                          pendingModelCheck.report.fileSizeBytes,
                                                          true,
                                                        )}
                                                      </span>
                                                    )}
                                                  </div>
                                                )}
                                              </div>
                                            </div>
                                            <button
                                              type="button"
                                              onClick={() => setPendingModelCheck(null)}
                                              aria-label="Dismiss model check"
                                              className="text-ink-telemetry hover:text-ink-primary cursor-pointer shrink-0"
                                            >
                                              <span className="material-symbols-outlined text-[15px]">
                                                close
                                              </span>
                                            </button>
                                          </div>

                                          {pendingModelCheck.report &&
                                            pendingModelCheck.report.warnings.length > 0 && (
                                              <ul className="pl-6 flex flex-col gap-1 text-[11px] text-ink-secondary list-disc">
                                                {pendingModelCheck.report.warnings.map((w, i) => (
                                                  <li key={i}>{w}</li>
                                                ))}
                                              </ul>
                                            )}

                                          {pendingModelCheck.report?.compatibility === 'warning' && (
                                            <div className="pl-6 flex items-center gap-2">
                                              <DarkroomButton
                                                variant="primary"
                                                size="xs"
                                                disabled={isApplying}
                                                onClick={handleConfirmWarningModel}
                                              >
                                                Use Anyway
                                              </DarkroomButton>
                                              <DarkroomButton
                                                variant="ghost"
                                                size="xs"
                                                onClick={() => setPendingModelCheck(null)}
                                              >
                                                Cancel
                                              </DarkroomButton>
                                            </div>
                                          )}
                                        </div>
                                      </motion.div>
                                    )}
                                  </AnimatePresence>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      </motion.div>
                    )}

                    {/* ── 4. Storage & Cache ── */}
                    {activeSection === 'storage' && (
                      <motion.div
                        key="section-storage"
                        variants={SECTION_CONTAINER_VARIANTS}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        className="flex flex-col gap-7"
                      >
                        <motion.header variants={SECTION_ITEM_VARIANTS} className="flex flex-col gap-1 pr-8">
                          <h2 className="font-display text-[19px] font-semibold tracking-tight text-ink-primary">
                            Storage &amp; Cache
                          </h2>
                          <p className="text-[12px] text-ink-secondary">
                            Manage temporary previews, face scan caches, and saved preferences. Your
                            original photos and edits are never deleted.
                          </p>
                        </motion.header>

                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>Temporary Caches</SectionHeading>

                          <SettingRow
                            title="Cache Location"
                            description={
                              draft.customCacheDir
                                ? draft.customCacheDir
                                : (cacheSummary?.cacheDirPath ?? 'System temporary folder (Default)')
                            }
                            control={
                              <div className="flex items-center gap-2">
                                {draft.customCacheDir && (
                                  <button
                                    type="button"
                                    onClick={() => patchDraft({ customCacheDir: null })}
                                    className="text-[11px] text-ink-telemetry hover:text-amber-soft transition-colors cursor-pointer"
                                  >
                                    Reset
                                  </button>
                                )}
                                <DarkroomButton
                                  variant="secondary"
                                  size="xs"
                                  icon="folder_open"
                                  onClick={async () => {
                                    const chosen = await pickCacheDirectory();
                                    if (chosen) patchDraft({ customCacheDir: chosen });
                                  }}
                                >
                                  Choose…
                                </DarkroomButton>
                              </div>
                            }
                          />

                          <SettingRow
                            title="RAW Preview Cache"
                            description={
                              isLoadingCache
                                ? 'Calculating disk usage…'
                                : cacheSummary
                                  ? `${formatByteSize(cacheSummary.previewCacheBytes, true)} across ${cacheSummary.previewFileCount.toLocaleString()} cached ${cacheSummary.previewFileCount === 1 ? 'preview' : 'previews'}. Rebuilds automatically when needed.`
                                  : 'Cached thumbnails for fast RAW browsing.'
                            }
                            control={
                              <DarkroomButton
                                variant="secondary"
                                size="xs"
                                icon="delete_sweep"
                                disabled={
                                  clearingCacheTarget !== null ||
                                  (cacheSummary !== null && cacheSummary.previewFileCount === 0)
                                }
                                onClick={() => void handleClearCache('previews')}
                              >
                                {clearingCacheTarget === 'previews' ? 'Clearing…' : 'Clear Previews'}
                              </DarkroomButton>
                            }
                          />

                          <SettingRow
                            title="Face Scan Cache"
                            description={
                              isLoadingCache
                                ? 'Calculating disk usage…'
                                : cacheSummary
                                  ? `${formatByteSize(cacheSummary.faceCacheBytes, true)} across ${cacheSummary.faceCacheFileCount.toLocaleString()} project ${cacheSummary.faceCacheFileCount === 1 ? 'cache' : 'caches'}. Clearing forces a fresh face scan without touching edits.`
                                  : 'Saved face detection data for instant re-grouping.'
                            }
                            last
                            control={
                              <DarkroomButton
                                variant="secondary"
                                size="xs"
                                icon="face_retouching_off"
                                disabled={
                                  clearingCacheTarget !== null ||
                                  (cacheSummary !== null && cacheSummary.faceCacheFileCount === 0)
                                }
                                onClick={() => void handleClearCache('face_cache')}
                              >
                                {clearingCacheTarget === 'face_cache'
                                  ? 'Clearing…'
                                  : 'Clear Face Cache'}
                              </DarkroomButton>
                            }
                          />
                        </motion.div>

                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>History &amp; Maintenance</SectionHeading>

                          <SettingRow
                            title="Recent Projects List"
                            description={
                              recentProjects.length > 0
                                ? `${recentProjects.length} recent ${recentProjects.length === 1 ? 'project' : 'projects'} shown on the Home screen. Clearing only removes shortcuts.`
                                : 'No recent projects saved.'
                            }
                            control={
                              <DarkroomButton
                                variant="secondary"
                                size="xs"
                                icon="history"
                                disabled={recentProjects.length === 0}
                                onClick={clearRecentProjects}
                              >
                                Clear List
                              </DarkroomButton>
                            }
                          />

                          <SettingRow
                            title="Reset All Preferences"
                            description="Restore matching cutoffs, GPU settings, and startup preferences to factory defaults."
                            last
                            control={
                              <DarkroomButton
                                variant="secondary"
                                size="xs"
                                icon="restart_alt"
                                disabled={!isDraftModified}
                                onClick={handleStageFactoryDefaults}
                              >
                                Reset Defaults
                              </DarkroomButton>
                            }
                          />
                        </motion.div>
                      </motion.div>
                    )}

                    {/* ── 5. Keyboard Shortcuts ── */}
                    {activeSection === 'shortcuts' && (
                      <motion.div
                        key="section-shortcuts"
                        variants={SECTION_CONTAINER_VARIANTS}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        className="flex flex-col gap-5"
                      >
                        <motion.header variants={SECTION_ITEM_VARIANTS} className="flex flex-col gap-3 pr-8">
                          <div className="flex flex-col gap-1">
                            <h2 className="font-display text-[19px] font-semibold tracking-tight text-ink-primary">
                              Keyboard Shortcuts
                            </h2>
                            <p className="text-[12px] text-ink-secondary">
                              Quick reference for editing, culling, and reviewing shoots faster.
                            </p>
                          </div>

                          <div className="flex items-center gap-2 h-8 px-2.5 rounded-lg bg-surface border border-hairline focus-within:border-amber-core">
                            <span className="material-symbols-outlined text-[15px] text-ink-telemetry shrink-0">
                              search
                            </span>
                            <input
                              type="text"
                              value={shortcutQuery}
                              onChange={(e) => setShortcutQuery(e.target.value)}
                              placeholder="Filter shortcuts…"
                              className="bg-transparent text-[12px] text-ink-primary placeholder:text-ink-telemetry flex-1 min-w-0 outline-none"
                            />
                            {shortcutQuery && (
                              <button
                                type="button"
                                onClick={() => setShortcutQuery('')}
                                className="text-ink-telemetry hover:text-ink-primary cursor-pointer flex items-center"
                              >
                                <span className="material-symbols-outlined text-[14px]">close</span>
                              </button>
                            )}
                          </div>
                        </motion.header>

                        {filteredShortcutGroups.length === 0 ? (
                          <div className="py-10 text-center text-[12px] text-ink-telemetry">
                            No shortcuts matching &ldquo;{shortcutQuery}&rdquo;
                          </div>
                        ) : (
                          filteredShortcutGroups.map((group) => (
                            <motion.div
                              key={group.title}
                              variants={SECTION_ITEM_VARIANTS}
                              className="flex flex-col"
                            >
                              <SectionHeading>{group.title}</SectionHeading>
                              <div className="rounded-lg bg-surface border border-hairline divide-y divide-hairline/70">
                                {group.items.map((item) => (
                                  <div
                                    key={`${group.title}-${item.action}`}
                                    className="px-3.5 py-2.5 flex items-center justify-between gap-4"
                                  >
                                    <span className="text-[12px] text-ink-primary">
                                      {item.action}
                                    </span>
                                    <div className="flex items-center gap-1 shrink-0">
                                      {item.keys.map((keyLabel) => (
                                        <kbd
                                          key={keyLabel}
                                          className="min-w-[22px] px-1.5 py-0.5 rounded bg-canvas border border-hairline-strong text-center tabular-nums-mono text-[10.5px] text-ink-secondary shadow-[0_1px_0_rgba(0,0,0,0.45)]"
                                        >
                                          {formatShortcutKey(keyLabel, isMac)}
                                        </kbd>
                                      ))}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </motion.div>
                          ))
                        )}
                      </motion.div>
                    )}

                    {/* ── 6. About & Updates ── */}
                    {activeSection === 'about' && (
                      <motion.div
                        key="section-about"
                        variants={SECTION_CONTAINER_VARIANTS}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        className="flex flex-col gap-7"
                      >
                        {/* Brand Identity Header Card */}
                        <motion.div
                          variants={SECTION_ITEM_VARIANTS}
                          className="p-4 rounded-xl bg-surface border border-hairline flex items-center justify-between gap-4 mr-8"
                        >
                          <div className="flex items-center gap-3.5 min-w-0">
                            <img
                              src={kromaLogo}
                              alt="Kroma"
                              draggable={false}
                              className="w-11 h-11 rounded-xl object-cover shrink-0 shadow-[0_4px_14px_rgba(0,0,0,0.45)]"
                            />
                            <div className="min-w-0">
                              <div className="flex items-center gap-2">
                                <h2 className="font-display text-[17px] font-bold tracking-wide text-ink-primary">
                                  KROMA
                                </h2>
                                <span className="tabular-nums-mono text-[10.5px] px-2 py-0.5 rounded-full bg-amber-core/15 border border-amber-core/35 text-amber-soft font-semibold">
                                  v{appVersion}
                                </span>
                              </div>
                              <p className="text-[11.5px] text-ink-secondary mt-0.5 truncate">
                                Local-first AI photography studio for organizing, grading, and
                                delivering shoots.
                              </p>
                            </div>
                          </div>
                        </motion.div>

                        {/* Updates Section */}
                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>Updates</SectionHeading>

                          <SettingRow
                            title="Software Updates"
                            description={`Checks github.com/${KROMA_GITHUB_REPO} for newer desktop releases.`}
                            last={!updateCheckResult}
                            control={
                              <DarkroomButton
                                variant="secondary"
                                size="xs"
                                icon={isCheckingUpdates ? 'progress_activity' : 'system_update'}
                                iconClassName={isCheckingUpdates ? 'animate-spin' : ''}
                                disabled={isCheckingUpdates}
                                onClick={() => void handleCheckForUpdates()}
                              >
                                {isCheckingUpdates ? 'Checking…' : 'Check for Updates'}
                              </DarkroomButton>
                            }
                          />

                          <AnimatePresence initial={false}>
                            {updateCheckResult && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={FAST_FADE}
                                className="overflow-hidden"
                              >
                                <div
                                  className={`mt-2 px-3.5 py-2.5 rounded-lg border flex items-center justify-between gap-3 text-[11.5px] ${
                                    updateCheckResult.status === 'update_available'
                                      ? 'bg-amber-core/12 border-amber-core/45 text-amber-soft'
                                      : updateCheckResult.status === 'up_to_date'
                                        ? 'bg-emerald-focus/10 border-emerald-focus/35 text-emerald-bright'
                                        : 'bg-surface border-hairline text-ink-secondary'
                                  }`}
                                >
                                  <span className="leading-snug">{updateCheckResult.message}</span>
                                  {updateCheckResult.status === 'update_available' && (
                                    <button
                                      type="button"
                                      onClick={() => setShowReleaseNotesModal(true)}
                                      className="shrink-0 px-2.5 py-1 rounded-md bg-amber-core text-amber-ink font-display text-[11px] font-bold hover:brightness-110 transition-all cursor-pointer"
                                    >
                                      View Release Notes
                                    </button>
                                  )}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>

                        {/* Compact Hardware & Runtime Summary */}
                        <motion.div variants={SECTION_ITEM_VARIANTS} className="flex flex-col">
                          <SectionHeading>Environment</SectionHeading>
                          <dl className="divide-y divide-hairline/70 text-[12px]">
                            <div className="py-2.5 flex items-center justify-between gap-4">
                              <dt className="text-ink-secondary">Platform</dt>
                              <dd className="tabular-nums-mono text-[11px] text-ink-primary">
                                {isDesktopRuntime ? 'Kroma Desktop' : 'Browser Preview'}
                              </dd>
                            </div>
                            <div className="py-2.5 flex items-center justify-between gap-4">
                              <dt className="text-ink-secondary">Graphics / AI Compute</dt>
                              <dd className="tabular-nums-mono text-[11px] text-ink-primary truncate max-w-[340px]">
                                {detectedGpuName
                                  ? `${detectedGpuName} • ${runtimeInfo?.backendLabel ?? 'CPU'}`
                                  : (runtimeInfo?.backendLabel ?? 'CPU')}
                              </dd>
                            </div>
                            <div className="py-2.5 flex items-center justify-between gap-4">
                              <dt className="text-ink-secondary">Supported Camera RAWs</dt>
                              <dd className="tabular-nums-mono text-[11px] text-ink-primary">
                                {runtimeInfo?.supportedRawCount ?? 29} formats
                              </dd>
                            </div>
                            <div className="py-2.5 flex items-center justify-between gap-4">
                              <dt className="text-ink-secondary">Repository</dt>
                              <dd className="tabular-nums-mono text-[11px]">
                                <a
                                  href={KROMA_GITHUB_URL}
                                  target="_blank"
                                  rel="noreferrer"
                                  onClick={(e) => {
                                    e.preventDefault();
                                    void openExternalUrl(KROMA_GITHUB_URL);
                                  }}
                                  className="text-amber-soft hover:underline cursor-pointer"
                                >
                                  github.com/{KROMA_GITHUB_REPO}
                                </a>
                              </dd>
                            </div>
                          </dl>
                        </motion.div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>

              {/* Interactive Unsaved Changes / "Apply changes?" Bar */}
              <AnimatePresence initial={false}>
                {hasUnsavedChanges && (
                  <motion.div
                    key={confirmClosePrompt ? 'confirm-close' : 'dirty-bar'}
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 14 }}
                    transition={SOFT_SPRING}
                    className={`shrink-0 px-6 py-3 border-t flex items-center justify-between gap-3 ${
                      confirmClosePrompt
                        ? 'bg-amber-core/12 border-amber-core/50'
                        : 'bg-surface border-hairline-strong'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="w-2 h-2 rounded-full bg-amber-core shrink-0" />
                      <span className="font-display text-[12px] font-medium text-ink-primary truncate">
                        {confirmClosePrompt
                          ? 'Apply changes before closing?'
                          : `Apply ${changedCount === 1 ? '1 change' : `${changedCount} changes`}?`}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      {confirmClosePrompt ? (
                        <>
                          <DarkroomButton
                            variant="ghost"
                            size="xs"
                            disabled={isApplying}
                            onClick={handleDiscardAndClose}
                          >
                            Discard
                          </DarkroomButton>
                          <DarkroomButton
                            variant="secondary"
                            size="xs"
                            disabled={isApplying}
                            onClick={() => setConfirmClosePrompt(false)}
                          >
                            Keep Editing
                          </DarkroomButton>
                          <DarkroomButton
                            variant="primary"
                            size="xs"
                            icon="check"
                            disabled={isApplying}
                            onClick={() => void handleApplyChanges(true)}
                          >
                            {isApplying ? 'Applying…' : 'Apply & Close'}
                          </DarkroomButton>
                        </>
                      ) : (
                        <>
                          <DarkroomButton
                            variant="ghost"
                            size="xs"
                            disabled={isApplying}
                            onClick={handleDiscardChanges}
                          >
                            Discard
                          </DarkroomButton>
                          <DarkroomButton
                            variant="primary"
                            size="xs"
                            icon="check"
                            disabled={isApplying}
                            onClick={() => void handleApplyChanges(false)}
                          >
                            {isApplying ? 'Applying…' : 'Apply Changes'}
                          </DarkroomButton>
                        </>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </section>
          </motion.div>
        </div>
      )}
      </AnimatePresence>
      <AppUpdateModal
        isOpen={settingsOpen && showReleaseNotesModal && updateCheckResult?.status === 'update_available'}
        update={updateCheckResult}
        mode="manual"
        onDismiss={() => setShowReleaseNotesModal(false)}
      />
    </>
  );
}
