import { memo } from 'react';
import type {
  ModelDownloadProgress,
  OptionalRetouchModelId,
} from '../../application/ports/studioGateway';
import { formatByteSize } from '../../domain/exportEstimate';
import {
  createDefaultPortraitRetouch,
  createDefaultRetouchSettings,
  isPortraitRetouchSliderActive,
  PORTRAIT_RETOUCH_PRESETS,
  type FaceParseMapData,
  type HealSpot,
  type HealSpotMode,
  type PortraitRetouchSettings,
  type RetouchSettings,
} from '../../domain/retouch';
import type { YuNetFaceOverlay } from '../../domain/photo';
import {
  DarkroomButton,
  DarkroomScrubber,
  DarkroomToggle,
  SegmentedControl,
  type SegmentedOption,
} from '../darkroom-kit';

export type RetouchToolMode = HealSpotMode | 'remove';

export interface HealToolOptions {
  readonly mode: RetouchToolMode;
  readonly radius: number; // 0.004..0.15 (fraction of shorter oriented image side)
  readonly feather: number; // 0..1
  readonly opacity: number; // 0.05..1
}

interface RetouchPanelProps {
  readonly retouch: RetouchSettings;
  readonly faceOverlay: YuNetFaceOverlay | null;
  readonly faceParseMap: FaceParseMapData | null;
  readonly detectingFace: boolean;
  readonly parsingFace: boolean;
  readonly faceParserReady: boolean;
  readonly lamaReady: boolean;
  readonly downloadingOptionalModelId: OptionalRetouchModelId | null;
  readonly modelDownloadProgress: ModelDownloadProgress | null;
  readonly pendingRemoveStrokesCount: number;
  readonly busyInpainting: boolean;
  readonly isDesktopRuntime: boolean;
  readonly healTool: HealToolOptions;
  readonly selectedSpotId: string | null;
  readonly showSpotPins: boolean;
  readonly onChangeHealTool: (next: HealToolOptions) => void;
  readonly onSelectSpot: (spotId: string | null) => void;
  readonly onToggleShowSpotPins: () => void;
  readonly onDeleteSpot: (spotId: string) => void;
  readonly onClearAllSpots: () => void;
  readonly onApplyInpaint: () => void;
  readonly onClearPendingRemoveStrokes: () => void;
  readonly onDeleteInpaintPatch: (patchId: string) => void;
  readonly onClearAllInpaintPatches: () => void;
  readonly onDownloadOptionalModel: (modelId: OptionalRetouchModelId) => void;
  readonly onCancelOptionalModelDownload: () => void;
  readonly onDetectFace: () => void;
  readonly onChangeRetouch: (next: RetouchSettings, label: string, coalesce?: boolean) => void;
}

const HEAL_MODE_OPTIONS: readonly SegmentedOption<RetouchToolMode>[] = [
  {
    id: 'heal',
    label: 'Heal',
    title: 'Seamless harmonic tone & texture healing (best for skin blemishes & hairs)',
  },
  {
    id: 'clone',
    label: 'Clone',
    title: 'Direct soft-edged pixel clone (best for crisp edges & backdrop cleanup)',
  },
  {
    id: 'remove',
    label: 'Remove',
    title: 'Paint over objects or distractions for generative LaMa / Harmonic removal',
  },
];

function computeDownloadPct(progress: ModelDownloadProgress | null): number {
  if (!progress || progress.totalBytes <= 0) return 0;
  return Math.min(100, Math.max(0, Math.round((progress.downloadedBytes / progress.totalBytes) * 100)));
}

export const RetouchPanel = memo(function RetouchPanel({
  retouch,
  faceOverlay,
  faceParseMap,
  detectingFace,
  parsingFace,
  faceParserReady,
  lamaReady,
  downloadingOptionalModelId,
  modelDownloadProgress,
  pendingRemoveStrokesCount,
  busyInpainting,
  isDesktopRuntime,
  healTool,
  selectedSpotId,
  showSpotPins,
  onChangeHealTool,
  onSelectSpot,
  onToggleShowSpotPins,
  onDeleteSpot,
  onClearAllSpots,
  onApplyInpaint,
  onClearPendingRemoveStrokes,
  onDeleteInpaintPatch,
  onClearAllInpaintPatches,
  onDownloadOptionalModel,
  onCancelOptionalModelDownload,
  onDetectFace,
  onChangeRetouch,
}: RetouchPanelProps) {
  const spots = retouch.healSpots;
  const inpaintPatches = retouch.inpaintPatches ?? [];
  const portrait = retouch.portrait;
  const hasPortraitEdits = isPortraitRetouchSliderActive(portrait);
  const selectedSpot: HealSpot | null =
    spots.find((s) => s.id === selectedSpotId) ?? null;
  const isRemoveMode = healTool.mode === 'remove';
  const downloadingLama = downloadingOptionalModelId === 'lama-inpainting';
  const downloadingBiSeNet = downloadingOptionalModelId === 'bisenet-face-parsing';
  const activeDownloadPct = computeDownloadPct(modelDownloadProgress);

  const patchPortrait = <K extends keyof PortraitRetouchSettings>(
    key: K,
    val: PortraitRetouchSettings[K],
    label: string,
    coalesce = true,
  ) => {
    onChangeRetouch(
      {
        ...retouch,
        portrait: {
          ...portrait,
          enabled: key === 'enabled' ? Boolean(val) : true,
          [key]: val,
        },
      },
      label,
      coalesce,
    );
  };

  const patchSelectedSpot = (patch: Partial<HealSpot>, label: string, coalesce = true) => {
    if (!selectedSpot) return;
    onChangeRetouch(
      {
        ...retouch,
        healSpots: spots.map((s) => (s.id === selectedSpot.id ? { ...s, ...patch } : s)),
      },
      label,
      coalesce,
    );
  };

  return (
    <div className="flex flex-col divide-y divide-hairline/55">
      {/* ── Section 1: Spot Heal, Clone & AI Object Removal ── */}
      <div className="px-4 py-3.5 flex flex-col gap-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-display text-[12px] font-semibold tracking-tight text-ink-primary truncate">
              {isRemoveMode ? 'Object Removal' : 'Spot Heal & Clone'}
            </span>
            {(spots.length + inpaintPatches.length) > 0 && (
              <span className="tabular-nums-mono text-[10px] text-amber-soft font-medium">
                {spots.length + inpaintPatches.length}
              </span>
            )}
          </div>
          {!isRemoveMode && (
            <DarkroomButton
              variant="ghost"
              size="xs"
              active={showSpotPins}
              onClick={onToggleShowSpotPins}
              title="Toggle spot pins on canvas (H)"
              aria-label="Toggle spot pins"
            >
              {showSpotPins ? 'Hide Pins' : 'Show Pins'}
            </DarkroomButton>
          )}
        </div>

        <SegmentedControl
          layoutId="retouch-heal-mode"
          options={HEAL_MODE_OPTIONS}
          value={isRemoveMode ? 'remove' : (selectedSpot ? selectedSpot.mode : healTool.mode)}
          onChange={(mode) => {
            onChangeHealTool({
              ...healTool,
              mode,
              radius: mode === 'remove' && healTool.radius < 0.025 ? 0.036 : healTool.radius,
            });
            if (mode === 'remove') {
              onSelectSpot(null);
            } else if (selectedSpot) {
              patchSelectedSpot({ mode }, `Spot Mode: ${mode}`, false);
            }
          }}
          className="w-full grid grid-cols-3"
        />

        <div className="flex flex-col gap-2">
          <DarkroomScrubber
            label="Brush Size"
            value={Math.round((!isRemoveMode && selectedSpot ? selectedSpot.radius : healTool.radius) * 1000)}
            min={4}
            max={150}
            step={1}
            defaultValue={isRemoveMode ? 36 : 18}
            formatValue={(v) => `${(v / 10).toFixed(1)}%`}
            onChange={(v) => {
              const radius = v / 1000;
              onChangeHealTool({ ...healTool, radius });
              if (!isRemoveMode && selectedSpot) {
                patchSelectedSpot({ radius }, 'Spot Size', true);
              }
            }}
          />
          {!isRemoveMode && (
            <>
              <DarkroomScrubber
                label="Feather"
                value={Math.round((selectedSpot ? selectedSpot.feather : healTool.feather) * 100)}
                min={0}
                max={100}
                step={1}
                defaultValue={55}
                formatValue={(v) => `${v}%`}
                onChange={(v) => {
                  const feather = v / 100;
                  onChangeHealTool({ ...healTool, feather });
                  if (selectedSpot) {
                    patchSelectedSpot({ feather }, 'Spot Feather', true);
                  }
                }}
              />
              <DarkroomScrubber
                label="Opacity"
                value={Math.round((selectedSpot ? selectedSpot.opacity : healTool.opacity) * 100)}
                min={5}
                max={100}
                step={1}
                defaultValue={100}
                formatValue={(v) => `${v}%`}
                onChange={(v) => {
                  const opacity = v / 100;
                  onChangeHealTool({ ...healTool, opacity });
                  if (selectedSpot) {
                    patchSelectedSpot({ opacity }, 'Spot Opacity', true);
                  }
                }}
              />
            </>
          )}
        </div>

        {isRemoveMode ? (
          <div className="flex flex-col gap-2.5">
            <p className="text-[10.5px] leading-relaxed text-ink-telemetry">
              Paint over a distraction, then press{' '}
              <span className="text-ink-secondary font-medium">Enter</span>. Resize with{' '}
              <span className="tabular-nums-mono text-ink-secondary">[ ]</span>.
            </p>

            <div className="flex items-center gap-1.5">
              <DarkroomButton
                variant="primary"
                size="sm"
                icon={busyInpainting ? 'progress_activity' : undefined}
                disabled={!isDesktopRuntime || pendingRemoveStrokesCount === 0 || busyInpainting}
                onClick={onApplyInpaint}
                className="flex-1 justify-center"
                title={
                  isDesktopRuntime
                    ? 'Synthesize seamless fill for painted strokes (Enter)'
                    : 'AI Remove requires Kroma Desktop'
                }
              >
                {busyInpainting
                  ? 'Removing…'
                  : pendingRemoveStrokesCount > 0
                    ? `Remove Object (${pendingRemoveStrokesCount})`
                    : 'Remove Object'}
              </DarkroomButton>
              {pendingRemoveStrokesCount > 0 && !busyInpainting && (
                <DarkroomButton
                  variant="ghost"
                  size="sm"
                  icon="close"
                  onClick={onClearPendingRemoveStrokes}
                  title="Clear painted selection (Esc)"
                  aria-label="Clear painted selection"
                />
              )}
            </div>

            {/* Subtle inline LaMa status */}
            <div className="flex flex-col gap-1.5 pt-0.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span
                    className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                      lamaReady ? 'bg-emerald-400' : 'bg-amber-core/80'
                    }`}
                  />
                  <span className="text-[10.5px] text-ink-telemetry truncate">
                    {lamaReady
                      ? 'LaMa Neural AI Ready'
                      : downloadingLama
                        ? `Installing LaMa AI (${activeDownloadPct}%)`
                        : 'Harmonic Fill (Built-in)'}
                  </span>
                </div>
                {isDesktopRuntime && !lamaReady && !downloadingLama && (
                  <DarkroomButton
                    variant="ghost"
                    size="xs"
                    icon="download"
                    disabled={downloadingOptionalModelId !== null}
                    onClick={() => onDownloadOptionalModel('lama-inpainting')}
                    title="Download optional LaMa generative neural inpainting model (~198 MB)"
                  >
                    LaMa AI (198 MB)
                  </DarkroomButton>
                )}
                {downloadingLama && (
                  <DarkroomButton
                    variant="ghost"
                    size="xs"
                    icon="close"
                    onClick={onCancelOptionalModelDownload}
                    title="Cancel download"
                    aria-label="Cancel LaMa download"
                  />
                )}
              </div>
              {downloadingLama && (
                <div className="flex flex-col gap-1">
                  <div className="h-1 w-full rounded-full bg-surface-lowest overflow-hidden">
                    <div
                      style={{ width: `${activeDownloadPct}%` }}
                      className="h-full bg-amber-core transition-[width] duration-150"
                    />
                  </div>
                  {modelDownloadProgress && modelDownloadProgress.totalBytes > 0 && (
                    <span className="tabular-nums-mono text-[9.5px] text-ink-telemetry">
                      {formatByteSize(modelDownloadProgress.downloadedBytes, true)} /{' '}
                      {formatByteSize(modelDownloadProgress.totalBytes, true)}
                      {modelDownloadProgress.bytesPerSecond > 0
                        ? ` · ${formatByteSize(modelDownloadProgress.bytesPerSecond, true)}/s`
                        : ''}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : (
          <p className="text-[10.5px] leading-relaxed text-ink-telemetry">
            Click a blemish or drag over stray hairs. Resize with{' '}
            <span className="tabular-nums-mono text-ink-secondary">[ ]</span>.
          </p>
        )}

        {inpaintPatches.length > 0 && (
          <div className="flex flex-col gap-1.5 pt-2 border-t border-hairline/50">
            <div className="flex items-center justify-between">
              <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                Removed ({inpaintPatches.length})
              </span>
              <button
                type="button"
                onClick={onClearAllInpaintPatches}
                title="Remove all object removal patches on this photo"
                className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-amber-soft transition-colors cursor-pointer"
              >
                Clear
              </button>
            </div>
            <div className="max-h-24 overflow-y-auto flex flex-col gap-0.5 pr-0.5">
              {inpaintPatches.map((patch, idx) => (
                <div
                  key={patch.id}
                  className="flex items-center justify-between gap-2 px-2 py-1.5 rounded-md hover:bg-surface/65 text-[11px] text-ink-secondary transition-colors"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="w-1.5 h-1.5 rounded-full shrink-0 bg-rose-400" />
                    <span className="truncate font-medium text-ink-primary">
                      Remove #{idx + 1}
                    </span>
                    <span className="tabular-nums-mono text-[9.5px] text-ink-telemetry truncate">
                      {patch.engineLabel}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => onDeleteInpaintPatch(patch.id)}
                    title="Delete object removal patch"
                    aria-label={`Remove object patch ${idx + 1}`}
                    className="text-ink-telemetry hover:text-rose-400 transition-colors cursor-pointer shrink-0 flex items-center"
                  >
                    <span className="material-symbols-outlined text-[13px]">close</span>
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {spots.length > 0 && (
          <div className="flex flex-col gap-1.5 pt-2 border-t border-hairline/50">
            <div className="flex items-center justify-between">
              <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
                Spots ({spots.length})
              </span>
              <div className="flex items-center gap-2">
                {selectedSpot && (
                  <button
                    type="button"
                    onClick={() => onDeleteSpot(selectedSpot.id)}
                    title="Delete selected spot (Del)"
                    className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-rose-400 transition-colors cursor-pointer"
                  >
                    Delete
                  </button>
                )}
                <button
                  type="button"
                  onClick={onClearAllSpots}
                  title="Remove all heal and clone spots on this photo"
                  className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-amber-soft transition-colors cursor-pointer"
                >
                  Clear All
                </button>
              </div>
            </div>
            <div className="max-h-28 overflow-y-auto flex flex-col gap-0.5 pr-0.5">
              {spots.map((spot, idx) => {
                const isSel = spot.id === selectedSpotId;
                const isStroke = Boolean(spot.points && spot.points.length > 1);
                return (
                  <div
                    key={spot.id}
                    onClick={() => onSelectSpot(isSel ? null : spot.id)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onSelectSpot(isSel ? null : spot.id);
                      }
                    }}
                    className={`flex items-center justify-between gap-2 px-2 py-1.5 rounded-md text-[11px] transition-colors cursor-pointer ${
                      isSel
                        ? 'bg-surface-active text-ink-primary'
                        : 'text-ink-secondary hover:bg-surface/65 hover:text-ink-primary'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span
                        className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                          spot.mode === 'clone' ? 'bg-sky-400' : 'bg-amber-core'
                        }`}
                      />
                      <span className="truncate font-medium">
                        {spot.mode === 'clone' ? 'Clone' : 'Heal'} #{idx + 1}
                        {isStroke ? ' Stroke' : ''}
                      </span>
                      <span className="tabular-nums-mono text-[9.5px] text-ink-telemetry">
                        {Math.round(spot.opacity * 100)}%
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSpot(spot.id);
                      }}
                      title="Remove spot"
                      aria-label={`Remove spot ${idx + 1}`}
                      className="text-ink-telemetry hover:text-rose-400 transition-colors cursor-pointer flex items-center"
                    >
                      <span className="material-symbols-outlined text-[13px]">close</span>
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── Section 2: Portrait Retouch — Frequency Separation & Features ── */}
      <div className="px-4 py-3.5 flex flex-col gap-3">
        <div className="flex items-center justify-between gap-2">
          <span className="font-display text-[12px] font-semibold tracking-tight text-ink-primary truncate">
            Portrait Retouch
          </span>
          <div className="flex items-center gap-2 shrink-0">
            {hasPortraitEdits && (
              <button
                type="button"
                onClick={() =>
                  onChangeRetouch(
                    { ...retouch, portrait: createDefaultPortraitRetouch() },
                    'Reset Portrait Retouch',
                    false,
                  )
                }
                title="Reset all portrait retouching sliders"
                className="tabular-nums-mono text-[10px] text-ink-telemetry hover:text-amber-soft transition-colors cursor-pointer"
              >
                Reset
              </button>
            )}
            <DarkroomToggle
              checked={portrait.enabled}
              onChange={(enabled) =>
                patchPortrait('enabled', enabled, enabled ? 'Enable Portrait Retouch' : 'Bypass Portrait Retouch', false)
              }
              aria-label="Toggle portrait retouching"
            />
          </div>
        </div>

        {/* Subtle inline Face Parsing / Landmark status */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 min-w-0">
              <span
                className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                  faceParseMap || faceOverlay ? 'bg-emerald-400' : 'bg-amber-core/80'
                }`}
              />
              <span
                className="text-[10.5px] text-ink-telemetry truncate"
                title={
                  faceParseMap
                    ? 'Pixel-exact 19-class neural skin, eye, teeth & lip masks (protects glasses & hairlines)'
                    : undefined
                }
              >
                {faceParseMap
                  ? 'BiSeNet Neural Parsing Locked'
                  : parsingFace
                    ? 'Parsing neural face regions…'
                    : faceOverlay
                      ? 'YuNet Face Landmarks Locked'
                      : detectingFace
                        ? 'Detecting facial landmarks…'
                        : 'Skin Tone Auto (no landmarks)'}
              </span>
            </div>
            {!faceOverlay && !faceParseMap && isDesktopRuntime && (
              <DarkroomButton
                variant="ghost"
                size="xs"
                icon={detectingFace ? 'progress_activity' : undefined}
                disabled={detectingFace}
                onClick={onDetectFace}
                title="Detect face landmarks for precision Eyes, Under-Eye, Teeth & Lips retouching"
              >
                {detectingFace ? 'Scanning' : 'Detect'}
              </DarkroomButton>
            )}
          </div>

          {isDesktopRuntime && !faceParserReady && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] text-ink-telemetry/80 truncate">
                  {downloadingBiSeNet
                    ? `Installing BiSeNet Parsing (${activeDownloadPct}%)`
                    : 'Optional: Pixel-exact BiSeNet mask'}
                </span>
                {downloadingBiSeNet ? (
                  <DarkroomButton
                    variant="ghost"
                    size="xs"
                    icon="close"
                    onClick={onCancelOptionalModelDownload}
                    title="Cancel download"
                    aria-label="Cancel BiSeNet download"
                  />
                ) : (
                  <DarkroomButton
                    variant="ghost"
                    size="xs"
                    icon="download"
                    disabled={downloadingOptionalModelId !== null}
                    onClick={() => onDownloadOptionalModel('bisenet-face-parsing')}
                    title="Download optional BiSeNet 19-class face parsing model (~51 MB) to protect eyeglasses, hairlines & refine AI Face & Skin masks"
                  >
                    51 MB
                  </DarkroomButton>
                )}
              </div>
              {downloadingBiSeNet && (
                <div className="h-1 w-full rounded-full bg-surface-lowest overflow-hidden">
                  <div
                    style={{ width: `${activeDownloadPct}%` }}
                    className="h-full bg-emerald-400 transition-[width] duration-150"
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* One-Click Portrait Presets */}
        <div className="grid grid-cols-3 gap-1.5">
          {PORTRAIT_RETOUCH_PRESETS.map((preset) => (
            <DarkroomButton
              key={preset.id}
              variant="subtle"
              size="xs"
              onClick={() =>
                onChangeRetouch(
                  { ...retouch, portrait: { ...preset.portrait } },
                  `Portrait: ${preset.label}`,
                  false,
                )
              }
              title={preset.summary}
              className="justify-center"
            >
              {preset.label}
            </DarkroomButton>
          ))}
        </div>

        {/* Skin — 3-Band Frequency Separation */}
        <div
          className={`flex flex-col gap-2 transition-opacity ${
            portrait.enabled ? 'opacity-100' : 'opacity-45 pointer-events-none'
          }`}
        >
          <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry pt-0.5">
            Skin
          </span>

          <DarkroomScrubber
            label="Skin Smoothing"
            value={portrait.skinSmoothing}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            formatValue={(v) => `${v}`}
            onChange={(v) => patchPortrait('skinSmoothing', v, 'Skin Smoothing')}
          />
          <DarkroomScrubber
            label="Texture Preserve"
            value={portrait.skinTexturePreserve}
            min={0}
            max={100}
            step={1}
            defaultValue={70}
            formatValue={(v) => `${v}%`}
            onChange={(v) => patchPortrait('skinTexturePreserve', v, 'Texture Preserve')}
          />
          <DarkroomScrubber
            label="Tone Evenness"
            value={portrait.skinToneEvenness}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            formatValue={(v) => `${v}`}
            onChange={(v) => patchPortrait('skinToneEvenness', v, 'Skin Tone Evenness')}
          />
          <DarkroomScrubber
            label="Shine Removal"
            value={portrait.shineRemoval}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            formatValue={(v) => `${v}`}
            onChange={(v) => patchPortrait('shineRemoval', v, 'Shine Removal')}
          />

          <div className="h-px bg-hairline/55 my-1" />

          {/* Landmark-Guided Facial Features */}
          <div className="flex items-center justify-between">
            <span className="tabular-nums-mono text-[9.5px] uppercase tracking-wider text-ink-telemetry">
              Eyes &amp; Smile
            </span>
            {!faceOverlay && (
              <span className="tabular-nums-mono text-[9.5px] text-amber-soft/80">
                Needs face lock
              </span>
            )}
          </div>

          <DarkroomScrubber
            label="Under-Eye Brighten"
            value={portrait.underEyeBrighten}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            formatValue={(v) => `${v}`}
            onChange={(v) => patchPortrait('underEyeBrighten', v, 'Under-Eye Brighten')}
          />
          <DarkroomScrubber
            label="Eye Enhance"
            value={portrait.eyeEnhance}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            formatValue={(v) => `${v}`}
            onChange={(v) => patchPortrait('eyeEnhance', v, 'Eye Enhance')}
          />
          <DarkroomScrubber
            label="Teeth Whiten"
            value={portrait.teethWhiten}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            formatValue={(v) => `${v}`}
            onChange={(v) => patchPortrait('teethWhiten', v, 'Teeth Whiten')}
          />
          <DarkroomScrubber
            label="Lip Enhance"
            value={portrait.lipEnhance}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            formatValue={(v) => `${v}`}
            onChange={(v) => patchPortrait('lipEnhance', v, 'Lip Enhance')}
          />
        </div>
      </div>

      {(spots.length > 0 || inpaintPatches.length > 0 || hasPortraitEdits) && (
        <div className="px-4 py-3">
          <DarkroomButton
            variant="ghost"
            size="xs"
            onClick={() =>
              onChangeRetouch(createDefaultRetouchSettings(), 'Reset All Retouching', false)
            }
            className="w-full justify-center"
          >
            Reset All Retouching
          </DarkroomButton>
        </div>
      )}
    </div>
  );
});
