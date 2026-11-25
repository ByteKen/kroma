import { useEffect, useMemo, useState } from 'react';
import type { SyncTargetScope } from '../../app/StudioContext';
import {
  createColorAndToneSyncMask,
  getModifiedDevelopModules,
  syncDevelopRecipe,
  type DevelopSyncMask,
} from '../../domain/develop';
import type { PhotoAsset } from '../../domain/photo';
import { DarkroomButton, DarkroomModal, DarkroomToggle, SegmentedControl } from '../darkroom-kit';
import { GradedThumbnailImg } from './GradedThumbnailImg';

interface SyncSettingsModalProps {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly sourcePhoto: PhotoAsset;
  readonly scopeLabel: string;
  readonly scopePhotos: readonly PhotoAsset[];
  readonly allProjectPhotos: readonly PhotoAsset[];
  readonly selectedPhotoIds: ReadonlySet<string>;
  readonly onConfirmSync: (mask: DevelopSyncMask, scope: SyncTargetScope) => void;
}

const MASK_ITEMS: readonly {
  readonly key: keyof DevelopSyncMask;
  readonly label: string;
  readonly hint: string;
}[] = [
  { key: 'whiteBalance', label: 'White Balance', hint: 'Temp (Kelvin) & Tint' },
  {
    key: 'basicTone',
    label: 'Basic Tone & Presence',
    hint: 'Exposure, Contrast, Highlights, Shadows, Vibrance',
  },
  { key: 'toneCurve', label: 'Tone Curve', hint: 'RGB & per-channel R/G/B curves' },
  { key: 'hslMixer', label: 'HSL / B&W Color Mix', hint: 'Red through Magenta Hue/Sat/Lum' },
  {
    key: 'colorGrading',
    label: 'Color Grading (3-Way)',
    hint: 'Shadows, Midtones, Highlights wheels & Balance',
  },
  {
    key: 'cameraCalibration',
    label: 'Camera Calibration',
    hint: 'Shadows Tint & R/G/B Primary Hue/Sat',
  },
  { key: 'lut3d', label: '3D LUT & Intensity', hint: 'Creative studio color profile' },
  {
    key: 'detailOptics',
    label: 'Detail, Vignette & Grain',
    hint: 'Sharpening, Luma/Color NR, Vignette & Film Grain',
  },
  {
    key: 'cropTransform',
    label: 'Crop, Flip & Straighten',
    hint: 'Off by default to preserve framing',
  },
  {
    key: 'masks',
    label: 'Masks & Background Cutout',
    hint: 'Copy local masks, AI mattes & background removal (off by default)',
  },
  {
    key: 'portraitRetouch',
    label: 'Portrait Retouching',
    hint: 'Frequency-separation skin, shine, eyes, teeth & lips (auto-adapts to each face)',
  },
  {
    key: 'healSpots',
    label: 'Spot Heal & Clone',
    hint: 'Per-photo blemish & stray-hair patches (off by default)',
  },
];

const MAX_PREVIEW_TARGETS = 5;

export function SyncSettingsModal({
  open,
  onClose,
  sourcePhoto,
  scopeLabel,
  scopePhotos,
  allProjectPhotos,
  selectedPhotoIds,
  onConfirmSync,
}: SyncSettingsModalProps) {
  const [mask, setMask] = useState<DevelopSyncMask>(createColorAndToneSyncMask);
  const [scope, setScope] = useState<SyncTargetScope>('selected_in_cluster');

  // Default to "Selected" every time the dialog opens so batch edits stay intentional & safe
  useEffect(() => {
    if (open) {
      setScope('selected_in_cluster');
    }
  }, [open]);

  const modifiedModules = useMemo(
    () => getModifiedDevelopModules(sourcePhoto.developRecipe),
    [sourcePhoto.developRecipe],
  );
  const hasAnyModifiedModule = useMemo(
    () => Object.values(modifiedModules).some(Boolean),
    [modifiedModules],
  );

  const selectedTargetPhotos = useMemo(
    () =>
      allProjectPhotos.filter(
        (p) => selectedPhotoIds.has(p.id) && p.id !== sourcePhoto.id,
      ),
    [allProjectPhotos, selectedPhotoIds, sourcePhoto.id],
  );
  const selectedCount = selectedTargetPhotos.length;
  const scopeTargetCount = useMemo(
    () => scopePhotos.filter((p) => p.id !== sourcePhoto.id).length,
    [scopePhotos, sourcePhoto.id],
  );
  const projectTargetCount = useMemo(
    () => allProjectPhotos.filter((p) => p.id !== sourcePhoto.id).length,
    [allProjectPhotos, sourcePhoto.id],
  );

  const targetPhotos = useMemo<readonly PhotoAsset[]>(() => {
    if (!open) return [];
    if (scope === 'selected_in_cluster') {
      return selectedTargetPhotos;
    }
    if (scope === 'entire_cluster') {
      return scopePhotos.filter((p) => p.id !== sourcePhoto.id);
    }
    return allProjectPhotos.filter((p) => p.id !== sourcePhoto.id);
  }, [open, scope, scopePhotos, allProjectPhotos, selectedTargetPhotos, sourcePhoto.id]);

  const previewTargets = useMemo(() => targetPhotos.slice(0, MAX_PREVIEW_TARGETS), [targetPhotos]);

  const overflowCount = Math.max(0, targetPhotos.length - previewTargets.length);
  const checkedModuleCount = useMemo(
    () => Object.values(mask).filter(Boolean).length,
    [mask],
  );

  const toggleKey = (key: keyof DevelopSyncMask, nextVal: boolean) => {
    setMask((prev) => ({ ...prev, [key]: nextVal }));
  };

  const setAll = (checked: boolean) => {
    setMask({
      whiteBalance: checked,
      basicTone: checked,
      toneCurve: checked,
      hslMixer: checked,
      colorGrading: checked,
      cameraCalibration: checked,
      lut3d: checked,
      detailOptics: checked,
      cropTransform: checked,
      masks: checked,
      portraitRetouch: checked,
      healSpots: checked,
    });
  };

  const applyDisabled = targetPhotos.length === 0 || checkedModuleCount === 0;
  const confirmLabel =
    targetPhotos.length > 0
      ? `Sync to ${targetPhotos.length.toLocaleString()} ${targetPhotos.length === 1 ? 'Photo' : 'Photos'}`
      : 'Sync to Selected';

  return (
    <DarkroomModal
      open={open}
      onClose={onClose}
      icon="tune"
      maxWidthClass="max-w-2xl"
      title="Sync to Other Photos"
      subtitle={`Edits on ${sourcePhoto.fileName} auto-save automatically — copy checked modules to other photos`}
      footer={
        <>
          <div className="flex items-center gap-1">
            {hasAnyModifiedModule && (
              <DarkroomButton
                variant="subtle"
                size="xs"
                onClick={() =>
                  setMask({ ...modifiedModules, cropTransform: false, healSpots: false })
                }
                title="Select only adjustments edited on this photo"
              >
                Modified Only
              </DarkroomButton>
            )}
            <DarkroomButton variant="ghost" size="xs" onClick={() => setAll(true)}>
              All
            </DarkroomButton>
            <DarkroomButton variant="ghost" size="xs" onClick={() => setAll(false)}>
              None
            </DarkroomButton>
          </div>
          <div className="flex-1" />
          <DarkroomButton variant="secondary" size="md" onClick={onClose}>
            Cancel
          </DarkroomButton>
          <DarkroomButton
            variant="primary"
            size="md"
            icon="done_all"
            disabled={applyDisabled}
            onClick={() => {
              onConfirmSync(mask, scope);
              onClose();
            }}
          >
            {confirmLabel}
          </DarkroomButton>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        {/* Target Scope Selector (Selected by default) */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry">
              Apply To
            </span>
            <span className="tabular-nums-mono text-[10px] text-ink-telemetry">
              {checkedModuleCount} of {MASK_ITEMS.length} modules checked
            </span>
          </div>
          <SegmentedControl
            layoutId="sync-target-scope"
            value={scope}
            onChange={setScope}
            className="w-full justify-between"
            options={[
              {
                id: 'selected_in_cluster',
                label: `Selected (${selectedCount})`,
                title: 'Apply only to checked filmstrip photos',
              },
              {
                id: 'entire_cluster',
                label: `${scopeLabel} (${scopeTargetCount.toLocaleString()})`,
                title: 'Apply to all photos in current folder or subject',
              },
              {
                id: 'entire_shoot',
                label: `Entire Project (${projectTargetCount.toLocaleString()})`,
                title: 'Apply across every photo in this project',
              },
            ]}
          />
        </div>

        {/* Live Source -> Target Contact-Sheet Preview */}
        <div className="p-3 rounded-lg bg-canvas border border-hairline flex items-center gap-3">
          {/* Source Thumbnail */}
          <div className="flex flex-col gap-1 shrink-0 w-[68px]">
            <div className="relative h-[86px] w-[68px] rounded-md overflow-hidden border border-amber-core/70 bg-surface-lowest">
              <GradedThumbnailImg
                src={sourcePhoto.imageUrl}
                renderedSrc={sourcePhoto.renderedThumbnailUrl}
                alt={sourcePhoto.fileName}
                recipe={sourcePhoto.developRecipe}
                loading="eager"
                decoding="async"
                className="w-full h-full object-cover object-top"
              />
              <span className="absolute top-1 left-1 px-1 py-0.5 rounded bg-amber-core text-amber-ink tabular-nums-mono text-[7.5px] font-bold uppercase tracking-wider leading-none">
                SRC
              </span>
            </div>
            <span className="tabular-nums-mono text-[9px] text-ink-secondary truncate text-center">
              {sourcePhoto.fileName.replace(/\.[^.]+$/, '')}
            </span>
          </div>

          <span className="material-symbols-outlined text-[18px] text-amber-soft shrink-0">
            arrow_forward
          </span>

          {/* Live Target Previews (updates dynamically as modules are toggled!) */}
          <div className="flex-1 min-w-0">
            {previewTargets.length === 0 ? (
              <div className="h-[86px] rounded-md bg-surface-low/80 border border-hairline px-3.5 flex items-center justify-between gap-3">
                <div className="min-w-0 flex flex-col gap-0.5">
                  <span className="font-display text-[12px] font-semibold text-ink-primary">
                    No photos selected in the filmstrip
                  </span>
                  <span className="text-[11px] text-ink-telemetry">
                    Check thumbnails in the bottom filmstrip, or switch target scope above.
                  </span>
                </div>
                {scopeTargetCount > 0 && (
                  <DarkroomButton
                    variant="subtle"
                    size="xs"
                    onClick={() => setScope('entire_cluster')}
                    className="shrink-0"
                  >
                    Use Current ({scopeTargetCount.toLocaleString()})
                  </DarkroomButton>
                )}
              </div>
            ) : (
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2 overflow-hidden">
                  {previewTargets.map((photo) => {
                    const previewRecipe = syncDevelopRecipe(
                      photo.developRecipe,
                      sourcePhoto.developRecipe,
                      mask,
                    );

                    return (
                      <div key={photo.id} className="flex flex-col gap-1 shrink-0 w-[68px]">
                        <div className="relative h-[86px] w-[68px] rounded-md overflow-hidden border border-hairline-strong bg-surface-lowest">
                          <GradedThumbnailImg
                            src={photo.imageUrl}
                            alt={photo.fileName}
                            recipe={previewRecipe}
                            loading="lazy"
                            decoding="async"
                            className="w-full h-full object-cover object-top"
                          />
                        </div>
                        <span className="tabular-nums-mono text-[9px] text-ink-telemetry truncate text-center">
                          {photo.fileName.replace(/\.[^.]+$/, '')}
                        </span>
                      </div>
                    );
                  })}

                  {overflowCount > 0 && (
                    <div className="flex flex-col gap-1 shrink-0 w-[68px]">
                      <div className="h-[86px] w-[68px] rounded-md border border-hairline bg-surface-low flex flex-col items-center justify-center text-center px-1">
                        <span className="tabular-nums-mono text-[12px] font-bold text-amber-soft">
                          +{overflowCount.toLocaleString()}
                        </span>
                        <span className="tabular-nums-mono text-[8.5px] text-ink-telemetry uppercase">
                          more
                        </span>
                      </div>
                      <span className="tabular-nums-mono text-[9px] text-ink-telemetry text-center">
                        total {targetPhotos.length.toLocaleString()}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Compact 2-Column Adjustment Checklist */}
        <div className="flex flex-col gap-1.5">
          <span className="tabular-nums-mono text-[10px] uppercase tracking-wider text-ink-telemetry">
            Adjustments to Apply
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {MASK_ITEMS.map((item) => {
              const isEdited = modifiedModules[item.key];
              const isChecked = mask[item.key];
              return (
                <div
                  key={item.key}
                  className={`px-3 py-2 rounded-lg border transition-colors flex flex-col gap-0.5 ${
                    isChecked
                      ? 'bg-canvas border-hairline-strong'
                      : 'bg-surface-low/60 border-hairline opacity-75'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5 min-w-0 flex-1">
                      <DarkroomToggle
                        checked={isChecked}
                        onChange={(val) => toggleKey(item.key, val)}
                        label={item.label}
                        className="flex-1"
                      />
                    </div>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[10px] text-ink-telemetry truncate">{item.hint}</span>
                    {isEdited && (
                      <span className="shrink-0 tabular-nums-mono text-[8.5px] px-1.5 py-0.5 rounded bg-amber-core/15 text-amber-soft border border-amber-core/30 uppercase tracking-wider">
                        Edited
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </DarkroomModal>
  );
}
