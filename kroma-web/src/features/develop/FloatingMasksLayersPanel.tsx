import { memo, useEffect, useRef, useState } from 'react';
import {
  backgroundRemovalPreviewCss,
  compileMaskLayerForWebGl,
  createMaskId,
  isBackgroundRemovalActive,
  maskComponentLabel,
  type AiMaskKind,
  type BackgroundRemovalSettings,
  type MaskCombineOp,
  type MaskLayer,
} from '../../domain/masking';

interface MaskMiniThumbnailProps {
  readonly layer: MaskLayer;
}

/** Live 28x28 monochrome alpha preview thumbnail (Lightroom Masks panel style). */
const MaskMiniThumbnail = memo(function MaskMiniThumbnail({ layer }: MaskMiniThumbnailProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const compiled = compileMaskLayerForWebGl(layer, 28, 36, 36);
    const w = compiled.spatialWidth;
    const h = compiled.spatialHeight;
    canvas.width = w;
    canvas.height = h;
    const imgData = ctx.createImageData(w, h);
    if (!compiled.hasSpatial) {
      // Luminance/Color range only: draw stylized tonal gradient swatch
      for (let y = 0; y < h; y++) {
        const lum = Math.round((1 - y / Math.max(1, h - 1)) * 210);
        for (let x = 0; x < w; x++) {
          const o = (y * w + x) * 4;
          imgData.data[o] = lum;
          imgData.data[o + 1] = lum;
          imgData.data[o + 2] = lum;
          imgData.data[o + 3] = 255;
        }
      }
    } else {
      for (let i = 0; i < compiled.spatialRgba.length; i += 4) {
        const raw = compiled.spatialRgba[i]!;
        const v = layer.inverted ? 255 - raw : raw;
        imgData.data[i] = v;
        imgData.data[i + 1] = v;
        imgData.data[i + 2] = v;
        imgData.data[i + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [layer]);

  return (
    <canvas
      ref={canvasRef}
      className="w-7 h-8 rounded-[4px] bg-black border border-hairline-strong shrink-0 object-cover"
    />
  );
});

interface FloatingMasksLayersPanelProps {
  readonly masks: readonly MaskLayer[];
  readonly backgroundRemoval: BackgroundRemovalSettings;
  readonly activeMaskId: string | null;
  readonly activeComponentId: string | null;
  readonly showMaskOverlay: boolean;
  readonly busyAiKind: AiMaskKind | null;
  readonly onSelectMask: (layerId: string | null, componentId?: string | null) => void;
  readonly onToggleMaskOverlay: () => void;
  readonly onAttachManualComponent: (
    layerId: string,
    type: 'brush' | 'linear_gradient' | 'radial_gradient' | 'luminance_range' | 'color_range',
    op: MaskCombineOp,
  ) => void;
  readonly onAttachAiComponent: (layerId: string, kind: AiMaskKind, op: MaskCombineOp) => void;
  readonly onChangeMasks: (
    nextMasks: readonly MaskLayer[],
    label: string,
    coalesce?: boolean,
  ) => void;
  readonly onToggleBackgroundRemoval: () => void;
}

const COMPOSE_TOOLS: readonly {
  readonly type: 'brush' | 'linear_gradient' | 'radial_gradient' | 'luminance_range' | 'color_range';
  readonly label: string;
}[] = [
    { type: 'brush', label: 'Brush' },
    { type: 'linear_gradient', label: 'Linear' },
    { type: 'radial_gradient', label: 'Radial' },
    { type: 'luminance_range', label: 'Luminance' },
    { type: 'color_range', label: 'Color' },
  ];

/**
 * Floating Lightroom-style Masks & Layers card docked cleanly at top-right of the photo viewport
 * whenever the Masking tool (`M`) is active.
 */
export const FloatingMasksLayersPanel = memo(function FloatingMasksLayersPanel({
  masks,
  backgroundRemoval,
  activeMaskId,
  activeComponentId,
  showMaskOverlay,
  busyAiKind,
  onSelectMask,
  onToggleMaskOverlay,
  onAttachManualComponent,
  onAttachAiComponent,
  onChangeMasks,
  onToggleBackgroundRemoval,
}: FloatingMasksLayersPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [composeMenuOp, setComposeMenuOp] = useState<MaskCombineOp | null>(null);

  const activeLayer = masks.find((m) => m.id === activeMaskId) ?? null;
  const bgActive = isBackgroundRemovalActive(backgroundRemoval);

  const updateLayer = (
    layerId: string,
    updater: (layer: MaskLayer) => MaskLayer,
    label: string,
  ) => {
    onChangeMasks(
      masks.map((m) => (m.id === layerId ? updater(m) : m)),
      label,
      false,
    );
  };

  return (
    <aside
      aria-label="Masks Layers"
      className="absolute top-3 right-3 z-30 w-[236px] max-h-[calc(100%-24px)] rounded-xl bg-surface-lowest/92 backdrop-blur-md border border-hairline-strong shadow-[0_12px_32px_rgba(0,0,0,0.6)] flex flex-col overflow-hidden"
    >
      {/* Floating Card Header */}
      <div className="px-3 py-2 bg-surface/85 border-b border-hairline flex items-center justify-between gap-2 shrink-0">
        <button
          type="button"
          onClick={() => setCollapsed((v) => !v)}
          className="flex items-center gap-1.5 text-left min-w-0 flex-1 cursor-pointer"
        >
          <span
            className={`material-symbols-outlined text-[15px] text-ink-telemetry transition-transform ${collapsed ? '' : 'rotate-90'
              }`}
          >
            chevron_right
          </span>
          <span className="font-display text-[11.5px] font-semibold text-ink-primary">
            Masks
          </span>
          <span className="tabular-nums-mono text-[10px] text-ink-telemetry">
            ({masks.length})
          </span>
        </button>
      </div>

      {!collapsed && (
        <>
          <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-1.5">
            {/* Adjustment Mask Layers (applied on top of the photo + any Solid/Gradient Backdrop) */}
            {masks.length === 0 && !bgActive ? (
              <div className="py-3 px-2 text-center text-[10.5px] text-ink-telemetry leading-snug">
                No masks yet. Choose a tool on the right.
              </div>
            ) : (
              masks.map((layer) => {
                const selected = activeLayer?.id === layer.id;
                return (
                  <div
                    key={layer.id}
                    onClick={() =>
                      onSelectMask(
                        layer.id,
                        layer.components[layer.components.length - 1]?.id ?? null,
                      )
                    }
                    className={`rounded-lg border p-1.5 transition-colors cursor-pointer ${selected
                        ? 'bg-surface-active/95 border-amber-core/65'
                        : 'bg-surface/70 border-hairline hover:border-hairline-strong'
                      }`}
                  >
                    <div className="flex items-center gap-2">
                      <MaskMiniThumbnail layer={layer} />
                      <div className="min-w-0 flex-1">
                        <div
                          className={`font-display text-[11px] font-medium truncate ${layer.enabled ? 'text-ink-primary' : 'text-ink-telemetry line-through'
                            }`}
                        >
                          {layer.name}
                        </div>
                        <div className="tabular-nums-mono text-[9px] text-ink-telemetry truncate">
                          {layer.components.length}{' '}
                          {layer.components.length === 1 ? 'component' : 'components'}
                          {layer.inverted ? ' · Inverted' : ''}
                        </div>
                      </div>
                      <div className="flex items-center gap-0.5 shrink-0">
                        <button
                          type="button"
                          title={layer.enabled ? 'Hide mask' : 'Show mask'}
                          onClick={(e) => {
                            e.stopPropagation();
                            updateLayer(
                              layer.id,
                              (l) => ({ ...l, enabled: !l.enabled }),
                              layer.enabled ? 'Hide Mask' : 'Show Mask',
                            );
                          }}
                          className={`p-0.5 rounded cursor-pointer ${layer.enabled
                              ? 'text-amber-soft'
                              : 'text-ink-telemetry opacity-50'
                            }`}
                        >
                          <span className="material-symbols-outlined text-[13px]">
                            {layer.enabled ? 'visibility' : 'visibility_off'}
                          </span>
                        </button>
                        <button
                          type="button"
                          title="Invert mask"
                          onClick={(e) => {
                            e.stopPropagation();
                            updateLayer(
                              layer.id,
                              (l) => ({ ...l, inverted: !l.inverted }),
                              'Invert Mask',
                            );
                          }}
                          className={`p-0.5 rounded cursor-pointer ${layer.inverted
                              ? 'text-amber-core bg-amber-core/15'
                              : 'text-ink-telemetry hover:text-ink-primary'
                            }`}
                        >
                          <span className="material-symbols-outlined text-[13px]">
                            invert_colors
                          </span>
                        </button>
                        <button
                          type="button"
                          title="Duplicate mask"
                          onClick={(e) => {
                            e.stopPropagation();
                            const copy: MaskLayer = {
                              ...layer,
                              id: createMaskId('layer'),
                              name: `${layer.name} Copy`,
                              components: layer.components.map((c) => ({
                                ...c,
                                id: createMaskId(c.type.slice(0, 3)),
                              })),
                              adjustments: { ...layer.adjustments },
                            };
                            onChangeMasks([...masks, copy], 'Duplicate Mask', false);
                            onSelectMask(copy.id, copy.components[0]?.id ?? null);
                          }}
                          className="p-0.5 rounded text-ink-telemetry hover:text-ink-primary cursor-pointer"
                        >
                          <span className="material-symbols-outlined text-[13px]">
                            content_copy
                          </span>
                        </button>
                        <button
                          type="button"
                          title="Delete mask"
                          onClick={(e) => {
                            e.stopPropagation();
                            const rest = masks.filter((m) => m.id !== layer.id);
                            onChangeMasks(rest, 'Delete Mask', false);
                            if (selected) {
                              const next = rest[0] ?? null;
                              onSelectMask(next?.id ?? null, next?.components[0]?.id ?? null);
                            }
                          }}
                          className="p-0.5 rounded text-ink-telemetry hover:text-rose-400 cursor-pointer"
                        >
                          <span className="material-symbols-outlined text-[13px]">delete</span>
                        </button>
                      </div>
                    </div>

                    {/* Selected Layer Nested Sub-Masks + Add/Subtract/Intersect */}
                    {selected && (
                      <div className="mt-1.5 pt-1.5 border-t border-hairline/70 flex flex-col gap-1">
                        {layer.components.map((comp, idx) => {
                          const isCompActive =
                            (activeComponentId ?? layer.components[layer.components.length - 1]?.id) ===
                            comp.id;
                          const badge =
                            idx === 0 && comp.op === 'add'
                              ? null
                              : comp.op === 'subtract'
                                ? '−'
                                : comp.op === 'intersect'
                                  ? '∩'
                                  : '+';
                          return (
                            <div
                              key={comp.id}
                              onClick={(e) => {
                                e.stopPropagation();
                                onSelectMask(layer.id, comp.id);
                              }}
                              className={`px-1.5 py-0.5 rounded text-[10px] flex items-center justify-between gap-1 border ${isCompActive
                                  ? 'bg-canvas border-amber-core/60 text-ink-primary'
                                  : 'bg-surface-low/80 border-hairline/60 text-ink-secondary hover:text-ink-primary'
                                }`}
                            >
                              <div className="flex items-center gap-1 min-w-0">
                                {badge && (
                                  <span
                                    className={`w-3.5 h-3.5 rounded text-[9px] font-bold flex items-center justify-center shrink-0 ${comp.op === 'subtract'
                                        ? 'bg-rose-500/20 text-rose-300'
                                        : comp.op === 'intersect'
                                          ? 'bg-sky-500/20 text-sky-300'
                                          : 'bg-emerald-500/20 text-emerald-300'
                                      }`}
                                  >
                                    {badge}
                                  </span>
                                )}
                                <span className="truncate font-display">
                                  {maskComponentLabel(comp)}
                                </span>
                              </div>
                              {layer.components.length > 1 && (
                                <button
                                  type="button"
                                  title="Remove component"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    const nextComps = layer.components.filter(
                                      (c) => c.id !== comp.id,
                                    );
                                    updateLayer(
                                      layer.id,
                                      (l) => ({ ...l, components: nextComps }),
                                      'Remove Sub-mask',
                                    );
                                    if (isCompActive) {
                                      onSelectMask(layer.id, nextComps[0]?.id ?? null);
                                    }
                                  }}
                                  className="text-ink-telemetry hover:text-rose-400 cursor-pointer"
                                >
                                  <span className="material-symbols-outlined text-[11px]">
                                    close
                                  </span>
                                </button>
                              )}
                            </div>
                          );
                        })}

                        <div className="grid grid-cols-3 gap-1 pt-0.5">
                          {(['add', 'subtract', 'intersect'] as const).map((op) => (
                            <button
                              key={op}
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setComposeMenuOp((prev) => (prev === op ? null : op));
                              }}
                              className={`py-0.5 rounded text-[9.5px] font-display border transition-colors cursor-pointer ${composeMenuOp === op
                                  ? 'bg-amber-core/20 border-amber-core/60 text-amber-soft font-semibold'
                                  : 'bg-canvas border-hairline text-ink-secondary hover:text-ink-primary'
                                }`}
                            >
                              {op === 'add' ? '+ Add' : op === 'subtract' ? '− Sub' : '∩ Int'}
                            </button>
                          ))}
                        </div>

                        {composeMenuOp && (
                          <div
                            onClick={(e) => e.stopPropagation()}
                            className="p-1.5 rounded bg-canvas border border-amber-core/40 grid grid-cols-2 gap-1"
                          >
                            {COMPOSE_TOOLS.map((t) => (
                              <button
                                key={t.type}
                                type="button"
                                onClick={() => {
                                  onAttachManualComponent(layer.id, t.type, composeMenuOp);
                                  setComposeMenuOp(null);
                                }}
                                className="py-1 px-1.5 rounded bg-surface hover:bg-surface-active text-[9.5px] text-ink-primary text-left cursor-pointer"
                              >
                                {t.label}
                              </button>
                            ))}
                            <button
                              type="button"
                              disabled={busyAiKind !== null}
                              onClick={() => {
                                onAttachAiComponent(layer.id, 'subject', composeMenuOp);
                                setComposeMenuOp(null);
                              }}
                              className="py-1 px-1.5 rounded bg-surface hover:bg-surface-active text-[9.5px] text-amber-soft text-left cursor-pointer disabled:opacity-50"
                            >
                              AI Subject
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}

            {/* Base Backdrop Layer Row (sits beneath adjustment masks visually & in the render pipeline) */}
            {bgActive && (
              <div
                className={
                  masks.length > 0 ? 'pt-1.5 mt-0.5 border-t border-hairline/70' : undefined
                }
              >
                <div className="px-2 py-1.5 rounded-lg bg-surface border border-emerald-focus/40 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="w-6 h-6 rounded border border-white/25 shrink-0"
                      style={
                        backgroundRemoval.mode === 'transparent'
                          ? {
                            backgroundImage:
                              'linear-gradient(45deg, #555 25%, transparent 25%), linear-gradient(-45deg, #555 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #555 75%), linear-gradient(-45deg, transparent 75%, #555 75%)',
                            backgroundSize: '8px 8px',
                            backgroundPosition: '0 0, 0 4px, 4px -4px, -4px 0',
                            backgroundColor: '#2a2a2e',
                          }
                          : backgroundRemoval.mode === 'gradient'
                            ? {
                              background: backgroundRemovalPreviewCss(backgroundRemoval),
                            }
                            : {
                              backgroundColor: `rgb(${backgroundRemoval.colorRgb[0]}, ${backgroundRemoval.colorRgb[1]}, ${backgroundRemoval.colorRgb[2]})`,
                            }
                      }
                    />
                    <div className="min-w-0">
                      <div className="text-[11px] font-display font-medium text-ink-primary truncate">
                        {backgroundRemoval.mode === 'transparent'
                          ? 'Cutout (PNG)'
                          : backgroundRemoval.mode === 'gradient'
                            ? 'Gradient Backdrop'
                            : 'Studio Backdrop'}
                      </div>
                      <div className="tabular-nums-mono text-[9px] text-emerald-bright">
                        Base Layer
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={onToggleBackgroundRemoval}
                    title="Restore Original Background"
                    className="text-ink-telemetry hover:text-rose-400 cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-[14px]">visibility_off</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Footer: Lightroom Show Overlay (O) */}
          <div className="px-3 py-1.5 bg-surface/80 border-t border-hairline flex items-center justify-between gap-2 shrink-0">
            <label className="flex items-center gap-1.5 text-[10.5px] text-ink-secondary cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showMaskOverlay}
                onChange={onToggleMaskOverlay}
                className="accent-rose-500 rounded cursor-pointer"
              />
              <span>Show Overlay</span>
            </label>
            <kbd className="tabular-nums-mono text-[9.5px] text-ink-telemetry px-1 rounded bg-canvas border border-hairline">
              O
            </kbd>
          </div>
        </>
      )}
    </aside>
  );
});
