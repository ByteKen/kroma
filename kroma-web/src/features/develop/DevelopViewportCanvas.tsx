import { memo, useCallback, useEffect, useState } from 'react';
import { detectWebGpuAvailable } from '../../domain/wgslUniformPacker';
import { WebGlDevelopCanvas } from './WebGlDevelopCanvas';
import {
  WebGpuDevelopCanvas,
  type DevelopViewportCanvasProps,
} from './WebGpuDevelopCanvas';

export type ActiveViewportBackend = 'webgpu' | 'webgl2';

interface UnifiedDevelopViewportCanvasProps extends DevelopViewportCanvasProps {
  /** When `false` (e.g. user selected Performance → CPU Only), skips WebGPU and uses WebGL2 fallback. */
  readonly preferWebGpu?: boolean;
  readonly onBackendChange?: (backend: ActiveViewportBackend) => void;
}

/**
 * Unified Studio Develop viewport that automatically renders through Kroma's 32-bit wide-gamut
 * WebGPU WGSL compute engine when supported (`navigator.gpu` in WebView2 / WKWebView / Chromium),
 * and seamlessly falls back to the Rec.2020-matched WebGL2 renderer if WebGPU is unavailable.
 */
export const DevelopViewportCanvas = memo(function DevelopViewportCanvas({
  preferWebGpu = true,
  onBackendChange,
  ...canvasProps
}: UnifiedDevelopViewportCanvasProps) {
  const [webGpuFailed, setWebGpuFailed] = useState(false);
  const useWebGpu = preferWebGpu && !webGpuFailed && detectWebGpuAvailable();
  const activeBackend: ActiveViewportBackend = useWebGpu ? 'webgpu' : 'webgl2';

  useEffect(() => {
    onBackendChange?.(activeBackend);
  }, [activeBackend, onBackendChange]);

  const handleUnavailable = useCallback(() => {
    setWebGpuFailed(true);
  }, []);

  if (useWebGpu) {
    return <WebGpuDevelopCanvas {...canvasProps} onUnavailable={handleUnavailable} />;
  }

  return <WebGlDevelopCanvas {...canvasProps} />;
});
