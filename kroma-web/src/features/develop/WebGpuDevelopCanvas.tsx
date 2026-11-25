import { memo, useEffect, useRef, useState } from 'react';
import gradePassWgsl from '../../../../shaders/develop/grade_pass.wgsl?raw';
import healPassWgsl from '../../../../shaders/develop/heal_pass.wgsl?raw';
import inpaintPassWgsl from '../../../../shaders/develop/inpaint_pass.wgsl?raw';
import maskPassWgsl from '../../../../shaders/develop/mask_pass.wgsl?raw';
import portraitRetouchPassWgsl from '../../../../shaders/develop/portrait_retouch_pass.wgsl?raw';
import presentWgsl from '../../../../shaders/develop/present.wgsl?raw';
import spatialPassWgsl from '../../../../shaders/develop/spatial_pass.wgsl?raw';
import type { DevelopRecipe } from '../../domain/develop';
import type { YuNetFaceOverlay } from '../../domain/photo';
import {
  compileHealSpotsForGpu,
  decodeInpaintPatchRgba,
  faceOverlayToOrientedGuide,
  hasEffectiveHealSpots,
  isInpaintPatchEffective,
  isPortraitRetouchActive,
  type FaceParseMapData,
  type InpaintPatch,
} from '../../domain/retouch';
import {
  ensureCubeLut3DLoaded,
  getCachedCubeLut3D,
  type ParsedCubeLut3D,
} from '../../domain/cubeLut3d';
import {
  buildToneCurveLutRgba32F,
  compileDevelopShaderUniforms,
  compressRec2020ToLinearSrgb,
  linearToSrgb,
  softClip,
} from '../../domain/developShaderMath';
import {
  computeViewportUvTransform,
  ZERO_VIEWPORT_PAN,
  type ViewportPan,
} from '../../domain/viewportZoom';
import type { DevelopWorkingBuffer } from '../../domain/developWorkingBuffer';
import {
  loadDevelopBitmap,
  peekDevelopBitmap,
  type CachedDevelopBitmap,
} from './developBitmapCache';
import {
  compileMaskLayerForWebGl,
  decodeMatteRleB64,
  isBackgroundRemovalActive,
  isMaskLayerEffective,
  spatialMaskGeometrySignature,
  type BackgroundRemovalSettings,
  type CompiledWebGlMaskData,
  type MaskLayer,
} from '../../domain/masking';
import {
  GRADE_UNIFORM_FLOATS,
  HEAL_UNIFORM_FLOATS,
  INPAINT_UNIFORM_FLOATS,
  MASK_UNIFORM_FLOATS,
  PORTRAIT_RETOUCH_UNIFORM_FLOATS,
  PRESENT_UNIFORM_FLOATS,
  SPATIAL_UNIFORM_FLOATS,
  packGradeUniforms,
  packHealUniforms,
  packInpaintUniforms,
  packMaskUniforms,
  packPortraitRetouchUniforms,
  packPresentUniforms,
  packSpatialUniforms,
} from '../../domain/wgslUniformPacker';

export interface DevelopViewportCanvasProps {
  readonly src: string;
  readonly alt: string;
  readonly recipe: DevelopRecipe;
  readonly faceOverlay?: YuNetFaceOverlay | null;
  readonly faceParseMap?: FaceParseMapData | null;
  readonly workingBuffer?: DevelopWorkingBuffer | null;
  readonly previewMaxDim?: number;
  readonly fit?: 'cover' | 'contain';
  readonly viewportZoom?: number;
  readonly viewportPan?: ViewportPan;
  readonly viewportPadPx?: number;
  readonly rotationDeg?: number;
  readonly rotationScale?: number;
  readonly applyCrop?: boolean;
  readonly showShadowClip?: boolean;
  readonly showHighlightClip?: boolean;
  readonly eyedropperActive?: boolean;
  readonly onSampleNeutralRgb?: (r: number, g: number, b: number) => void;
  readonly activeMaskId?: string | null;
  readonly showMaskOverlay?: boolean;
  readonly colorPickActive?: boolean;
  readonly onSampleColorRangeRgb?: (r: number, g: number, b: number) => void;
  readonly className?: string;
  readonly onNaturalSize?: (width: number, height: number) => void;
  readonly onMaxTextureDim?: (maxDim: number) => void;
}

interface WebGpuDevelopCanvasProps extends DevelopViewportCanvasProps {
  readonly onUnavailable?: () => void;
}

const DEFAULT_MAX_WORK_DIM = 4096;
/** Upload raw working textures in 512-row strips (~8-16 MB each) so Chromium/Dawn's GPU IPC ring buffer never overflows. */
const UPLOAD_CHUNK_ROWS = 512;

function writeTextureRowsChunked(
  queue: GPUQueue,
  tex: GPUTexture,
  data: Uint16Array | Float32Array,
  width: number,
  height: number,
  bytesPerPixel: 8 | 16,
): void {
  const elementsPerRow = width * 4;
  const bytesPerRow = width * bytesPerPixel;
  for (let y = 0; y < height; y += UPLOAD_CHUNK_ROWS) {
    const rows = Math.min(UPLOAD_CHUNK_ROWS, height - y);
    const startElem = y * elementsPerRow;
    const endElem = (y + rows) * elementsPerRow;
    const chunk = data.subarray(startElem, endElem);
    queue.writeTexture(
      { texture: tex, origin: [0, y, 0] },
      chunk as Uint16Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
      { bytesPerRow, rowsPerImage: rows },
      [width, rows],
    );
  }
}

function f16BitsToF32(h: number): number {
  const sign = (h & 0x8000) !== 0 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 31) return frac === 0 ? sign * 65504 : 0;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function orientSourceUvJs(
  u: number,
  v: number,
  flipH: boolean,
  flipV: boolean,
  quarterTurns: number,
): [number, number] {
  const px = flipH ? 1 - u : u;
  const py = flipV ? 1 - v : v;
  const qt = ((quarterTurns % 4) + 4) % 4;
  if (qt === 1) return [py, 1 - px];
  if (qt === 2) return [1 - px, 1 - py];
  if (qt === 3) return [1 - py, px];
  return [px, py];
}

interface SharedWebGpuCore {
  readonly device: GPUDevice;
  readonly queue: GPUQueue;
  readonly swapFormat: GPUTextureFormat;
  readonly maxTextureDim: number;
  readonly gradeBgl: GPUBindGroupLayout;
  readonly gradePipeline: GPUComputePipeline;
  readonly inpaintBgl: GPUBindGroupLayout;
  readonly inpaintPipeline: GPUComputePipeline;
  readonly healBgl: GPUBindGroupLayout;
  readonly healPipeline: GPUComputePipeline;
  readonly portraitBgl: GPUBindGroupLayout;
  readonly portraitPipeline: GPUComputePipeline;
  readonly maskBgl: GPUBindGroupLayout;
  readonly maskPipeline: GPUComputePipeline;
  readonly spatialBgl: GPUBindGroupLayout;
  readonly spatialPipeline: GPUComputePipeline;
  readonly presentBgl: GPUBindGroupLayout;
  readonly presentPipeline: GPURenderPipeline;
  readonly dummyLut3dBuf: GPUBuffer;
  readonly dummyWhiteTex: GPUTexture;
  readonly dummyWhiteView: GPUTextureView;
  readonly dummyTransparentTex: GPUTexture;
  readonly dummyTransparentView: GPUTextureView;
  readonly dummyToneLutTex: GPUTexture;
  readonly dummyToneLutView: GPUTextureView;
  readonly lostListeners: Set<() => void>;
  isLost: boolean;
}

let sharedCorePromise: Promise<SharedWebGpuCore | null> | null = null;

function createSharedWebGpuCore(): Promise<SharedWebGpuCore | null> {
  if (sharedCorePromise) {
    return sharedCorePromise;
  }
  sharedCorePromise = (async () => {
    try {
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        return null;
      }
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        return null;
      }
      const device = await adapter.requestDevice();
      const lostListeners = new Set<() => void>();
      let coreRef: SharedWebGpuCore | null = null;
      device.lost.then(() => {
        sharedCorePromise = null;
        if (coreRef) {
          coreRef.isLost = true;
        }
        for (const listener of [...lostListeners]) {
          listener();
        }
        lostListeners.clear();
      });

      const swapFormat = navigator.gpu.getPreferredCanvasFormat();
      const maxTextureDim = device.limits.maxTextureDimension2D || 8192;

      const gradeModule = device.createShaderModule({ label: 'kroma-grade-wgsl', code: gradePassWgsl });
      const inpaintModule = device.createShaderModule({
        label: 'kroma-inpaint-wgsl',
        code: inpaintPassWgsl,
      });
      const healModule = device.createShaderModule({ label: 'kroma-heal-wgsl', code: healPassWgsl });
      const portraitModule = device.createShaderModule({
        label: 'kroma-portrait-retouch-wgsl',
        code: portraitRetouchPassWgsl,
      });
      const maskModule = device.createShaderModule({ label: 'kroma-mask-wgsl', code: maskPassWgsl });
      const spatialModule = device.createShaderModule({
        label: 'kroma-spatial-wgsl',
        code: spatialPassWgsl,
      });
      const presentModule = device.createShaderModule({
        label: 'kroma-present-wgsl',
        code: presentWgsl,
      });

      const gradeBgl = device.createBindGroupLayout({
        label: 'kroma-grade-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          {
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba32float' },
          },
        ],
      });
      const gradePipeline = await device.createComputePipelineAsync({
        label: 'kroma-grade-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [gradeBgl] }),
        compute: { module: gradeModule, entryPoint: 'main' },
      });

      const inpaintBgl = device.createBindGroupLayout({
        label: 'kroma-inpaint-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba32float' },
          },
        ],
      });
      const inpaintPipeline = await device.createComputePipelineAsync({
        label: 'kroma-inpaint-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [inpaintBgl] }),
        compute: { module: inpaintModule, entryPoint: 'main' },
      });

      const healBgl = device.createBindGroupLayout({
        label: 'kroma-heal-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba32float' },
          },
        ],
      });
      const healPipeline = await device.createComputePipelineAsync({
        label: 'kroma-heal-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [healBgl] }),
        compute: { module: healModule, entryPoint: 'main' },
      });

      const portraitBgl = device.createBindGroupLayout({
        label: 'kroma-portrait-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba32float' },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
        ],
      });
      const portraitPipeline = await device.createComputePipelineAsync({
        label: 'kroma-portrait-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [portraitBgl] }),
        compute: { module: portraitModule, entryPoint: 'main' },
      });

      const maskBgl = device.createBindGroupLayout({
        label: 'kroma-mask-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba32float' },
          },
        ],
      });
      const maskPipeline = await device.createComputePipelineAsync({
        label: 'kroma-mask-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [maskBgl] }),
        compute: { module: maskModule, entryPoint: 'main' },
      });

      const spatialBgl = device.createBindGroupLayout({
        label: 'kroma-spatial-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            texture: { sampleType: 'unfilterable-float' },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba32float' },
          },
        ],
      });
      const spatialPipeline = await device.createComputePipelineAsync({
        label: 'kroma-spatial-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [spatialBgl] }),
        compute: { module: spatialModule, entryPoint: 'main' },
      });

      const presentBgl = device.createBindGroupLayout({
        label: 'kroma-present-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'unfilterable-float' },
          },
        ],
      });
      const presentPipeline = await device.createRenderPipelineAsync({
        label: 'kroma-present-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [presentBgl] }),
        vertex: { module: presentModule, entryPoint: 'vs_main' },
        fragment: {
          module: presentModule,
          entryPoint: 'fs_main',
          targets: [{ format: swapFormat }],
        },
        primitive: { topology: 'triangle-list' },
      });

      const dummyLut3dBuf = device.createBuffer({
        label: 'kroma-dummy-lut3d',
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(dummyLut3dBuf, 0, new Float32Array([0, 0, 0, 0]));

      const dummyWhiteTex = device.createTexture({
        label: 'kroma-dummy-white',
        size: [1, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: dummyWhiteTex },
        new Uint8Array([255, 255, 255, 255]),
        { bytesPerRow: 4, rowsPerImage: 1 },
        [1, 1],
      );

      const dummyTransparentTex = device.createTexture({
        label: 'kroma-dummy-transparent',
        size: [1, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: dummyTransparentTex },
        new Uint8Array([0, 0, 0, 0]),
        { bytesPerRow: 4, rowsPerImage: 1 },
        [1, 1],
      );

      const identityLut = new Float32Array(256 * 4);
      for (let i = 0; i < 256; i++) {
        const v = i / 255;
        identityLut[i * 4] = v;
        identityLut[i * 4 + 1] = v;
        identityLut[i * 4 + 2] = v;
        identityLut[i * 4 + 3] = 1.0;
      }
      const dummyToneLutTex = device.createTexture({
        label: 'kroma-dummy-tone-lut',
        size: [256, 1],
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: dummyToneLutTex },
        identityLut,
        { bytesPerRow: 256 * 16, rowsPerImage: 1 },
        [256, 1],
      );

      const core: SharedWebGpuCore = {
        device,
        queue: device.queue,
        swapFormat,
        maxTextureDim,
        gradeBgl,
        gradePipeline,
        inpaintBgl,
        inpaintPipeline,
        healBgl,
        healPipeline,
        portraitBgl,
        portraitPipeline,
        maskBgl,
        maskPipeline,
        spatialBgl,
        spatialPipeline,
        presentBgl,
        presentPipeline,
        dummyLut3dBuf,
        dummyWhiteTex,
        dummyWhiteView: dummyWhiteTex.createView(),
        dummyTransparentTex,
        dummyTransparentView: dummyTransparentTex.createView(),
        dummyToneLutTex,
        dummyToneLutView: dummyToneLutTex.createView(),
        lostListeners,
        isLost: false,
      };
      coreRef = core;
      return core;
    } catch {
      return null;
    }
  })();

  return sharedCorePromise;
}

interface CachedSpatialMaskGpu {
  readonly sig: string;
  readonly tex: GPUTexture;
  readonly compiled: CompiledWebGlMaskData;
}

interface CachedInpaintPatchGpu {
  readonly sig: string;
  status: 'loading' | 'ready' | 'error';
  tex?: GPUTexture;
  view?: GPUTextureView;
  w: number;
  h: number;
}

interface CachedGradeBindGroup {
  readonly srcView: GPUTextureView;
  readonly toneView: GPUTextureView;
  readonly bgMatteView: GPUTextureView;
  readonly lutBuf: GPUBuffer;
  readonly outView: GPUTextureView;
  readonly bg: GPUBindGroup;
}

interface CanvasGpuResources {
  readonly core: SharedWebGpuCore;
  readonly ctx: GPUCanvasContext;
  readonly gradeUbo: GPUBuffer;
  readonly healUbo: GPUBuffer;
  readonly portraitUbo: GPUBuffer;
  readonly spatialUbo: GPUBuffer;
  readonly presentUbo: GPUBuffer;
  readonly toneLutTex: GPUTexture;
  readonly toneLutView: GPUTextureView;
  srcTex: GPUTexture | null;
  srcView: GPUTextureView | null;
  workTexA: GPUTexture | null;
  workViewA: GPUTextureView | null;
  workTexB: GPUTexture | null;
  workViewB: GPUTextureView | null;
  workW: number;
  workH: number;
  lastFinalTex: GPUTexture | null;
  bgMatteTex: GPUTexture | null;
  bgMatteView: GPUTextureView | null;
  bgMatteSig: string;
  lut3dBuf: GPUBuffer | null;
  lut3dPath: string;
  lut3dParsed: ParsedCubeLut3D | null;
  healSpotsBuf: GPUBuffer | null;
  healSpotsCap: number;
  healSegsBuf: GPUBuffer | null;
  healSegsCap: number;
  faceParseTex: GPUTexture | null;
  faceParseView: GPUTextureView | null;
  faceParseRef: FaceParseMapData | null;
  readonly inpaintPatchCache: Map<string, CachedInpaintPatchGpu>;
  inpaintUboPool: GPUBuffer[];
  gradeBgCache: CachedGradeBindGroup | null;
  presentBgA: GPUBindGroup | null;
  presentBgB: GPUBindGroup | null;
  readonly spatialMaskCache: Map<string, CachedSpatialMaskGpu>;
  maskUboPool: GPUBuffer[];
}

function ensureWorkTextures(
  res: CanvasGpuResources,
  w: number,
  h: number,
  needPingPong: boolean,
): void {
  const maxDim = Math.min(DEFAULT_MAX_WORK_DIM, res.core.maxTextureDim);
  const clampedW = Math.max(1, Math.min(maxDim, w));
  const clampedH = Math.max(1, Math.min(maxDim, h));
  const sizeChanged = res.workW !== clampedW || res.workH !== clampedH;
  const desc: GPUTextureDescriptor = {
    size: [clampedW, clampedH],
    format: 'rgba32float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  };

  if (sizeChanged || !res.workTexA || !res.workViewA) {
    res.workTexA?.destroy();
    res.workTexB?.destroy();
    res.workTexB = null;
    res.workViewB = null;
    res.workTexA = res.core.device.createTexture({ ...desc, label: 'kroma-studio-work-a' });
    res.workViewA = res.workTexA.createView();
    res.workW = clampedW;
    res.workH = clampedH;
    res.lastFinalTex = null;
    res.gradeBgCache = null;
    res.presentBgA = null;
    res.presentBgB = null;
  }

  if (needPingPong && (!res.workTexB || !res.workViewB)) {
    res.workTexB = res.core.device.createTexture({ ...desc, label: 'kroma-studio-work-b' });
    res.workViewB = res.workTexB.createView();
    res.presentBgB = null;
  } else if (!needPingPong && res.workTexB && clampedW * clampedH > 2560 * 2560) {
    // Release high-res ping-pong VRAM immediately when masks/spatial are not active
    res.workTexB.destroy();
    res.workTexB = null;
    res.workViewB = null;
    res.presentBgB = null;
  }
}

function ensureBgMatteTextureGpu(
  res: CanvasGpuResources,
  bg: BackgroundRemovalSettings | undefined,
): boolean {
  if (!bg || !isBackgroundRemovalActive(bg)) return false;
  const sig = `${bg.matteWidth}x${bg.matteHeight}:${bg.matteRleB64}`;
  if (res.bgMatteSig === sig && res.bgMatteTex && res.bgMatteView) return true;
  const decoded = decodeMatteRleB64(bg.matteRleB64, bg.matteWidth * bg.matteHeight);
  if (!decoded) return false;

  const rgba = new Uint8Array(decoded.length * 4);
  for (let i = 0; i < decoded.length; i++) {
    const v = decoded[i]!;
    const o = i * 4;
    rgba[o] = v;
    rgba[o + 1] = v;
    rgba[o + 2] = v;
    rgba[o + 3] = 255;
  }

  res.bgMatteTex?.destroy();
  const tex = res.core.device.createTexture({
    label: 'kroma-bg-matte',
    size: [bg.matteWidth, bg.matteHeight],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  res.core.device.queue.writeTexture(
    { texture: tex },
    rgba,
    { bytesPerRow: bg.matteWidth * 4, rowsPerImage: bg.matteHeight },
    [bg.matteWidth, bg.matteHeight],
  );
  res.bgMatteTex = tex;
  res.bgMatteView = tex.createView();
  res.bgMatteSig = sig;
  res.gradeBgCache = null;
  return true;
}

function ensureLut3dBufferGpu(
  res: CanvasGpuResources,
  cubePath: string,
  parsed: ParsedCubeLut3D,
): GPUBuffer {
  if (res.lut3dBuf && res.lut3dPath === cubePath && res.lut3dParsed === parsed) {
    return res.lut3dBuf;
  }
  res.lut3dBuf?.destroy();
  const buf = res.core.device.createBuffer({
    label: `kroma-lut3d-${cubePath}`,
    size: parsed.rgbaTable.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  res.core.queue.writeBuffer(buf, 0, parsed.rgbaTable as Float32Array<ArrayBuffer>);
  res.lut3dBuf = buf;
  res.lut3dPath = cubePath;
  res.lut3dParsed = parsed;
  return buf;
}

function getOrUploadSpatialMaskGpu(
  res: CanvasGpuResources,
  layer: MaskLayer,
  orientedW: number,
  orientedH: number,
): CachedSpatialMaskGpu {
  const sig = spatialMaskGeometrySignature(layer, orientedW, orientedH);
  const cached = res.spatialMaskCache.get(layer.id);
  if (cached && cached.sig === sig) {
    const rangeOnly = compileMaskLayerForWebGl(
      {
        ...layer,
        components: layer.components.filter(
          (c) => c.type === 'luminance_range' || c.type === 'color_range',
        ),
      },
      orientedW,
      orientedH,
      16,
    );
    return {
      sig,
      tex: cached.tex,
      compiled: { ...cached.compiled, ranges: rangeOnly.ranges },
    };
  }

  const hasAiMatte = layer.components.some((c) => c.type === 'ai_matte');
  const compiled = compileMaskLayerForWebGl(layer, orientedW, orientedH, hasAiMatte ? 512 : 320);

  cached?.tex.destroy();
  const tex = res.core.device.createTexture({
    label: `kroma-spatial-mask-${layer.id}`,
    size: [compiled.spatialWidth, compiled.spatialHeight],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  res.core.device.queue.writeTexture(
    { texture: tex },
    compiled.spatialRgba as Uint8Array<ArrayBuffer>,
    { bytesPerRow: compiled.spatialWidth * 4, rowsPerImage: compiled.spatialHeight },
    [compiled.spatialWidth, compiled.spatialHeight],
  );

  const entry: CachedSpatialMaskGpu = { sig, tex, compiled };
  res.spatialMaskCache.set(layer.id, entry);
  return entry;
}

function ensureFaceParseTextureGpu(
  res: CanvasGpuResources,
  faceParseMap: FaceParseMapData | null | undefined,
): boolean {
  if (!faceParseMap || faceParseMap.width <= 0 || faceParseMap.height <= 0) {
    return false;
  }
  if (res.faceParseRef === faceParseMap && res.faceParseTex && res.faceParseView) {
    return true;
  }
  const { width, height, rgba } = faceParseMap;
  if (rgba.length !== width * height * 4) return false;

  res.faceParseTex?.destroy();
  const tex = res.core.device.createTexture({
    label: 'kroma-face-parse-rgba',
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  res.core.queue.writeTexture(
    { texture: tex },
    rgba as Uint8Array<ArrayBuffer>,
    { bytesPerRow: width * 4, rowsPerImage: height },
    [width, height],
  );
  res.faceParseTex = tex;
  res.faceParseView = tex.createView();
  res.faceParseRef = faceParseMap;
  return true;
}

function getOrLoadInpaintPatchGpu(
  res: CanvasGpuResources,
  patch: InpaintPatch,
  isCurrentRes: () => boolean,
  onDecoded: () => void,
): CachedInpaintPatchGpu | null {
  const sig = `${patch.id}:${patch.patchWidth}x${patch.patchHeight}:${patch.patchImageB64.length}:${patch.maskRleB64.length}`;
  const existing = res.inpaintPatchCache.get(patch.id);
  if (existing && existing.sig === sig) {
    return existing.status === 'ready' ? existing : null;
  }

  existing?.tex?.destroy();
  const pending: CachedInpaintPatchGpu = {
    sig,
    status: 'loading',
    w: patch.patchWidth,
    h: patch.patchHeight,
  };
  res.inpaintPatchCache.set(patch.id, pending);

  void decodeInpaintPatchRgba(patch).then((decoded) => {
    if (!isCurrentRes() || res.core.isLost) return;
    const latest = res.inpaintPatchCache.get(patch.id);
    if (!latest || latest.sig !== sig) return;
    if (!decoded) {
      latest.status = 'error';
      return;
    }
    const tex = res.core.device.createTexture({
      label: `kroma-inpaint-patch-${patch.id}`,
      size: [decoded.width, decoded.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    res.core.queue.writeTexture(
      { texture: tex },
      decoded.rgba as Uint8Array<ArrayBuffer>,
      { bytesPerRow: decoded.width * 4, rowsPerImage: decoded.height },
      [decoded.width, decoded.height],
    );
    latest.tex = tex;
    latest.view = tex.createView();
    latest.w = decoded.width;
    latest.h = decoded.height;
    latest.status = 'ready';
    onDecoded();
  });

  return null;
}

/**
 * Unified 32-Bit Scene-Referred Wide-Gamut (Linear Rec.2020) WebGPU Develop Viewport.
 * Executes the EXACT same WGSL compute shaders (`shaders/develop/*.wgsl`) as Kroma's Rust `wgpu`
 * DX12/Metal export engine at 60–120 FPS with full `rgba32float` precision.
 */
export const WebGpuDevelopCanvas = memo(function WebGpuDevelopCanvas({
  src,
  alt,
  recipe,
  faceOverlay = null,
  faceParseMap = null,
  workingBuffer = null,
  previewMaxDim = 2048,
  fit = 'cover',
  viewportZoom = 1,
  viewportPan = ZERO_VIEWPORT_PAN,
  viewportPadPx = 0,
  applyCrop = true,
  showShadowClip = false,
  showHighlightClip = false,
  eyedropperActive = false,
  onSampleNeutralRgb,
  activeMaskId = null,
  showMaskOverlay = false,
  colorPickActive = false,
  onSampleColorRangeRgb,
  className = '',
  onNaturalSize,
  onMaxTextureDim,
  onUnavailable,
}: WebGpuDevelopCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resRef = useRef<CanvasGpuResources | null>(null);
  const imgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const sourceBitmapRef = useRef<CachedDevelopBitmap | null>(null);
  const loadedSrcRef = useRef<string>('');
  const inputLinearRef = useRef(false);
  const inputWideGamutRef = useRef(false);
  const textureReadyRef = useRef(false);
  const gradeDirtyRef = useRef(true);
  const lastFinalIsARef = useRef(true);
  const lastBgTransparentRef = useRef(false);
  const workingBufferRef = useRef<DevelopWorkingBuffer | null>(workingBuffer);
  const recipeRef = useRef(recipe);
  const faceOverlayRef = useRef<YuNetFaceOverlay | null>(faceOverlay);
  const faceParseMapRef = useRef<FaceParseMapData | null>(faceParseMap);
  const fitRef = useRef(fit);
  const viewportZoomRef = useRef(viewportZoom);
  const viewportPanRef = useRef(viewportPan);
  const viewportPadPxRef = useRef(viewportPadPx);
  const applyCropRef = useRef(applyCrop);
  const clipFlagsRef = useRef({ shadow: showShadowClip, highlight: showHighlightClip });
  const maskOverlayRef = useRef({ activeMaskId, showMaskOverlay });
  const onNaturalSizeRef = useRef(onNaturalSize);
  const onMaxTextureDimRef = useRef(onMaxTextureDim);
  const onUnavailableRef = useRef(onUnavailable);

  workingBufferRef.current = workingBuffer;
  recipeRef.current = recipe;
  faceOverlayRef.current = faceOverlay;
  faceParseMapRef.current = faceParseMap;
  fitRef.current = fit;
  viewportZoomRef.current = viewportZoom;
  viewportPanRef.current = viewportPan;
  viewportPadPxRef.current = viewportPadPx;
  applyCropRef.current = applyCrop;
  clipFlagsRef.current = { shadow: showShadowClip, highlight: showHighlightClip };
  maskOverlayRef.current = { activeMaskId, showMaskOverlay };
  onNaturalSizeRef.current = onNaturalSize;
  onMaxTextureDimRef.current = onMaxTextureDim;
  onUnavailableRef.current = onUnavailable;

  const [gpuReady, setGpuReady] = useState(false);
  const [textureVersion, setTextureVersion] = useState(0);
  const renderFrameRef = useRef<() => void>(() => {});
  const pendingRafRef = useRef<number | null>(null);

  const scheduleRender = () => {
    if (pendingRafRef.current !== null) return;
    pendingRafRef.current = window.requestAnimationFrame(() => {
      pendingRafRef.current = null;
      renderFrameRef.current();
    });
  };

  const renderFrame = () => {
    const res = resRef.current;
    const canvas = canvasRef.current;
    if (!res || !canvas || !textureReadyRef.current || !res.srcTex || !res.srcView) return;
    if (res.core.isLost) {
      onUnavailableRef.current?.();
      return;
    }
    // Skip GPU compute dispatches & swapchain resizes while Studio is hidden (`display: none`)
    if (canvas.clientWidth === 0 || canvas.clientHeight === 0) return;

    const { core, ctx } = res;
    const { device, queue } = core;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    const pixelW = Math.max(1, Math.round(cssW * dpr));
    const pixelH = Math.max(1, Math.round(cssH * dpr));

    if (canvas.width !== pixelW || canvas.height !== pixelH) {
      canvas.width = pixelW;
      canvas.height = pixelH;
    }

    const curRecipe = recipeRef.current;
    const crop = curRecipe.cropTransform;
    const qt = (((crop.quarterTurns ?? 0) % 4) + 4) % 4;
    const swapAxes = qt % 2 === 1;
    const orientedW = swapAxes ? imgSizeRef.current.h : imgSizeRef.current.w;
    const orientedH = swapAxes ? imgSizeRef.current.w : imgSizeRef.current.h;
    if (orientedW <= 0 || orientedH <= 0) return;

    const maxWorkDim = Math.min(DEFAULT_MAX_WORK_DIM, core.maxTextureDim);
    const scaleDown =
      Math.max(orientedW, orientedH) > maxWorkDim
        ? maxWorkDim / Math.max(orientedW, orientedH)
        : 1;
    const workW = Math.max(1, Math.round(orientedW * scaleDown));
    const workH = Math.max(1, Math.round(orientedH * scaleDown));

    const uv = computeViewportUvTransform(
      orientedW,
      orientedH,
      cssW,
      cssH,
      fitRef.current,
      applyCropRef.current ? crop.rect : undefined,
      viewportPadPxRef.current,
      viewportZoomRef.current,
      viewportPanRef.current,
    );

    const canPresentOnly =
      !gradeDirtyRef.current &&
      res.lastFinalTex !== null &&
      res.workW === workW &&
      res.workH === workH &&
      Boolean(lastFinalIsARef.current ? res.workViewA : res.workViewB);

    const encoder = device.createCommandEncoder({
      label: canPresentOnly ? 'kroma-studio-present-only' : 'kroma-studio-frame',
    });

    let currentIsA = lastFinalIsARef.current;
    let bgTransparent = lastBgTransparentRef.current;

    if (!canPresentOnly) {
      const u = compileDevelopShaderUniforms(curRecipe);
      const allMasks = curRecipe.masks ?? [];
      const effectiveMasks = allMasks.filter(isMaskLayerEffective);
      const overlayMask = maskOverlayRef.current.showMaskOverlay
        ? (allMasks.find(
            (m) =>
              m.id === maskOverlayRef.current.activeMaskId &&
              m.enabled &&
              m.components.length > 0,
          ) ?? null)
        : null;

      if (res.spatialMaskCache.size > 0) {
        const liveIds = new Set(allMasks.map((m) => m.id));
        for (const [id, entry] of res.spatialMaskCache.entries()) {
          if (!liveIds.has(id)) {
            entry.tex.destroy();
            res.spatialMaskCache.delete(id);
          }
        }
      }

      const bgReady = ensureBgMatteTextureGpu(res, curRecipe.backgroundRemoval);
      const bgCfg = curRecipe.backgroundRemoval;
      bgTransparent = Boolean(bgReady && bgCfg?.mode === 'transparent');

      const effectiveInpaint = (curRecipe.retouch?.inpaintPatches ?? []).filter(
        isInpaintPatchEffective,
      );
      if (res.inpaintPatchCache.size > 0) {
        const livePatchIds = new Set(effectiveInpaint.map((p) => p.id));
        for (const [id, entry] of res.inpaintPatchCache.entries()) {
          if (!livePatchIds.has(id)) {
            entry.tex?.destroy();
            res.inpaintPatchCache.delete(id);
          }
        }
      }

      const needInpaint = effectiveInpaint.length > 0;
      const needHeal = hasEffectiveHealSpots(curRecipe.retouch);
      const needPortrait = isPortraitRetouchActive(curRecipe.retouch?.portrait);

      const needPingPong =
        needInpaint ||
        needHeal ||
        needPortrait ||
        effectiveMasks.length > 0 ||
        u.spatialStageActive ||
        bgTransparent ||
        overlayMask !== null;
      ensureWorkTextures(res, workW, workH, needPingPong);
      if (!res.workTexA || !res.workViewA) return;
      if (needPingPong && (!res.workTexB || !res.workViewB)) return;

      if (u.toneCurveActive) {
        const lutFloats = buildToneCurveLutRgba32F(curRecipe.toneCurve);
        queue.writeTexture(
          { texture: res.toneLutTex },
          lutFloats as Float32Array<ArrayBuffer>,
          { bytesPerRow: 256 * 16, rowsPerImage: 1 },
          [256, 1],
        );
      }

      let activeLutBuf = core.dummyLut3dBuf;
      let activeLutMeta: {
        readonly size: number;
        readonly intensity: number;
        readonly domainMin: readonly [number, number, number];
        readonly domainMax: readonly [number, number, number];
      } | null = null;

      if (curRecipe.lut3d && curRecipe.lut3d.intensity > 0) {
        const cubePath = curRecipe.lut3d.cubePath;
        const cachedLut = getCachedCubeLut3D(cubePath);
        if (cachedLut) {
          activeLutBuf = ensureLut3dBufferGpu(res, cubePath, cachedLut);
          activeLutMeta = {
            size: cachedLut.size,
            intensity: curRecipe.lut3d.intensity / 100,
            domainMin: cachedLut.domainMin,
            domainMax: cachedLut.domainMax,
          };
        } else {
          void ensureCubeLut3DLoaded(cubePath).then((loaded) => {
            if (loaded && resRef.current) {
              gradeDirtyRef.current = true;
              scheduleRender();
            }
          });
        }
      }

      const gradeData = packGradeUniforms({
        outWidth: workW,
        outHeight: workH,
        srcWidth: imgSizeRef.current.w,
        srcHeight: imgSizeRef.current.h,
        inputIsLinear: inputLinearRef.current,
        inputIsWideGamut: inputWideGamutRef.current,
        recipe: curRecipe,
        includeFineRotation: true,
        bgMatteReady: bgReady,
        lut3d: activeLutMeta,
      });
      queue.writeBuffer(res.gradeUbo, 0, gradeData as Float32Array<ArrayBuffer>);

      const srcView = res.srcView;
      const toneView = u.toneCurveActive ? res.toneLutView : core.dummyToneLutView;
      const bgMatteView = bgReady && res.bgMatteView ? res.bgMatteView : core.dummyWhiteView;
      const viewA = res.workViewA;
      const viewB = res.workViewB;

      const wgX = Math.ceil(workW / 8);
      const wgY = Math.ceil(workH / 8);

      // Pass 1: 32-Bit Scene-Referred Wide-Gamut Color/Tone/Backdrop (cached bind group across slider frames)
      let gradeBg: GPUBindGroup;
      const cachedGrade = res.gradeBgCache;
      if (
        cachedGrade &&
        cachedGrade.srcView === srcView &&
        cachedGrade.toneView === toneView &&
        cachedGrade.bgMatteView === bgMatteView &&
        cachedGrade.lutBuf === activeLutBuf &&
        cachedGrade.outView === viewA
      ) {
        gradeBg = cachedGrade.bg;
      } else {
        gradeBg = device.createBindGroup({
          layout: core.gradeBgl,
          entries: [
            { binding: 0, resource: { buffer: res.gradeUbo } },
            { binding: 1, resource: srcView },
            { binding: 2, resource: toneView },
            { binding: 3, resource: bgMatteView },
            { binding: 4, resource: { buffer: activeLutBuf } },
            { binding: 5, resource: viewA },
          ],
        });
        res.gradeBgCache = {
          srcView,
          toneView,
          bgMatteView,
          lutBuf: activeLutBuf,
          outView: viewA,
          bg: gradeBg,
        };
      }
      {
        const pass = encoder.beginComputePass({ label: 'kroma-grade-compute' });
        pass.setPipeline(core.gradePipeline);
        pass.setBindGroup(0, gradeBg);
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
      }

      currentIsA = true;

      // Pass 1a-0: Non-Destructive Phase 3 Generative Inpaint Patches (LaMa / Harmonic Fill)
      if (needInpaint) {
        let inpaintUboIdx = 0;
        for (const patch of effectiveInpaint) {
          const cachedPatch = getOrLoadInpaintPatchGpu(
            res,
            patch,
            () => resRef.current === res,
            () => {
              gradeDirtyRef.current = true;
              scheduleRender();
            },
          );
          if (!cachedPatch?.view) continue;
          let inpaintUbo = res.inpaintUboPool[inpaintUboIdx];
          if (!inpaintUbo) {
            inpaintUbo = device.createBuffer({
              label: `kroma-inpaint-ubo-${inpaintUboIdx}`,
              size: INPAINT_UNIFORM_FLOATS * 4,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            res.inpaintUboPool.push(inpaintUbo);
          }
          inpaintUboIdx += 1;

          const inpaintData = packInpaintUniforms({
            outWidth: workW,
            outHeight: workH,
            patchWidth: cachedPatch.w,
            patchHeight: cachedPatch.h,
            bounds: patch.bounds,
            opacity: patch.opacity,
          });
          queue.writeBuffer(inpaintUbo, 0, inpaintData as Float32Array<ArrayBuffer>);

          const inpaintBg = device.createBindGroup({
            layout: core.inpaintBgl,
            entries: [
              { binding: 0, resource: { buffer: inpaintUbo } },
              { binding: 1, resource: currentIsA ? viewA : viewB! },
              { binding: 2, resource: cachedPatch.view },
              { binding: 3, resource: currentIsA ? viewB! : viewA },
            ],
          });
          const pass = encoder.beginComputePass({ label: 'kroma-inpaint-compute' });
          pass.setPipeline(core.inpaintPipeline);
          pass.setBindGroup(0, inpaintBg);
          pass.dispatchWorkgroups(wgX, wgY, 1);
          pass.end();
          currentIsA = !currentIsA;
        }
      }

      // Pass 1a: Non-Destructive Spot Heal & Clone (runs pre-portrait so healed blemishes never bleed into skin smoothing)
      if (needHeal && curRecipe.retouch) {
        const compiledHeal = compileHealSpotsForGpu(
          curRecipe.retouch.healSpots,
          orientedW,
          orientedH,
        );
        if (compiledHeal.spotCount > 0) {
          const healUboData = packHealUniforms({
            outWidth: workW,
            outHeight: workH,
            oriWidth: orientedW,
            oriHeight: orientedH,
            spotCount: compiledHeal.spotCount,
          });
          queue.writeBuffer(res.healUbo, 0, healUboData as Float32Array<ArrayBuffer>);

          const spotsBytes = Math.max(64, compiledHeal.spotsData.byteLength);
          if (!res.healSpotsBuf || res.healSpotsCap < spotsBytes) {
            res.healSpotsBuf?.destroy();
            res.healSpotsBuf = device.createBuffer({
              label: 'kroma-heal-spots-buf',
              size: spotsBytes,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            res.healSpotsCap = spotsBytes;
          }
          queue.writeBuffer(
            res.healSpotsBuf,
            0,
            compiledHeal.spotsData as Float32Array<ArrayBuffer>,
          );

          const segsBytes = Math.max(16, compiledHeal.segmentsData.byteLength);
          if (!res.healSegsBuf || res.healSegsCap < segsBytes) {
            res.healSegsBuf?.destroy();
            res.healSegsBuf = device.createBuffer({
              label: 'kroma-heal-segs-buf',
              size: segsBytes,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            res.healSegsCap = segsBytes;
          }
          queue.writeBuffer(
            res.healSegsBuf,
            0,
            compiledHeal.segmentsData as Float32Array<ArrayBuffer>,
          );

          const healBg = device.createBindGroup({
            layout: core.healBgl,
            entries: [
              { binding: 0, resource: { buffer: res.healUbo } },
              { binding: 1, resource: currentIsA ? viewA : viewB! },
              { binding: 2, resource: { buffer: res.healSpotsBuf } },
              { binding: 3, resource: { buffer: res.healSegsBuf } },
              { binding: 4, resource: currentIsA ? viewB! : viewA },
            ],
          });
          const pass = encoder.beginComputePass({ label: 'kroma-heal-compute' });
          pass.setPipeline(core.healPipeline);
          pass.setBindGroup(0, healBg);
          pass.dispatchWorkgroups(wgX, wgY, 1);
          pass.end();
          currentIsA = !currentIsA;
        }
      }

      // Pass 1a-2: Frequency-Separation & Landmark-Guided (or BiSeNet Neural) Portrait Retouching
      if (needPortrait && curRecipe.retouch?.portrait) {
        const orientedGuide = faceOverlayToOrientedGuide(
          faceOverlayRef.current,
          crop,
          orientedW,
          orientedH,
          true,
        );
        const faceParseReady = ensureFaceParseTextureGpu(res, faceParseMapRef.current);
        const portraitData = packPortraitRetouchUniforms({
          outWidth: workW,
          outHeight: workH,
          oriWidth: orientedW,
          oriHeight: orientedH,
          portrait: curRecipe.retouch.portrait,
          faceGuides: orientedGuide ? [orientedGuide] : [],
          hasFaceParseMap: faceParseReady,
        });
        queue.writeBuffer(res.portraitUbo, 0, portraitData as Float32Array<ArrayBuffer>);
        const portraitBg = device.createBindGroup({
          layout: core.portraitBgl,
          entries: [
            { binding: 0, resource: { buffer: res.portraitUbo } },
            { binding: 1, resource: currentIsA ? viewA : viewB! },
            { binding: 2, resource: currentIsA ? viewB! : viewA },
            {
              binding: 3,
              resource:
                faceParseReady && res.faceParseView
                  ? res.faceParseView
                  : core.dummyTransparentView,
            },
          ],
        });
        const pass = encoder.beginComputePass({ label: 'kroma-portrait-compute' });
        pass.setPipeline(core.portraitPipeline);
        pass.setBindGroup(0, portraitBg);
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
        currentIsA = !currentIsA;
      }

      let maskUboIdx = 0;
      const acquireMaskUbo = (): GPUBuffer => {
        let buf = res.maskUboPool[maskUboIdx];
        if (!buf) {
          buf = device.createBuffer({
            label: `kroma-mask-ubo-${maskUboIdx}`,
            size: MASK_UNIFORM_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          res.maskUboPool.push(buf);
        }
        maskUboIdx += 1;
        return buf;
      };

      // Pass 1b: Local & AI Masks
      for (const layer of effectiveMasks) {
        const cached = getOrUploadSpatialMaskGpu(res, layer, orientedW, orientedH);
        const adj = layer.adjustments;
        const hasTone =
          adj.temperatureShift !== 0 ||
          adj.tintShift !== 0 ||
          Math.abs(adj.exposureEv) > 1e-4 ||
          adj.contrast !== 0 ||
          adj.highlights !== 0 ||
          adj.shadows !== 0 ||
          adj.whites !== 0 ||
          adj.blacks !== 0 ||
          adj.dehaze !== 0 ||
          adj.saturation !== 0;
        const hasSpatial = adj.texture !== 0 || adj.clarity !== 0 || adj.sharpness !== 0;
        const stages: readonly (0 | 1 | 2)[] = hasTone && hasSpatial ? [1, 2] : [0];

        const spatialView = cached.tex.createView();
        for (const passMode of stages) {
          const maskUbo = acquireMaskUbo();
          const maskData = packMaskUniforms({
            outWidth: workW,
            outHeight: workH,
            layer,
            compiled: cached.compiled,
            overlayOnly: false,
            passMode,
          });
          queue.writeBuffer(maskUbo, 0, maskData as Float32Array<ArrayBuffer>);

          const maskBg = device.createBindGroup({
            layout: core.maskBgl,
            entries: [
              { binding: 0, resource: { buffer: maskUbo } },
              { binding: 1, resource: currentIsA ? viewA : viewB! },
              { binding: 2, resource: spatialView },
              { binding: 3, resource: currentIsA ? viewB! : viewA },
            ],
          });
          const pass = encoder.beginComputePass({ label: 'kroma-mask-compute' });
          pass.setPipeline(core.maskPipeline);
          pass.setBindGroup(0, maskBg);
          pass.dispatchWorkgroups(wgX, wgY, 1);
          pass.end();
          currentIsA = !currentIsA;
        }
      }

      // Pass 2: Global Spatial Detail + Transparent Cutout Alpha
      if (u.spatialStageActive || bgTransparent) {
        const spatialData = packSpatialUniforms({
          outWidth: workW,
          outHeight: workH,
          recipe: curRecipe,
          bgTransparent,
        });
        queue.writeBuffer(res.spatialUbo, 0, spatialData as Float32Array<ArrayBuffer>);
        const spatialBg = device.createBindGroup({
          layout: core.spatialBgl,
          entries: [
            { binding: 0, resource: { buffer: res.spatialUbo } },
            { binding: 1, resource: currentIsA ? viewA : viewB! },
            { binding: 2, resource: bgMatteView },
            { binding: 3, resource: currentIsA ? viewB! : viewA },
          ],
        });
        const pass = encoder.beginComputePass({ label: 'kroma-spatial-compute' });
        pass.setPipeline(core.spatialPipeline);
        pass.setBindGroup(0, spatialBg);
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
        currentIsA = !currentIsA;
      }

      // Optional Ruby Mask Overlay (O) composited cleanly on top
      if (overlayMask) {
        const cached = getOrUploadSpatialMaskGpu(res, overlayMask, orientedW, orientedH);
        const maskUbo = acquireMaskUbo();
        const overlayData = packMaskUniforms({
          outWidth: workW,
          outHeight: workH,
          layer: overlayMask,
          compiled: cached.compiled,
          overlayOnly: true,
          passMode: 0,
        });
        queue.writeBuffer(maskUbo, 0, overlayData as Float32Array<ArrayBuffer>);
        const overlayBg = device.createBindGroup({
          layout: core.maskBgl,
          entries: [
            { binding: 0, resource: { buffer: maskUbo } },
            { binding: 1, resource: currentIsA ? viewA : viewB! },
            { binding: 2, resource: cached.tex.createView() },
            { binding: 3, resource: currentIsA ? viewB! : viewA },
          ],
        });
        const pass = encoder.beginComputePass({ label: 'kroma-ruby-overlay' });
        pass.setPipeline(core.maskPipeline);
        pass.setBindGroup(0, overlayBg);
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
        currentIsA = !currentIsA;
      }

      res.lastFinalTex = currentIsA ? res.workTexA : res.workTexB;
      lastFinalIsARef.current = currentIsA;
      lastBgTransparentRef.current = bgTransparent;
      gradeDirtyRef.current = false;
    }

    // Pass 3: Present Blit to Swapchain (cached bind group per work texture; ~0.2ms on pan/zoom!)
    const presentData = packPresentUniforms(
      uv,
      clipFlagsRef.current.shadow,
      clipFlagsRef.current.highlight,
      bgTransparent,
      uv.cropBounds,
    );
    queue.writeBuffer(res.presentUbo, 0, presentData as Float32Array<ArrayBuffer>);

    const presentViewA = res.workViewA;
    const presentViewB = res.workViewB;
    let presentBg: GPUBindGroup;
    if (currentIsA) {
      if (!presentViewA) return;
      if (!res.presentBgA) {
        res.presentBgA = device.createBindGroup({
          layout: core.presentBgl,
          entries: [
            { binding: 0, resource: { buffer: res.presentUbo } },
            { binding: 1, resource: presentViewA },
          ],
        });
      }
      presentBg = res.presentBgA;
    } else {
      if (!presentViewB) return;
      if (!res.presentBgB) {
        res.presentBgB = device.createBindGroup({
          layout: core.presentBgl,
          entries: [
            { binding: 0, resource: { buffer: res.presentUbo } },
            { binding: 1, resource: presentViewB },
          ],
        });
      }
      presentBg = res.presentBgB;
    }

    try {
      const swapTex = ctx.getCurrentTexture();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: swapTex.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      renderPass.setPipeline(core.presentPipeline);
      renderPass.setBindGroup(0, presentBg);
      renderPass.draw(3, 1, 0, 0);
      renderPass.end();

      queue.submit([encoder.finish()]);
    } catch {
      onUnavailableRef.current?.();
    }
  };
  renderFrameRef.current = renderFrame;

  const handleCanvasClick = async (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const nx = (event.clientX - rect.left) / rect.width;
    const ny = (event.clientY - rect.top) / rect.height;
    const crop = recipeRef.current.cropTransform;
    const qt = (((crop.quarterTurns ?? 0) % 4) + 4) % 4;
    const swapAxes = qt % 2 === 1;
    const orientedW = swapAxes ? imgSizeRef.current.h : imgSizeRef.current.w;
    const orientedH = swapAxes ? imgSizeRef.current.w : imgSizeRef.current.h;
    const uv = computeViewportUvTransform(
      orientedW,
      orientedH,
      rect.width,
      rect.height,
      fitRef.current,
      applyCropRef.current ? crop.rect : undefined,
      viewportPadPxRef.current,
      viewportZoomRef.current,
      viewportPanRef.current,
    );
    const uOriented = nx * uv.scaleX + uv.offsetX;
    const vOriented = ny * uv.scaleY + uv.offsetY;
    const [minU, minV, maxU, maxV] = uv.cropBounds;
    if (uOriented < minU || uOriented > maxU || vOriented < minV || vOriented > maxV) return;

    // Color Range Eyedropper: read exact 5×5 patch from final 32-bit GPU texture
    if (colorPickActive && onSampleColorRangeRgb && resRef.current?.lastFinalTex) {
      const res = resRef.current;
      const { device, queue } = res.core;
      const tex = res.lastFinalTex!;
      const cx = Math.min(res.workW - 1, Math.max(0, Math.round(uOriented * (res.workW - 1))));
      const cy = Math.min(res.workH - 1, Math.max(0, Math.round(vOriented * (res.workH - 1))));
      const patchW = Math.min(5, res.workW);
      const patchH = Math.min(5, res.workH);
      const x0 = Math.max(0, Math.min(res.workW - patchW, cx - 2));
      const y0 = Math.max(0, Math.min(res.workH - patchH, cy - 2));

      const bytesPerRow = 256; // WebGPU minimum copyTextureToBuffer alignment
      const readBuf = device.createBuffer({
        size: bytesPerRow * patchH,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const enc = device.createCommandEncoder();
        enc.copyTextureToBuffer(
          { texture: tex, origin: [x0, y0, 0] },
          { buffer: readBuf, bytesPerRow, rowsPerImage: patchH },
          [patchW, patchH, 1],
        );
        queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const mapped = readBuf.getMappedRange();
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let cnt = 0;
        for (let row = 0; row < patchH; row++) {
          const rowFloats = new Float32Array(mapped, row * bytesPerRow, patchW * 4);
          for (let col = 0; col < patchW; col++) {
            const o = col * 4;
            sr += Math.max(0, Math.min(1, rowFloats[o]!));
            sg += Math.max(0, Math.min(1, rowFloats[o + 1]!));
            sb += Math.max(0, Math.min(1, rowFloats[o + 2]!));
            cnt += 1;
          }
        }
        readBuf.unmap();
        if (cnt > 0) {
          onSampleColorRangeRgb(sr / cnt, sg / cnt, sb / cnt);
        }
      } catch {
        // Ignore transient GPU readback errors cleanly
      } finally {
        readBuf.destroy();
      }
      return;
    }

    if (!eyedropperActive || !onSampleNeutralRgb) return;

    const [srcU, srcV] = orientSourceUvJs(
      uOriented,
      vOriented,
      Boolean(crop.flipHorizontal),
      Boolean(crop.flipVertical),
      qt,
    );

    const wb = workingBufferRef.current;
    if (wb) {
      const cx = Math.min(wb.width - 1, Math.max(0, Math.round(srcU * (wb.width - 1))));
      const cy = Math.min(wb.height - 1, Math.max(0, Math.round(srcV * (wb.height - 1))));
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let count = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const y = Math.min(wb.height - 1, Math.max(0, cy + dy));
        for (let dx = -2; dx <= 2; dx++) {
          const x = Math.min(wb.width - 1, Math.max(0, cx + dx));
          const idx = (y * wb.width + x) * 4;
          let r = 0;
          let g = 0;
          let b = 0;
          if (wb.float32Rgba) {
            r = wb.float32Rgba[idx] ?? 0;
            g = wb.float32Rgba[idx + 1] ?? 0;
            b = wb.float32Rgba[idx + 2] ?? 0;
          } else if (wb.halfFloatRgba) {
            r = f16BitsToF32(wb.halfFloatRgba[idx] ?? 0);
            g = f16BitsToF32(wb.halfFloatRgba[idx + 1] ?? 0);
            b = f16BitsToF32(wb.halfFloatRgba[idx + 2] ?? 0);
          }
          if (wb.isLinear) {
            const lin709 = wb.isWideGamut
              ? compressRec2020ToLinearSrgb([r, g, b])
              : ([r, g, b] as [number, number, number]);
            sr += linearToSrgb(softClip(lin709[0]));
            sg += linearToSrgb(softClip(lin709[1]));
            sb += linearToSrgb(softClip(lin709[2]));
          } else {
            sr += r;
            sg += g;
            sb += b;
          }
          count += 1;
        }
      }
      if (count > 0) {
        onSampleNeutralRgb(sr / count, sg / count, sb / count);
      }
      return;
    }

    const cachedSource = sourceBitmapRef.current;
    if (cachedSource && cachedSource.width > 0 && cachedSource.height > 0) {
      try {
        const sampleCanvas = document.createElement('canvas');
        sampleCanvas.width = 5;
        sampleCanvas.height = 5;
        const ctx2d = sampleCanvas.getContext('2d');
        if (!ctx2d) return;
        const sx = Math.min(
          cachedSource.width - 5,
          Math.max(0, Math.round(srcU * cachedSource.width) - 2),
        );
        const sy = Math.min(
          cachedSource.height - 5,
          Math.max(0, Math.round(srcV * cachedSource.height) - 2),
        );
        ctx2d.drawImage(cachedSource.bitmap, sx, sy, 5, 5, 0, 0, 5, 5);
        const data = ctx2d.getImageData(0, 0, 5, 5).data;
        let sr = 0;
        let sg = 0;
        let sb = 0;
        const pxCount = data.length / 4;
        for (let i = 0; i < data.length; i += 4) {
          sr += data[i]! / 255;
          sg += data[i + 1]! / 255;
          sb += data[i + 2]! / 255;
        }
        if (pxCount > 0) {
          onSampleNeutralRgb(sr / pxCount, sg / pxCount, sb / pxCount);
        }
      } catch {
        // Ignore cross-origin canvas tainting cleanly in Demo mode
      }
    }
  };

  // Initialize WebGPU context once per mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    let observer: ResizeObserver | null = null;
    let unregisterLost: (() => void) | null = null;

    void createSharedWebGpuCore().then((core) => {
      if (cancelled) return;
      if (!core || core.isLost) {
        onUnavailableRef.current?.();
        return;
      }
      const onDeviceLost = () => {
        if (!cancelled) {
          onUnavailableRef.current?.();
        }
      };
      core.lostListeners.add(onDeviceLost);
      unregisterLost = () => core.lostListeners.delete(onDeviceLost);
      const ctx = canvas.getContext('webgpu');
      if (!ctx) {
        core.lostListeners.delete(onDeviceLost);
        onUnavailableRef.current?.();
        return;
      }
      try {
        ctx.configure({
          device: core.device,
          format: core.swapFormat,
          alphaMode: 'premultiplied',
        });
      } catch {
        core.lostListeners.delete(onDeviceLost);
        onUnavailableRef.current?.();
        return;
      }

      const { device } = core;
      const gradeUbo = device.createBuffer({
        label: 'kroma-grade-ubo',
        size: GRADE_UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const healUbo = device.createBuffer({
        label: 'kroma-heal-ubo',
        size: HEAL_UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const portraitUbo = device.createBuffer({
        label: 'kroma-portrait-ubo',
        size: PORTRAIT_RETOUCH_UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const spatialUbo = device.createBuffer({
        label: 'kroma-spatial-ubo',
        size: SPATIAL_UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const presentUbo = device.createBuffer({
        label: 'kroma-present-ubo',
        size: PRESENT_UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const toneLutTex = device.createTexture({
        label: 'kroma-tone-lut-tex',
        size: [256, 1],
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });

      resRef.current = {
        core,
        ctx,
        gradeUbo,
        healUbo,
        portraitUbo,
        spatialUbo,
        presentUbo,
        toneLutTex,
        toneLutView: toneLutTex.createView(),
        srcTex: null,
        srcView: null,
        workTexA: null,
        workViewA: null,
        workTexB: null,
        workViewB: null,
        workW: 0,
        workH: 0,
        lastFinalTex: null,
        bgMatteTex: null,
        bgMatteView: null,
        bgMatteSig: '',
        lut3dBuf: null,
        lut3dPath: '',
        lut3dParsed: null,
        healSpotsBuf: null,
        healSpotsCap: 0,
        healSegsBuf: null,
        healSegsCap: 0,
        faceParseTex: null,
        faceParseView: null,
        faceParseRef: null,
        inpaintPatchCache: new Map(),
        inpaintUboPool: [],
        gradeBgCache: null,
        presentBgA: null,
        presentBgB: null,
        spatialMaskCache: new Map(),
        maskUboPool: [],
      };

      observer = new ResizeObserver(() => scheduleRender());
      observer.observe(canvas);
      onMaxTextureDimRef.current?.(core.maxTextureDim);
      setGpuReady(true);
    });

    return () => {
      cancelled = true;
      unregisterLost?.();
      unregisterLost = null;
      if (pendingRafRef.current !== null) {
        window.cancelAnimationFrame(pendingRafRef.current);
        pendingRafRef.current = null;
      }
      observer?.disconnect();
      const res = resRef.current;
      resRef.current = null;
      textureReadyRef.current = false;
      if (res) {
        res.srcTex?.destroy();
        res.workTexA?.destroy();
        res.workTexB?.destroy();
        res.bgMatteTex?.destroy();
        res.lut3dBuf?.destroy();
        res.healSpotsBuf?.destroy();
        res.healSegsBuf?.destroy();
        res.faceParseTex?.destroy();
        res.toneLutTex.destroy();
        res.gradeUbo.destroy();
        res.healUbo.destroy();
        res.portraitUbo.destroy();
        res.spatialUbo.destroy();
        res.presentUbo.destroy();
        for (const buf of res.inpaintUboPool) buf.destroy();
        for (const entry of res.inpaintPatchCache.values()) entry.tex?.destroy();
        res.inpaintPatchCache.clear();
        for (const buf of res.maskUboPool) buf.destroy();
        for (const entry of res.spatialMaskCache.values()) entry.tex.destroy();
        res.spatialMaskCache.clear();
      }
    };
  }, []);

  // Upload 16-bit half-float (`KR16` `rgba16float`) or 32-bit float (`KR32` `rgba32float`) working buffer
  // directly to GPU texture memory without JS float conversion loops; all downstream compute passes
  // (`grade_pass.wgsl`, `mask_pass.wgsl`, `spatial_pass.wgsl`) execute in 32-bit `rgba32float`.
  useEffect(() => {
    if (!gpuReady || !workingBuffer) return;
    const res = resRef.current;
    if (!res) return;

    const { device, queue, maxTextureDim } = res.core;
    if (res.core.isLost) {
      onUnavailableRef.current?.();
      return;
    }
    if (workingBuffer.width > maxTextureDim || workingBuffer.height > maxTextureDim) {
      return;
    }

    let cancelled = false;
    try {
      device.pushErrorScope('out-of-memory');
      device.pushErrorScope('validation');

      let tex: GPUTexture;
      if (workingBuffer.halfFloatRgba) {
        tex = device.createTexture({
          label: 'kroma-src-rgba16f',
          size: [workingBuffer.width, workingBuffer.height],
          format: 'rgba16float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        writeTextureRowsChunked(
          queue,
          tex,
          workingBuffer.halfFloatRgba,
          workingBuffer.width,
          workingBuffer.height,
          8,
        );
      } else if (workingBuffer.float32Rgba) {
        tex = device.createTexture({
          label: 'kroma-src-rgba32f',
          size: [workingBuffer.width, workingBuffer.height],
          format: 'rgba32float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        writeTextureRowsChunked(
          queue,
          tex,
          workingBuffer.float32Rgba,
          workingBuffer.width,
          workingBuffer.height,
          16,
        );
      } else {
        void device.popErrorScope();
        void device.popErrorScope();
        return;
      }

      const prevTex = res.srcTex;
      res.srcTex = tex;
      res.srcView = tex.createView();
      res.gradeBgCache = null;
      prevTex?.destroy();

      imgSizeRef.current = { w: workingBuffer.width, h: workingBuffer.height };
      inputLinearRef.current = workingBuffer.isLinear;
      inputWideGamutRef.current = workingBuffer.isWideGamut;
      textureReadyRef.current = true;
      gradeDirtyRef.current = true;
      if (workingBuffer.width > 0 && workingBuffer.height > 0) {
        onNaturalSizeRef.current?.(workingBuffer.width, workingBuffer.height);
      }
      scheduleRender();
      setTextureVersion((v) => v + 1);

      void Promise.all([device.popErrorScope(), device.popErrorScope()]).then(
        ([valErr, oomErr]) => {
          if (!cancelled && (valErr || oomErr)) {
            onUnavailableRef.current?.();
          }
        },
      );
    } catch {
      onUnavailableRef.current?.();
    }
    return () => {
      cancelled = true;
    };
  }, [workingBuffer, gpuReady]);

  // Upload 8-bit preview/source photo texture when no high-bit-depth working buffer is active.
  // Uses the shared downscaled ImageBitmap LRU cache so pre-decoded neighbors upload synchronously (<3ms),
  // and keeps the previous tier visible when upgrading to full native zoom resolution (zero flash).
  const hasWorkingBuffer = workingBuffer !== null;
  const clampedPreviewMaxDim = Math.min(DEFAULT_MAX_WORK_DIM, Math.max(640, previewMaxDim));
  useEffect(() => {
    if (!gpuReady || !src || hasWorkingBuffer) return;
    const res = resRef.current;
    if (!res) return;

    let cancelled = false;

    const uploadCachedBitmap = (entry: CachedDevelopBitmap) => {
      if (cancelled || !resRef.current || workingBufferRef.current) return;
      const curRes = resRef.current;
      const { device, queue, maxTextureDim } = curRes.core;
      if (entry.width > maxTextureDim || entry.height > maxTextureDim) {
        onUnavailableRef.current?.();
        return;
      }
      try {
        sourceBitmapRef.current = entry;
        loadedSrcRef.current = src;
        curRes.srcTex?.destroy();
        const tex = device.createTexture({
          label: 'kroma-src-rgba8',
          size: [entry.width, entry.height],
          format: 'rgba8unorm',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
        });
        queue.copyExternalImageToTexture(
          { source: entry.bitmap },
          { texture: tex },
          [entry.width, entry.height],
        );
        curRes.srcTex = tex;
        curRes.srcView = tex.createView();
        curRes.gradeBgCache = null;
        imgSizeRef.current = { w: entry.width, h: entry.height };
        inputLinearRef.current = false;
        inputWideGamutRef.current = false;
        textureReadyRef.current = true;
        gradeDirtyRef.current = true;
        onNaturalSizeRef.current?.(entry.naturalWidth, entry.naturalHeight);
        scheduleRender();
        setTextureVersion((v) => v + 1);
      } catch {
        onUnavailableRef.current?.();
      }
    };

    const syncHit = peekDevelopBitmap(src, clampedPreviewMaxDim);
    if (syncHit) {
      uploadCachedBitmap(syncHit);
      return () => {
        cancelled = true;
      };
    }

    // Only blank out while switching to a different photo — preserve existing texture during 1:1 tier upgrades
    if (loadedSrcRef.current !== src) {
      textureReadyRef.current = false;
    }
    void loadDevelopBitmap(src, clampedPreviewMaxDim).then((entry) => {
      if (cancelled) return;
      if (!entry) {
        onUnavailableRef.current?.();
        return;
      }
      uploadCachedBitmap(entry);
    });

    return () => {
      cancelled = true;
    };
  }, [src, hasWorkingBuffer, gpuReady, clampedPreviewMaxDim]);

  // Mark GPU compute grade dirty only when recipe, face landmarks/parse map, masks, overlay, or source texture changes
  useEffect(() => {
    if (gpuReady) {
      gradeDirtyRef.current = true;
      scheduleRender();
    }
  }, [gpuReady, recipe, faceOverlay, faceParseMap, activeMaskId, showMaskOverlay, textureVersion]);

  // Pan, zoom, padding, fit, crop preview bounds, and clipping warnings run Pass 3 present-only (~0.2ms!)
  useEffect(() => {
    if (gpuReady) {
      scheduleRender();
    }
  }, [
    gpuReady,
    fit,
    viewportZoom,
    viewportPan.x,
    viewportPan.y,
    viewportPadPx,
    applyCrop,
    showShadowClip,
    showHighlightClip,
  ]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={alt}
      onClick={(e) => {
        void handleCanvasClick(e);
      }}
      className={`block w-full h-full ${eyedropperActive || colorPickActive ? 'cursor-crosshair' : ''} ${className}`}
    />
  );
});
