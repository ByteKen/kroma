import { memo, useEffect, useRef, useState } from 'react';
import gradePassWgsl from '../../../../shaders/develop/grade_pass.wgsl?raw';
import maskPassWgsl from '../../../../shaders/develop/mask_pass.wgsl?raw';
import presentWgsl from '../../../../shaders/develop/present.wgsl?raw';
import spatialPassWgsl from '../../../../shaders/develop/spatial_pass.wgsl?raw';
import type { DevelopRecipe } from '../../domain/develop';
import {
  ensureCubeLut3DLoaded,
  getCachedCubeLut3D,
  type ParsedCubeLut3D,
} from '../../domain/cubeLut3d';
import {
  buildToneCurveLutRgba32F,
  compileDevelopShaderUniforms,
  compressRec2020ToLinearSrgb,
  computeObjectFitUvTransform,
  linearToSrgb,
  softClip,
} from '../../domain/developShaderMath';
import type { DevelopWorkingBuffer } from '../../domain/developWorkingBuffer';
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
  MASK_UNIFORM_FLOATS,
  PRESENT_UNIFORM_FLOATS,
  SPATIAL_UNIFORM_FLOATS,
  packGradeUniforms,
  packMaskUniforms,
  packPresentUniforms,
  packSpatialUniforms,
} from '../../domain/wgslUniformPacker';

export interface DevelopViewportCanvasProps {
  readonly src: string;
  readonly alt: string;
  readonly recipe: DevelopRecipe;
  readonly workingBuffer?: DevelopWorkingBuffer | null;
  readonly fit?: 'cover' | 'contain';
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
}

interface WebGpuDevelopCanvasProps extends DevelopViewportCanvasProps {
  readonly onUnavailable?: () => void;
}

const MAX_WORK_DIM = 2560;

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
  readonly maskBgl: GPUBindGroupLayout;
  readonly maskPipeline: GPUComputePipeline;
  readonly spatialBgl: GPUBindGroupLayout;
  readonly spatialPipeline: GPUComputePipeline;
  readonly presentBgl: GPUBindGroupLayout;
  readonly presentPipeline: GPURenderPipeline;
  readonly dummyLut3dBuf: GPUBuffer;
  readonly dummyWhiteTex: GPUTexture;
  readonly dummyToneLutTex: GPUTexture;
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
      device.lost.then(() => {
        sharedCorePromise = null;
      });

      const swapFormat = navigator.gpu.getPreferredCanvasFormat();
      const maxTextureDim = device.limits.maxTextureDimension2D || 8192;

      const gradeModule = device.createShaderModule({ label: 'kroma-grade-wgsl', code: gradePassWgsl });
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

      return {
        device,
        queue: device.queue,
        swapFormat,
        maxTextureDim,
        gradeBgl,
        gradePipeline,
        maskBgl,
        maskPipeline,
        spatialBgl,
        spatialPipeline,
        presentBgl,
        presentPipeline,
        dummyLut3dBuf,
        dummyWhiteTex,
        dummyToneLutTex,
      };
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

interface CanvasGpuResources {
  readonly core: SharedWebGpuCore;
  readonly ctx: GPUCanvasContext;
  readonly gradeUbo: GPUBuffer;
  readonly spatialUbo: GPUBuffer;
  readonly presentUbo: GPUBuffer;
  readonly toneLutTex: GPUTexture;
  srcTex: GPUTexture | null;
  workTexA: GPUTexture | null;
  workTexB: GPUTexture | null;
  workW: number;
  workH: number;
  lastFinalTex: GPUTexture | null;
  bgMatteTex: GPUTexture | null;
  bgMatteSig: string;
  lut3dBuf: GPUBuffer | null;
  lut3dPath: string;
  lut3dParsed: ParsedCubeLut3D | null;
  readonly spatialMaskCache: Map<string, CachedSpatialMaskGpu>;
  maskUboPool: GPUBuffer[];
}

function ensureWorkTextures(res: CanvasGpuResources, w: number, h: number): void {
  const clampedW = Math.max(1, Math.min(MAX_WORK_DIM, w));
  const clampedH = Math.max(1, Math.min(MAX_WORK_DIM, h));
  if (res.workTexA && res.workTexB && res.workW === clampedW && res.workH === clampedH) {
    return;
  }
  res.workTexA?.destroy();
  res.workTexB?.destroy();
  const desc: GPUTextureDescriptor = {
    size: [clampedW, clampedH],
    format: 'rgba32float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  };
  res.workTexA = res.core.device.createTexture({ ...desc, label: 'kroma-studio-work-a' });
  res.workTexB = res.core.device.createTexture({ ...desc, label: 'kroma-studio-work-b' });
  res.workW = clampedW;
  res.workH = clampedH;
}

function ensureBgMatteTextureGpu(
  res: CanvasGpuResources,
  bg: BackgroundRemovalSettings | undefined,
): boolean {
  if (!bg || !isBackgroundRemovalActive(bg)) return false;
  const sig = `${bg.matteWidth}x${bg.matteHeight}:${bg.matteRleB64}`;
  if (res.bgMatteSig === sig && res.bgMatteTex) return true;
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
  res.bgMatteSig = sig;
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

/**
 * Unified 32-Bit Scene-Referred Wide-Gamut (Linear Rec.2020) WebGPU Develop Viewport.
 * Executes the EXACT same WGSL compute shaders (`shaders/develop/*.wgsl`) as Kroma's Rust `wgpu`
 * DX12/Metal export engine at 60–120 FPS with full `rgba32float` precision.
 */
export const WebGpuDevelopCanvas = memo(function WebGpuDevelopCanvas({
  src,
  alt,
  recipe,
  workingBuffer = null,
  fit = 'cover',
  rotationDeg = 0,
  rotationScale = 1.06,
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
  onUnavailable,
}: WebGpuDevelopCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resRef = useRef<CanvasGpuResources | null>(null);
  const imgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const sourceImgRef = useRef<HTMLImageElement | null>(null);
  const inputLinearRef = useRef(false);
  const inputWideGamutRef = useRef(false);
  const textureReadyRef = useRef(false);
  const workingBufferRef = useRef<DevelopWorkingBuffer | null>(workingBuffer);
  const recipeRef = useRef(recipe);
  const fitRef = useRef(fit);
  const applyCropRef = useRef(applyCrop);
  const clipFlagsRef = useRef({ shadow: showShadowClip, highlight: showHighlightClip });
  const maskOverlayRef = useRef({ activeMaskId, showMaskOverlay });
  const onNaturalSizeRef = useRef(onNaturalSize);
  const onUnavailableRef = useRef(onUnavailable);

  workingBufferRef.current = workingBuffer;
  recipeRef.current = recipe;
  fitRef.current = fit;
  applyCropRef.current = applyCrop;
  clipFlagsRef.current = { shadow: showShadowClip, highlight: showHighlightClip };
  maskOverlayRef.current = { activeMaskId, showMaskOverlay };
  onNaturalSizeRef.current = onNaturalSize;
  onUnavailableRef.current = onUnavailable;

  const [gpuReady, setGpuReady] = useState(false);
  const [textureVersion, setTextureVersion] = useState(0);

  const renderFrame = () => {
    const res = resRef.current;
    const canvas = canvasRef.current;
    if (!res || !canvas || !textureReadyRef.current || !res.srcTex) return;

    const { core, ctx } = res;
    const { device, queue } = core;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = Math.max(1, canvas.clientWidth);
    const cssH = Math.max(1, canvas.clientHeight);
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

    const scaleDown = Math.max(orientedW, orientedH) > MAX_WORK_DIM
      ? MAX_WORK_DIM / Math.max(orientedW, orientedH)
      : 1;
    const workW = Math.max(1, Math.round(orientedW * scaleDown));
    const workH = Math.max(1, Math.round(orientedH * scaleDown));
    ensureWorkTextures(res, workW, workH);
    if (!res.workTexA || !res.workTexB) return;

    const u = compileDevelopShaderUniforms(curRecipe);
    const uv = computeObjectFitUvTransform(
      orientedW,
      orientedH,
      cssW,
      cssH,
      fitRef.current,
      applyCropRef.current ? crop.rect : undefined,
    );

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
    const bgTransparent = Boolean(bgReady && bgCfg?.mode === 'transparent');

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
            renderFrame();
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
      includeFineRotation: false,
      bgMatteReady: bgReady,
      lut3d: activeLutMeta,
    });
    queue.writeBuffer(res.gradeUbo, 0, gradeData as Float32Array<ArrayBuffer>);

    const srcView = res.srcTex.createView();
    const toneView = (u.toneCurveActive ? res.toneLutTex : core.dummyToneLutTex).createView();
    const bgMatteView = (bgReady && res.bgMatteTex ? res.bgMatteTex : core.dummyWhiteTex).createView();
    const viewA = res.workTexA.createView();
    const viewB = res.workTexB.createView();

    const wgX = Math.ceil(workW / 8);
    const wgY = Math.ceil(workH / 8);

    const encoder = device.createCommandEncoder({ label: 'kroma-studio-frame' });

    // Pass 1: 32-Bit Scene-Referred Wide-Gamut Color/Tone/Backdrop
    const gradeBg = device.createBindGroup({
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
    {
      const pass = encoder.beginComputePass({ label: 'kroma-grade-compute' });
      pass.setPipeline(core.gradePipeline);
      pass.setBindGroup(0, gradeBg);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    let currentIsA = true;
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
            { binding: 1, resource: currentIsA ? viewA : viewB },
            { binding: 2, resource: spatialView },
            { binding: 3, resource: currentIsA ? viewB : viewA },
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
          { binding: 1, resource: currentIsA ? viewA : viewB },
          { binding: 2, resource: bgMatteView },
          { binding: 3, resource: currentIsA ? viewB : viewA },
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
          { binding: 1, resource: currentIsA ? viewA : viewB },
          { binding: 2, resource: cached.tex.createView() },
          { binding: 3, resource: currentIsA ? viewB : viewA },
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

    // Pass 3: Present Blit to Swapchain
    const presentData = packPresentUniforms(
      uv,
      clipFlagsRef.current.shadow,
      clipFlagsRef.current.highlight,
      bgTransparent,
    );
    queue.writeBuffer(res.presentUbo, 0, presentData as Float32Array<ArrayBuffer>);

    const presentBg = device.createBindGroup({
      layout: core.presentBgl,
      entries: [
        { binding: 0, resource: { buffer: res.presentUbo } },
        { binding: 1, resource: currentIsA ? viewA : viewB },
      ],
    });

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
    const uv = computeObjectFitUvTransform(
      orientedW,
      orientedH,
      rect.width,
      rect.height,
      fitRef.current,
      applyCropRef.current ? crop.rect : undefined,
    );
    const uOriented = nx * uv.scaleX + uv.offsetX;
    const vOriented = ny * uv.scaleY + uv.offsetY;
    if (uOriented < 0 || uOriented > 1 || vOriented < 0 || vOriented > 1) return;

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

    const img = sourceImgRef.current;
    if (img && img.naturalWidth > 0 && img.naturalHeight > 0) {
      try {
        const sampleCanvas = document.createElement('canvas');
        sampleCanvas.width = 5;
        sampleCanvas.height = 5;
        const ctx2d = sampleCanvas.getContext('2d');
        if (!ctx2d) return;
        const sx = Math.min(
          img.naturalWidth - 5,
          Math.max(0, Math.round(srcU * img.naturalWidth) - 2),
        );
        const sy = Math.min(
          img.naturalHeight - 5,
          Math.max(0, Math.round(srcV * img.naturalHeight) - 2),
        );
        ctx2d.drawImage(img, sx, sy, 5, 5, 0, 0, 5, 5);
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

    void createSharedWebGpuCore().then((core) => {
      if (cancelled) return;
      if (!core) {
        onUnavailableRef.current?.();
        return;
      }
      const ctx = canvas.getContext('webgpu');
      if (!ctx) {
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
        onUnavailableRef.current?.();
        return;
      }

      const { device } = core;
      const gradeUbo = device.createBuffer({
        label: 'kroma-grade-ubo',
        size: GRADE_UNIFORM_FLOATS * 4,
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
        spatialUbo,
        presentUbo,
        toneLutTex,
        srcTex: null,
        workTexA: null,
        workTexB: null,
        workW: 0,
        workH: 0,
        lastFinalTex: null,
        bgMatteTex: null,
        bgMatteSig: '',
        lut3dBuf: null,
        lut3dPath: '',
        lut3dParsed: null,
        spatialMaskCache: new Map(),
        maskUboPool: [],
      };

      observer = new ResizeObserver(() => renderFrame());
      observer.observe(canvas);
      setGpuReady(true);
    });

    return () => {
      cancelled = true;
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
        res.toneLutTex.destroy();
        res.gradeUbo.destroy();
        res.spatialUbo.destroy();
        res.presentUbo.destroy();
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
    if (workingBuffer.width > maxTextureDim || workingBuffer.height > maxTextureDim) {
      return;
    }

    try {
      let tex: GPUTexture;
      if (workingBuffer.halfFloatRgba) {
        res.srcTex?.destroy();
        tex = device.createTexture({
          label: 'kroma-src-rgba16f',
          size: [workingBuffer.width, workingBuffer.height],
          format: 'rgba16float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        queue.writeTexture(
          { texture: tex },
          workingBuffer.halfFloatRgba as Uint16Array<ArrayBuffer>,
          { bytesPerRow: workingBuffer.width * 8, rowsPerImage: workingBuffer.height },
          [workingBuffer.width, workingBuffer.height],
        );
      } else if (workingBuffer.float32Rgba) {
        res.srcTex?.destroy();
        tex = device.createTexture({
          label: 'kroma-src-rgba32f',
          size: [workingBuffer.width, workingBuffer.height],
          format: 'rgba32float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        queue.writeTexture(
          { texture: tex },
          workingBuffer.float32Rgba as Float32Array<ArrayBuffer>,
          { bytesPerRow: workingBuffer.width * 16, rowsPerImage: workingBuffer.height },
          [workingBuffer.width, workingBuffer.height],
        );
      } else {
        return;
      }

      res.srcTex = tex;
      imgSizeRef.current = { w: workingBuffer.width, h: workingBuffer.height };
      inputLinearRef.current = workingBuffer.isLinear;
      inputWideGamutRef.current = workingBuffer.isWideGamut;
      textureReadyRef.current = true;
      if (workingBuffer.width > 0 && workingBuffer.height > 0) {
        onNaturalSizeRef.current?.(workingBuffer.width, workingBuffer.height);
      }
      renderFrame();
      setTextureVersion((v) => v + 1);
    } catch {
      onUnavailableRef.current?.();
    }
  }, [workingBuffer, gpuReady]);

  // Upload 8-bit preview/source photo texture when no high-bit-depth working buffer is active
  const hasWorkingBuffer = workingBuffer !== null;
  useEffect(() => {
    if (!gpuReady || !src || hasWorkingBuffer) return;
    const res = resRef.current;
    if (!res) return;

    let cancelled = false;
    textureReadyRef.current = false;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.decoding = 'async';
    img.onload = () => {
      if (cancelled || !resRef.current) return;
      sourceImgRef.current = img;
      if (workingBufferRef.current) return;

      const curRes = resRef.current;
      const { device, queue, maxTextureDim } = curRes.core;
      if (img.naturalWidth > maxTextureDim || img.naturalHeight > maxTextureDim) {
        onUnavailableRef.current?.();
        return;
      }

      void createImageBitmap(img)
        .then((bitmap) => {
          if (cancelled || !resRef.current || workingBufferRef.current) {
            bitmap.close();
            return;
          }
          try {
            curRes.srcTex?.destroy();
            const tex = device.createTexture({
              label: 'kroma-src-rgba8',
              size: [bitmap.width, bitmap.height],
              format: 'rgba8unorm',
              usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
            });
            queue.copyExternalImageToTexture(
              { source: bitmap },
              { texture: tex },
              [bitmap.width, bitmap.height],
            );
            curRes.srcTex = tex;
            imgSizeRef.current = { w: bitmap.width, h: bitmap.height };
            inputLinearRef.current = false;
            inputWideGamutRef.current = false;
            textureReadyRef.current = true;
            onNaturalSizeRef.current?.(bitmap.width, bitmap.height);
            renderFrame();
            setTextureVersion((v) => v + 1);
          } catch {
            onUnavailableRef.current?.();
          } finally {
            bitmap.close();
          }
        })
        .catch(() => {
          if (!cancelled) {
            onUnavailableRef.current?.();
          }
        });
    };
    img.onerror = () => {
      if (!cancelled) {
        onUnavailableRef.current?.();
      }
    };
    img.src = src;

    return () => {
      cancelled = true;
    };
  }, [src, hasWorkingBuffer, gpuReady]);

  // Redraw immediately on any recipe/mask/clipping/crop/fit change
  useEffect(() => {
    if (gpuReady) {
      renderFrame();
    }
  }, [
    gpuReady,
    recipe,
    fit,
    applyCrop,
    showShadowClip,
    showHighlightClip,
    activeMaskId,
    showMaskOverlay,
    textureVersion,
  ]);

  const transformStyle =
    rotationDeg !== 0
      ? rotationScale !== 1
        ? `rotate(${rotationDeg}deg) scale(${rotationScale})`
        : `rotate(${rotationDeg}deg)`
      : undefined;

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={alt}
      onClick={(e) => {
        void handleCanvasClick(e);
      }}
      style={{ transform: transformStyle }}
      className={`block w-full h-full ${eyedropperActive || colorPickActive ? 'cursor-crosshair' : ''} ${className}`}
    />
  );
});
