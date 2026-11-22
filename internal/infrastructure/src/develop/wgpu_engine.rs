//! Unified 32-Bit Scene-Referred Wide-Gamut `wgpu` Compute Develop Engine (DX12 / Metal / Vulkan).
//!
//! Executes the exact same WGSL compute shaders (`shaders/develop/{grade_pass,mask_pass,spatial_pass}.wgsl`)
//! used by Kroma's WebGPU Studio live preview so full-resolution batch exports and Studio previews share
//! one single GPU math pipeline with `< 1/255` parity against the 32-bit Rayon CPU fallback.

use crate::develop::color_math::{
    build_rgb_tone_curve_luts, compile_color_wheel, compute_calibration_primary_matrix,
    compute_white_balance_gains,
};
use crate::develop::cube_lut::CubeLut3D;
use crate::develop::image_decoder::DecodedDevelopBuffer;
use crate::develop::mask_engine::{compile_local_linear, compile_mask_layer_for_gpu};
use bytemuck::{Pod, Zeroable};
use image::Rgb32FImage;
use kroma_app::error::AppError;
use kroma_domain::develop::{DevelopRecipe, NEUTRAL_KELVIN};
use kroma_domain::masking::{
    decode_matte_rle_b64, BackgroundGradientStyle, BackgroundReplaceMode, MaskCombineOp,
    MaskComponent, MaskLayer,
};
use std::sync::{Mutex, OnceLock};

const GRADE_PASS_WGSL: &str = include_str!("../../../../shaders/develop/grade_pass.wgsl");
const MASK_PASS_WGSL: &str = include_str!("../../../../shaders/develop/mask_pass.wgsl");
const SPATIAL_PASS_WGSL: &str = include_str!("../../../../shaders/develop/spatial_pass.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GradeUniforms {
    dims: [f32; 4],
    crop_rect: [f32; 4],
    orient: [f32; 4],
    linear_flags: [f32; 4],
    wb_and_bw: [f32; 4],
    stage_flags: [f32; 4],
    cal_row0: [f32; 4],
    cal_row1: [f32; 4],
    cal_row2: [f32; 4],
    basic_tone_a: [f32; 4],
    basic_tone_b: [f32; 4],
    hsl: [[f32; 4]; 8],
    cg_shd: [f32; 4],
    cg_mid: [f32; 4],
    cg_hi: [f32; 4],
    cg_meta: [f32; 4],
    vignette: [f32; 4],
    lut_meta: [f32; 4],
    lut_domain: [f32; 4],
    bg_color1: [f32; 4],
    bg_color2: [f32; 4],
    bg_params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MaskUniforms {
    dims: [f32; 4],
    crop_rect: [f32; 4],
    layer_flags: [f32; 4],
    range_meta: [[f32; 4]; 4],
    range_params_a: [[f32; 4]; 4],
    range_params_b: [f32; 4],
    linear_params: [f32; 4],
    wb_gains: [f32; 4],
    tone_a: [f32; 4],
    tone_b: [f32; 4],
    detail: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SpatialUniforms {
    dims: [f32; 4],
    crop_rect: [f32; 4],
    detail_a: [f32; 4],
    detail_b: [f32; 4],
    bg_params: [f32; 4],
}

pub struct WgpuDevelopEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    max_texture_dim: u32,
    grade_bgl: wgpu::BindGroupLayout,
    grade_pipeline: wgpu::ComputePipeline,
    mask_bgl: wgpu::BindGroupLayout,
    mask_pipeline: wgpu::ComputePipeline,
    spatial_bgl: wgpu::BindGroupLayout,
    spatial_pipeline: wgpu::ComputePipeline,
    dummy_lut3d_buf: wgpu::Buffer,
    dummy_white_tex: wgpu::Texture,
    dummy_tone_lut_tex: wgpu::Texture,
    /// Serializes GPU dispatch + readback so parallel Rayon exports keep peak VRAM bounded (~150 MB)
    /// while CPU RAW decoding and JPEG/PNG/TIFF encoding still saturate all CPU cores in parallel.
    dispatch_lock: Mutex<()>,
}

#[derive(Clone, Debug)]
struct ProbedWgpuAdapterInfo {
    backend_label: String,
    adapter_name: String,
}

static PROBED_WGPU_ADAPTER: OnceLock<Option<ProbedWgpuAdapterInfo>> = OnceLock::new();
static SHARED_WGPU_ENGINE: OnceLock<Option<WgpuDevelopEngine>> = OnceLock::new();

/// Lightweight adapter metadata probe used on app startup (`check_runtime_status`).
/// Queries only `adapter.get_info()` and drops the adapter immediately without creating a
/// DX12/Metal `wgpu::Device`, shader compiler, or `gpu-allocator` heaps (~150-250 MB saved at boot).
fn probe_wgpu_adapter_info() -> Option<&'static ProbedWgpuAdapterInfo> {
    PROBED_WGPU_ADAPTER
        .get_or_init(|| {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            });
            let adapter = pollster::block_on(instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                },
            ))
            .ok()?;
            let info = adapter.get_info();
            let backend_label = match info.backend {
                wgpu::Backend::Dx12 => "wgpu DX12".to_string(),
                wgpu::Backend::Metal => "wgpu Metal".to_string(),
                wgpu::Backend::Vulkan => "wgpu Vulkan".to_string(),
                other => format!("wgpu {other:?}"),
            };
            Some(ProbedWgpuAdapterInfo {
                backend_label,
                adapter_name: info.name,
            })
        })
        .as_ref()
}

/// Returns the shared lazy-initialized `WgpuDevelopEngine` (DX12 on Windows, Metal on macOS,
/// Vulkan on Linux) or `None` if no compatible GPU adapter is present.
/// Only initialized on demand when running a GPU develop render/export.
pub fn shared_wgpu_develop_engine() -> Option<&'static WgpuDevelopEngine> {
    SHARED_WGPU_ENGINE
        .get_or_init(WgpuDevelopEngine::try_init)
        .as_ref()
}

/// Returns `true` when a native `wgpu` GPU adapter is present (lightweight probe, no device created).
pub fn is_wgpu_develop_available() -> bool {
    probe_wgpu_adapter_info().is_some()
}

/// Human-readable GPU compute backend label (e.g. `"wgpu DX12"` or `"wgpu Metal"`).
pub fn wgpu_develop_backend_label() -> String {
    match probe_wgpu_adapter_info() {
        Some(info) => info.backend_label.clone(),
        None => "CPU".to_string(),
    }
}

/// Human-readable GPU adapter hardware name (e.g. `"NVIDIA GeForce RTX 4070"` or `"Apple M3 Pro"`).
pub fn wgpu_develop_adapter_name() -> Option<String> {
    probe_wgpu_adapter_info().map(|info| info.adapter_name.clone())
}

impl WgpuDevelopEngine {
    fn try_init() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;

        let limits = adapter.limits();
        let max_texture_dim = limits.max_texture_dimension_2d;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("kroma-develop-wgpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        }))
        .ok()?;

        let grade_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kroma-grade-pass-wgsl"),
            source: wgpu::ShaderSource::Wgsl(GRADE_PASS_WGSL.into()),
        });
        let mask_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kroma-mask-pass-wgsl"),
            source: wgpu::ShaderSource::Wgsl(MASK_PASS_WGSL.into()),
        });
        let spatial_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kroma-spatial-pass-wgsl"),
            source: wgpu::ShaderSource::Wgsl(SPATIAL_PASS_WGSL.into()),
        });

        let grade_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kroma-grade-bgl"),
            entries: &[
                uniform_entry(0),
                unfilterable_tex_entry(1),
                unfilterable_tex_entry(2),
                unfilterable_tex_entry(3),
                readonly_storage_entry(4),
                write_storage_tex_entry(5),
            ],
        });
        let grade_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kroma-grade-pl"),
            bind_group_layouts: &[&grade_bgl],
            push_constant_ranges: &[],
        });
        let grade_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kroma-grade-pipeline"),
            layout: Some(&grade_pl),
            module: &grade_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mask_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kroma-mask-bgl"),
            entries: &[
                uniform_entry(0),
                unfilterable_tex_entry(1),
                unfilterable_tex_entry(2),
                write_storage_tex_entry(3),
            ],
        });
        let mask_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kroma-mask-pl"),
            bind_group_layouts: &[&mask_bgl],
            push_constant_ranges: &[],
        });
        let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kroma-mask-pipeline"),
            layout: Some(&mask_pl),
            module: &mask_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let spatial_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kroma-spatial-bgl"),
            entries: &[
                uniform_entry(0),
                unfilterable_tex_entry(1),
                unfilterable_tex_entry(2),
                write_storage_tex_entry(3),
            ],
        });
        let spatial_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kroma-spatial-pl"),
            bind_group_layouts: &[&spatial_bgl],
            push_constant_ranges: &[],
        });
        let spatial_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kroma-spatial-pipeline"),
            layout: Some(&spatial_pl),
            module: &spatial_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let dummy_lut3d_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kroma-dummy-lut3d"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dummy_lut3d_buf, 0, bytemuck::cast_slice(&[0.0_f32; 4]));

        let dummy_white_tex = upload_rgba8_texture(&device, &queue, 1, 1, &[255, 255, 255, 255]);

        let mut identity_lut = Vec::with_capacity(256 * 4);
        for i in 0..256 {
            let v = i as f32 / 255.0;
            identity_lut.extend_from_slice(&[v, v, v, 1.0]);
        }
        let dummy_tone_lut_tex = upload_rgba32f_texture(&device, &queue, 256, 1, &identity_lut);

        Some(Self {
            device,
            queue,
            max_texture_dim,
            grade_bgl,
            grade_pipeline,
            mask_bgl,
            mask_pipeline,
            spatial_bgl,
            spatial_pipeline,
            dummy_lut3d_buf,
            dummy_white_tex,
            dummy_tone_lut_tex,
            dispatch_lock: Mutex::new(()),
        })
    }

    /// Executes the full 32-bit `rgba32float` WGSL compute pipeline (`grade_pass` → `mask_pass` × N →
    /// `spatial_pass`) on GPU and returns the graded `Rgb32FImage` plus optional transparent cutout alpha.
    pub fn process_buffer_with_alpha_f32(
        &self,
        buffer: &DecodedDevelopBuffer,
        recipe: &DevelopRecipe,
        loaded_lut: Option<&CubeLut3D>,
    ) -> Result<(Rgb32FImage, Option<Vec<f32>>), AppError> {
        let (src_w, src_h) = buffer.image.dimensions();
        if src_w == 0 || src_h == 0 {
            return Err(AppError::ImageProcessingFailure(
                "cannot GPU-develop empty image".to_string(),
            ));
        }

        let crop = recipe.crop_transform;
        let qt = crop.quarter_turns % 4;
        let (ori_w, ori_h) = if qt % 2 == 1 {
            (src_h, src_w)
        } else {
            (src_w, src_h)
        };

        let (out_w, out_h, snapped_crop_rect) = if crop.rect.is_full_frame() {
            (ori_w, ori_h, [0.0_f32, 0.0, 1.0, 1.0])
        } else {
            let x = ((crop.rect.left * ori_w as f32).round() as u32).min(ori_w.saturating_sub(1));
            let y = ((crop.rect.top * ori_h as f32).round() as u32).min(ori_h.saturating_sub(1));
            let cw = ((crop.rect.width * ori_w as f32).round() as u32)
                .max(1)
                .min(ori_w.saturating_sub(x));
            let ch = ((crop.rect.height * ori_h as f32).round() as u32)
                .max(1)
                .min(ori_h.saturating_sub(y));
            (
                cw,
                ch,
                [
                    x as f32 / (ori_w as f32).max(1.0),
                    y as f32 / (ori_h as f32).max(1.0),
                    cw as f32 / (ori_w as f32).max(1.0),
                    ch as f32 / (ori_h as f32).max(1.0),
                ],
            )
        };

        if src_w > self.max_texture_dim
            || src_h > self.max_texture_dim
            || out_w > self.max_texture_dim
            || out_h > self.max_texture_dim
        {
            return Err(AppError::ImageProcessingFailure(format!(
                "image dimensions ({src_w}x{src_h} -> {out_w}x{out_h}) exceed GPU texture limit {}",
                self.max_texture_dim
            )));
        }

        // Pre-compile effective masks before locking the GPU queue (if any mask uses >4 ranges
        // or interleaved range-then-spatial, fall back cleanly to Rayon CPU).
        let effective_masks: Vec<&MaskLayer> =
            recipe.masks.iter().filter(|m| m.is_effective()).collect();
        let mut prepared_masks = Vec::with_capacity(effective_masks.len());
        for layer in effective_masks {
            let Some(compiled) = compile_mask_layer_for_gpu(layer, ori_w, ori_h, 1024) else {
                return Err(AppError::ImageProcessingFailure(
                    "mask stack requires CPU fallback".to_string(),
                ));
            };
            prepared_masks.push((layer, compiled));
        }

        // Pack source RGB32F into RGBA32F on CPU before acquiring GPU lock
        let src_raw = buffer.image.as_raw();
        let mut src_rgba = vec![0.0_f32; (src_w as usize) * (src_h as usize) * 4];
        for (dst, src) in src_rgba
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(src_raw.as_chunks::<3>().0)
        {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = 1.0;
        }

        // Decode optional AI background matte before locking GPU
        let bg = &recipe.background_removal;
        let bg_matte_decoded = if bg.is_active() {
            let mw = bg.matte_width as usize;
            let mh = bg.matte_height as usize;
            decode_matte_rle_b64(&bg.matte_rle_b64, mw * mh).map(|buf| {
                let mut rgba = Vec::with_capacity(buf.len() * 4);
                for &v in &buf {
                    rgba.extend_from_slice(&[v, v, v, 255]);
                }
                (bg.matte_width as u32, bg.matte_height as u32, rgba)
            })
        } else {
            None
        };
        let bg_opaque_mode = if bg_matte_decoded.is_some() {
            match bg.mode {
                BackgroundReplaceMode::SolidColor => 2.0_f32,
                BackgroundReplaceMode::Gradient => 3.0_f32,
                BackgroundReplaceMode::Original | BackgroundReplaceMode::Transparent => 0.0_f32,
            }
        } else {
            0.0_f32
        };
        let bg_transparent = bg_matte_decoded.is_some() && bg.mode == BackgroundReplaceMode::Transparent;

        let need_spatial = recipe.detail_optics.luminance_noise_reduction > 0
            || recipe.detail_optics.color_noise_reduction > 0
            || recipe.basic_tone.texture != 0
            || recipe.basic_tone.clarity != 0
            || recipe.detail_optics.sharpening_amount > 0
            || recipe.detail_optics.grain_amount > 0
            || bg_transparent;

        let _guard = self.dispatch_lock.lock().map_err(|_| {
            AppError::ImageProcessingFailure("wgpu dispatch mutex poisoned".to_string())
        })?;

        let src_tex = upload_rgba32f_texture(&self.device, &self.queue, src_w, src_h, &src_rgba);
        let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let tone_lut_tex_owned = if recipe.tone_curve.is_neutral() {
            None
        } else {
            let (r_lut, g_lut, b_lut) = build_rgb_tone_curve_luts(&recipe.tone_curve);
            let mut lut_rgba = Vec::with_capacity(256 * 4);
            for i in 0..256 {
                lut_rgba.extend_from_slice(&[r_lut[i], g_lut[i], b_lut[i], 1.0]);
            }
            Some(upload_rgba32f_texture(&self.device, &self.queue, 256, 1, &lut_rgba))
        };
        let tone_lut_view = tone_lut_tex_owned
            .as_ref()
            .unwrap_or(&self.dummy_tone_lut_tex)
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bg_matte_tex_owned = bg_matte_decoded
            .as_ref()
            .map(|(mw, mh, rgba)| upload_rgba8_texture(&self.device, &self.queue, *mw, *mh, rgba));
        let bg_matte_view = bg_matte_tex_owned
            .as_ref()
            .unwrap_or(&self.dummy_white_tex)
            .create_view(&wgpu::TextureViewDescriptor::default());

        let lut3d_buf_owned = loaded_lut.map(|lut| {
            let mut packed = Vec::with_capacity(lut.table().len() * 4);
            for rgb in lut.table() {
                packed.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
            }
            let bytes: &[u8] = bytemuck::cast_slice(&packed);
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kroma-lut3d-storage"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytes);
            buf
        });
        let lut3d_buf_ref = lut3d_buf_owned.as_ref().unwrap_or(&self.dummy_lut3d_buf);

        let tex_a = create_storage_rgba32f_texture(&self.device, out_w, out_h, "kroma-work-a");
        let view_a = tex_a.create_view(&wgpu::TextureViewDescriptor::default());
        let need_ping_pong = !prepared_masks.is_empty() || need_spatial;
        let tex_b = if need_ping_pong {
            Some(create_storage_rgba32f_texture(&self.device, out_w, out_h, "kroma-work-b"))
        } else {
            None
        };
        let view_b = tex_b
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let grade_uniforms = build_grade_uniforms(
            out_w,
            out_h,
            src_w,
            src_h,
            snapped_crop_rect,
            buffer.is_linear,
            buffer.is_wide_gamut,
            recipe,
            loaded_lut,
            bg_opaque_mode,
        );
        let grade_ubo = create_uniform_buffer(&self.device, &self.queue, &grade_uniforms);
        let grade_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kroma-grade-bg"),
            layout: &self.grade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grade_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&tone_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&bg_matte_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lut3d_buf_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
            ],
        });

        let wg_x = out_w.div_ceil(8);
        let wg_y = out_h.div_ceil(8);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kroma-develop-encoder"),
            });

        // Pass 1: Global 32-bit Wide-Gamut Color/Tone/LUT/Backdrop
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kroma-grade-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.grade_pipeline);
            pass.set_bind_group(0, &grade_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Track which ping-pong texture currently holds the latest image (`true` = A, `false` = B)
        let mut current_is_a = true;

        // Pass 2: Local & AI Masks
        let mut mask_temp_resources: Vec<(wgpu::Texture, wgpu::Buffer, wgpu::BindGroup)> = Vec::new();
        for (layer, (has_spatial, mw, mh, rgba, ranges)) in &prepared_masks {
            let spatial_tex = if *has_spatial {
                upload_rgba8_texture(&self.device, &self.queue, *mw, *mh, rgba)
            } else {
                upload_rgba8_texture(&self.device, &self.queue, 1, 1, &[255, 255, 255, 255])
            };
            let spatial_view = spatial_tex.create_view(&wgpu::TextureViewDescriptor::default());

            let adj = layer.adjustments;
            let has_tone_or_color = adj.temperature_shift != 0
                || adj.tint_shift != 0
                || adj.exposure_ev.abs() > 1e-4
                || adj.contrast != 0
                || adj.highlights != 0
                || adj.shadows != 0
                || adj.whites != 0
                || adj.blacks != 0
                || adj.dehaze != 0
                || adj.saturation != 0;
            let has_spatial_detail = adj.has_spatial_adjustments();

            // When both local tone AND local spatial detail are active on the same layer, run
            // Stage 1 (tone-only, storing alpha in .a) then Stage 2 (spatial-only) for exact
            // parity with `mask_engine.rs`.
            let stages: &[f32] = if has_tone_or_color && has_spatial_detail {
                &[1.0, 2.0]
            } else {
                &[0.0]
            };

            let mut last_tex_holder = Some(spatial_tex);
            for &pass_mode in stages {
                let u_mask = build_mask_uniforms(
                    out_w,
                    out_h,
                    snapped_crop_rect,
                    layer,
                    *has_spatial,
                    ranges,
                    pass_mode,
                );
                let mask_ubo = create_uniform_buffer(&self.device, &self.queue, &u_mask);
                let (src_v, dst_v) = if current_is_a {
                    (&view_a, view_b.as_ref().expect("ping-pong B"))
                } else {
                    (view_b.as_ref().expect("ping-pong B"), &view_a)
                };
                let mask_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("kroma-mask-bg"),
                    layout: &self.mask_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: mask_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src_v),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&spatial_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(dst_v),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("kroma-mask-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.mask_pipeline);
                    pass.set_bind_group(0, &mask_bg, &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
                current_is_a = !current_is_a;
                let dummy_keep = last_tex_holder.take().unwrap_or_else(|| {
                    upload_rgba8_texture(&self.device, &self.queue, 1, 1, &[0, 0, 0, 255])
                });
                mask_temp_resources.push((dummy_keep, mask_ubo, mask_bg));
            }
        }

        // Pass 3: Global Spatial Detail (Color NR, Luma NR, Texture, Clarity, Sharpening, Grain) + Transparent Cutout Alpha
        let spatial_resources = if need_spatial {
            let spatial_scale = (out_w.max(out_h) as f32 / 1024.0).max(1.0);
            let u_spatial = SpatialUniforms {
                dims: [out_w as f32, out_h as f32, spatial_scale, 0.0],
                crop_rect: snapped_crop_rect,
                detail_a: [
                    recipe.detail_optics.luminance_noise_reduction as f32,
                    recipe.detail_optics.color_noise_reduction as f32,
                    recipe.basic_tone.texture as f32,
                    recipe.basic_tone.clarity as f32,
                ],
                detail_b: [
                    recipe.detail_optics.sharpening_amount as f32,
                    recipe.detail_optics.grain_amount as f32,
                    recipe.detail_optics.grain_size as f32,
                    0.0,
                ],
                bg_params: [
                    if bg_transparent { 1.0 } else { 0.0 },
                    bg.feather,
                    0.0,
                    0.0,
                ],
            };
            let spatial_ubo = create_uniform_buffer(&self.device, &self.queue, &u_spatial);
            let (src_v, dst_v) = if current_is_a {
                (&view_a, view_b.as_ref().expect("ping-pong B"))
            } else {
                (view_b.as_ref().expect("ping-pong B"), &view_a)
            };
            let spatial_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kroma-spatial-bg"),
                layout: &self.spatial_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: spatial_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src_v),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&bg_matte_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(dst_v),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("kroma-spatial-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.spatial_pipeline);
                pass.set_bind_group(0, &spatial_bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            current_is_a = !current_is_a;
            Some((spatial_ubo, spatial_bg))
        } else {
            None
        };

        // Readback final `rgba32float` texture into CPU `Rgb32FImage` (+ optional cutout alpha plane)
        let final_tex = if current_is_a {
            &tex_a
        } else {
            tex_b.as_ref().expect("ping-pong B")
        };
        let unpadded_bytes_per_row = out_w * 16;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let readback_size = (padded_bytes_per_row as u64) * (out_h as u64);

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kroma-readback-staging"),
            size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: final_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(out_h),
                },
            },
            wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        // Keep bind groups alive until submit completes
        drop(spatial_resources);
        drop(mask_temp_resources);

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .map_err(|err| {
                AppError::ImageProcessingFailure(format!("wgpu readback channel failed: {err}"))
            })?
            .map_err(|err| {
                AppError::ImageProcessingFailure(format!("wgpu buffer map failed: {err:?}"))
            })?;

        let mapped = slice.get_mapped_range();
        let w_u = out_w as usize;
        let h_u = out_h as usize;
        let row_stride_bytes = padded_bytes_per_row as usize;
        let mut rgb_out = vec![0.0_f32; w_u * h_u * 3];
        let mut alpha_out = if bg_transparent {
            Some(vec![1.0_f32; w_u * h_u])
        } else {
            None
        };

        for y in 0..h_u {
            let row_bytes = &mapped[y * row_stride_bytes..y * row_stride_bytes + w_u * 16];
            let floats: &[f32] = bytemuck::cast_slice(row_bytes);
            let rgb_row = &mut rgb_out[y * w_u * 3..(y + 1) * w_u * 3];
            if let Some(ref mut a_vec) = alpha_out {
                let a_row = &mut a_vec[y * w_u..(y + 1) * w_u];
                for (x, a_slot) in a_row.iter_mut().enumerate().take(w_u) {
                    let s = x * 4;
                    let d = x * 3;
                    rgb_row[d] = floats[s];
                    rgb_row[d + 1] = floats[s + 1];
                    rgb_row[d + 2] = floats[s + 2];
                    *a_slot = floats[s + 3];
                }
            } else {
                for x in 0..w_u {
                    let s = x * 4;
                    let d = x * 3;
                    rgb_row[d] = floats[s];
                    rgb_row[d + 1] = floats[s + 1];
                    rgb_row[d + 2] = floats[s + 2];
                }
            }
        }

        drop(mapped);
        staging_buf.unmap();

        let img = Rgb32FImage::from_raw(out_w, out_h, rgb_out).ok_or_else(|| {
            AppError::ImageProcessingFailure("wgpu output dimensions mismatch".to_string())
        })?;
        Ok((img, alpha_out))
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn unfilterable_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn readonly_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn write_storage_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn create_uniform_buffer<T: Pod>(device: &wgpu::Device, queue: &wgpu::Queue, data: &T) -> wgpu::Buffer {
    let bytes = bytemuck::bytes_of(data);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kroma-uniform-buf"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

fn create_storage_rgba32f_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn upload_rgba32f_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgba_f32: &[f32],
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("kroma-rgba32f-upload"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(rgba_f32),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    tex
}

fn upload_rgba8_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgba_u8: &[u8],
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("kroma-rgba8-upload"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba_u8,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    tex
}

#[allow(clippy::too_many_arguments)]
fn build_grade_uniforms(
    out_w: u32,
    out_h: u32,
    src_w: u32,
    src_h: u32,
    crop_rect: [f32; 4],
    is_linear: bool,
    is_wide_gamut: bool,
    recipe: &DevelopRecipe,
    loaded_lut: Option<&CubeLut3D>,
    bg_opaque_mode: f32,
) -> GradeUniforms {
    let exposure_ev = recipe.basic_tone.exposure_ev.clamp(-5.0, 5.0);
    let linear_active = exposure_ev.abs() > 1e-4
        || recipe.white_balance.temperature_kelvin != NEUTRAL_KELVIN
        || recipe.white_balance.tint != 0;
    let exposure_gain = 2.0_f32.powf(exposure_ev);
    let wb = compute_white_balance_gains(recipe.white_balance);
    let cal_active = !recipe.camera_calibration.is_neutral();
    let cal_mat = compute_calibration_primary_matrix(&recipe.camera_calibration);
    let tone_curve_active = !recipe.tone_curve.is_neutral();
    let hsl_active = !recipe.hsl_mixer.is_neutral();

    let hsl_bands = [
        recipe.hsl_mixer.red,
        recipe.hsl_mixer.orange,
        recipe.hsl_mixer.yellow,
        recipe.hsl_mixer.green,
        recipe.hsl_mixer.aqua,
        recipe.hsl_mixer.blue,
        recipe.hsl_mixer.purple,
        recipe.hsl_mixer.magenta,
    ];
    let mut hsl = [[0.0_f32; 4]; 8];
    for (i, band) in hsl_bands.iter().enumerate() {
        hsl[i] = [band.hue as f32, band.saturation as f32, band.luminance as f32, 0.0];
    }

    let cg = &recipe.color_grading;
    let cg_active = !cg.is_neutral();
    let (shd_tint, shd_lum) = compile_color_wheel(cg.shadows);
    let (mid_tint, mid_lum) = compile_color_wheel(cg.midtones);
    let (hi_tint, hi_lum) = compile_color_wheel(cg.highlights);
    let balance_shift = (cg.balance as f32 / 100.0) * 0.25;
    let cg_pivot = (0.5 - balance_shift).clamp(0.18, 0.82);
    let cg_spread = 0.14 + (cg.blending as f32 / 100.0).clamp(0.0, 1.0) * 0.36;

    let (lut_meta, lut_domain) = match (loaded_lut, recipe.lut_3d.as_ref()) {
        (Some(lut), Some(lut_ref)) if lut_ref.intensity > f32::EPSILON => {
            let dmin = lut.domain_min();
            let dmax = lut.domain_max();
            (
                [lut.size() as f32, lut_ref.intensity.clamp(0.0, 1.0), dmin[0], dmin[1]],
                [dmin[2], dmax[0], dmax[1], dmax[2]],
            )
        }
        _ => ([0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]),
    };

    let bg = &recipe.background_removal;
    let crop = recipe.crop_transform;

    GradeUniforms {
        dims: [out_w as f32, out_h as f32, src_w as f32, src_h as f32],
        crop_rect,
        orient: [
            if crop.flip_horizontal { 1.0 } else { 0.0 },
            if crop.flip_vertical { 1.0 } else { 0.0 },
            (crop.quarter_turns % 4) as f32,
            if crop.rotation_deg.abs() > 1e-3 {
                crop.rotation_deg.to_radians()
            } else {
                0.0
            },
        ],
        linear_flags: [
            if is_linear { 1.0 } else { 0.0 },
            if is_wide_gamut { 1.0 } else { 0.0 },
            if linear_active { 1.0 } else { 0.0 },
            exposure_gain,
        ],
        wb_and_bw: [
            wb[0],
            wb[1],
            wb[2],
            if recipe.basic_tone.black_and_white { 1.0 } else { 0.0 },
        ],
        stage_flags: [
            if cal_active { 1.0 } else { 0.0 },
            recipe.camera_calibration.shadows_tint as f32 / 100.0,
            if tone_curve_active { 1.0 } else { 0.0 },
            if hsl_active { 1.0 } else { 0.0 },
        ],
        cal_row0: [cal_mat[0][0], cal_mat[0][1], cal_mat[0][2], 0.0],
        cal_row1: [cal_mat[1][0], cal_mat[1][1], cal_mat[1][2], 0.0],
        cal_row2: [cal_mat[2][0], cal_mat[2][1], cal_mat[2][2], 0.0],
        basic_tone_a: [
            recipe.basic_tone.shadows as f32,
            recipe.basic_tone.highlights as f32,
            recipe.basic_tone.whites as f32,
            recipe.basic_tone.blacks as f32,
        ],
        basic_tone_b: [
            recipe.basic_tone.dehaze as f32,
            recipe.basic_tone.contrast as f32,
            recipe.basic_tone.vibrance as f32,
            recipe.basic_tone.saturation as f32,
        ],
        hsl,
        cg_shd: [shd_tint[0], shd_tint[1], shd_tint[2], shd_lum],
        cg_mid: [mid_tint[0], mid_tint[1], mid_tint[2], mid_lum],
        cg_hi: [hi_tint[0], hi_tint[1], hi_tint[2], hi_lum],
        cg_meta: [if cg_active { 1.0 } else { 0.0 }, cg_pivot, cg_spread, 0.0],
        vignette: [
            recipe.detail_optics.vignette_amount as f32,
            recipe.detail_optics.vignette_midpoint as f32,
            recipe.detail_optics.vignette_feather as f32,
            0.0,
        ],
        lut_meta,
        lut_domain,
        bg_color1: [
            bg.color_rgb[0] as f32 / 255.0,
            bg.color_rgb[1] as f32 / 255.0,
            bg.color_rgb[2] as f32 / 255.0,
            bg_opaque_mode,
        ],
        bg_color2: [
            bg.color2_rgb[0] as f32 / 255.0,
            bg.color2_rgb[1] as f32 / 255.0,
            bg.color2_rgb[2] as f32 / 255.0,
            if bg.gradient_style == BackgroundGradientStyle::Linear {
                1.0
            } else {
                0.0
            },
        ],
        bg_params: [bg.gradient_angle_deg, bg.feather, 0.0, 0.0],
    }
}

fn build_mask_uniforms(
    out_w: u32,
    out_h: u32,
    crop_rect: [f32; 4],
    layer: &MaskLayer,
    has_spatial: bool,
    ranges: &[&MaskComponent],
    pass_mode: f32,
) -> MaskUniforms {
    let mut range_meta = [[0.0_f32; 4]; 4];
    let mut range_params_a = [[0.0_f32; 4]; 4];
    let mut range_params_b = [0.0_f32; 4];

    for (i, comp) in ranges.iter().take(4).enumerate() {
        let op_code = |op: MaskCombineOp| -> f32 {
            match op {
                MaskCombineOp::Add => 0.0,
                MaskCombineOp::Subtract => 1.0,
                MaskCombineOp::Intersect => 2.0,
            }
        };
        match comp {
            MaskComponent::LuminanceRange {
                op,
                inverted,
                min_luma,
                max_luma,
                feather,
                ..
            } => {
                range_meta[i] = [1.0, op_code(*op), if *inverted { 1.0 } else { 0.0 }, 0.0];
                range_params_a[i] = [*min_luma, *max_luma, *feather, 0.0];
            }
            MaskComponent::ColorRange {
                op,
                inverted,
                target_hue_deg,
                target_sat,
                target_luma,
                tolerance,
                feather,
                ..
            } => {
                range_meta[i] = [2.0, op_code(*op), if *inverted { 1.0 } else { 0.0 }, 0.0];
                range_params_a[i] = [*target_hue_deg, *target_sat, *target_luma, *tolerance];
                range_params_b[i] = *feather;
            }
            _ => {}
        }
    }

    let adj = layer.adjustments;
    let (wb_gains, exp_gain, linear_active) = compile_local_linear(adj);
    let spatial_scale = ((out_w.max(out_h) as f32) / 1024.0).clamp(0.5, 4.0);

    MaskUniforms {
        dims: [out_w as f32, out_h as f32, spatial_scale, pass_mode],
        crop_rect,
        layer_flags: [
            if has_spatial { 1.0 } else { 0.0 },
            if layer.inverted { 1.0 } else { 0.0 },
            (layer.opacity as f32 / 100.0).clamp(0.0, 1.0),
            0.0,
        ],
        range_meta,
        range_params_a,
        range_params_b,
        linear_params: [if linear_active { 1.0 } else { 0.0 }, exp_gain, 0.0, 0.0],
        wb_gains: [wb_gains[0], wb_gains[1], wb_gains[2], 0.0],
        tone_a: [
            adj.highlights as f32,
            adj.shadows as f32,
            adj.whites as f32,
            adj.blacks as f32,
        ],
        tone_b: [
            adj.dehaze as f32,
            adj.contrast as f32,
            adj.saturation as f32,
            0.0,
        ],
        detail: [adj.texture as f32, adj.clarity as f32, adj.sharpness as f32, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::develop::pixel_engine::RayonImageDevelopEngine;
    use image::Rgb;
    use kroma_domain::develop::{NormalizedRect, ToneCurvePoint};
    use kroma_domain::masking::{encode_matte_rle_b64, LocalAdjustments};

    fn assert_images_match(label: &str, gpu: &Rgb32FImage, cpu: &Rgb32FImage, max_delta: f32) {
        assert_eq!(
            gpu.dimensions(),
            cpu.dimensions(),
            "{label}: dimension mismatch"
        );
        let mut worst = 0.0_f32;
        let mut worst_pos = (0_u32, 0_u32);
        for (x, y, g_px) in gpu.enumerate_pixels() {
            let c_px = cpu.get_pixel(x, y);
            for ch in 0..3 {
                let d = (g_px.0[ch] - c_px.0[ch]).abs();
                if d > worst {
                    worst = d;
                    worst_pos = (x, y);
                }
            }
        }
        assert!(
            worst <= max_delta,
            "{label}: GPU vs CPU max delta {worst:.5} exceeded {max_delta:.5} at {worst_pos:?} (gpu={:?}, cpu={:?})",
            gpu.get_pixel(worst_pos.0, worst_pos.1).0,
            cpu.get_pixel(worst_pos.0, worst_pos.1).0,
        );
    }

    fn synthetic_wide_gamut_hdr_buffer(w: u32, h: u32) -> DecodedDevelopBuffer {
        let mut img = Rgb32FImage::new(w, h);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let u = x as f32 / (w.saturating_sub(1).max(1) as f32);
            let v = y as f32 / (h.saturating_sub(1).max(1) as f32);
            // Includes dark shadows, skin midtones, vivid Rec.2020 colors, and >1.0 HDR highlights
            let r = 0.02 + u * 2.1;
            let g = 0.04 + v * 1.4;
            let b = 0.01 + (1.0 - u) * 1.6;
            *px = Rgb([r, g, b]);
        }
        DecodedDevelopBuffer {
            image: img,
            is_linear: true,
            is_wide_gamut: true,
            bit_depth: 32,
            is_raw_demosaic: true,
        }
    }

    fn synthetic_srgb_buffer(w: u32, h: u32) -> DecodedDevelopBuffer {
        let mut img = Rgb32FImage::new(w, h);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let u = x as f32 / (w.saturating_sub(1).max(1) as f32);
            let v = y as f32 / (h.saturating_sub(1).max(1) as f32);
            let wave = ((x as f32 * 0.45).sin() * 0.15 + 0.5).clamp(0.0, 1.0);
            *px = Rgb([u, v, wave]);
        }
        DecodedDevelopBuffer {
            image: img,
            is_linear: false,
            is_wide_gamut: false,
            bit_depth: 16,
            is_raw_demosaic: false,
        }
    }

    #[test]
    fn uniform_structs_satisfy_16_byte_wgsl_alignment() {
        assert_eq!(std::mem::size_of::<GradeUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<MaskUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<SpatialUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<GradeUniforms>(), 29 * 16);
        assert_eq!(std::mem::size_of::<MaskUniforms>(), 17 * 16);
        assert_eq!(std::mem::size_of::<SpatialUniforms>(), 5 * 16);
    }

    #[test]
    fn present_wgsl_compiles_cleanly_on_wgpu() {
        let Some(gpu) = shared_wgpu_develop_engine() else {
            return;
        };
        const PRESENT_WGSL: &str = include_str!("../../../../shaders/develop/present.wgsl");
        let _module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kroma-present-wgsl-check"),
            source: wgpu::ShaderSource::Wgsl(PRESENT_WGSL.into()),
        });
    }

    #[test]
    fn wgpu_golden_parity_matches_rayon_cpu_across_full_develop_pipeline() {
        let Some(gpu) = shared_wgpu_develop_engine() else {
            // Headless CI runners without a GPU adapter skip execution cleanly after struct checks
            return;
        };
        let cpu = RayonImageDevelopEngine::new();
        let tolerance = 1.0 / 255.0; // Strictly < 1/255 8-bit quantum

        // 1. Wide-gamut Rec.2020 Linear RAW + HDR Highlight Recovery + Calibration + Curves + HSL + 3-Way + Vignette
        let raw_buf = synthetic_wide_gamut_hdr_buffer(48, 36);
        let mut full_grade = DevelopRecipe::default();
        full_grade.white_balance.temperature_kelvin = 6400;
        full_grade.white_balance.tint = -12;
        full_grade.basic_tone.exposure_ev = -0.65;
        full_grade.basic_tone.highlights = -75;
        full_grade.basic_tone.shadows = 40;
        full_grade.basic_tone.whites = -30;
        full_grade.basic_tone.blacks = 15;
        full_grade.basic_tone.dehaze = 20;
        full_grade.basic_tone.contrast = 22;
        full_grade.basic_tone.vibrance = 35;
        full_grade.basic_tone.saturation = 10;
        full_grade.camera_calibration.red_hue = 18;
        full_grade.camera_calibration.blue_hue = -25;
        full_grade.camera_calibration.shadows_tint = -15;
        full_grade.tone_curve.points = vec![
            ToneCurvePoint { x: 0.0, y: 0.04 },
            ToneCurvePoint { x: 0.3, y: 0.26 },
            ToneCurvePoint { x: 0.75, y: 0.80 },
            ToneCurvePoint { x: 1.0, y: 0.98 },
        ];
        full_grade.tone_curve.red_points = vec![
            ToneCurvePoint { x: 0.0, y: 0.02 },
            ToneCurvePoint { x: 1.0, y: 1.0 },
        ];
        full_grade.hsl_mixer.orange.saturation = 25;
        full_grade.hsl_mixer.blue.luminance = -20;
        full_grade.color_grading.shadows.hue_deg = 215.0;
        full_grade.color_grading.shadows.saturation = 30;
        full_grade.color_grading.highlights.hue_deg = 42.0;
        full_grade.color_grading.highlights.saturation = 24;
        full_grade.detail_optics.vignette_amount = -35;
        full_grade.crop_transform.rect = NormalizedRect::new(0.1, 0.1, 0.75, 0.75).unwrap();

        let (gpu_img, _) = gpu
            .process_buffer_with_alpha_f32(&raw_buf, &full_grade, None)
            .expect("gpu wide-gamut grade");
        let (cpu_img, _) = cpu
            .process_buffer_with_alpha_cpu_f32(&raw_buf, &full_grade)
            .expect("cpu wide-gamut grade");
        assert_images_match("wide-gamut HDR full grade", &gpu_img, &cpu_img, tolerance);

        // 2. B&W Mix + 3D .cube LUT + Quarter-Turn + Flip
        let srgb_buf = synthetic_srgb_buffer(40, 32);
        let mut bw_recipe = DevelopRecipe::default();
        bw_recipe.basic_tone.black_and_white = true;
        bw_recipe.hsl_mixer.red.luminance = 30;
        bw_recipe.hsl_mixer.blue.luminance = -40;
        bw_recipe.crop_transform.quarter_turns = 1;
        bw_recipe.crop_transform.flip_horizontal = true;

        let (gpu_bw, _) = gpu
            .process_buffer_with_alpha_f32(&srgb_buf, &bw_recipe, None)
            .expect("gpu bw");
        let (cpu_bw, _) = cpu
            .process_buffer_with_alpha_cpu_f32(&srgb_buf, &bw_recipe)
            .expect("cpu bw");
        assert_images_match("B&W + orient", &gpu_bw, &cpu_bw, tolerance);

        // 3. Spatial Detail (Color NR + Luma NR + Texture + Clarity + Sharpening + Film Grain)
        let mut spatial_recipe = DevelopRecipe::default();
        spatial_recipe.detail_optics.color_noise_reduction = 55;
        spatial_recipe.detail_optics.luminance_noise_reduction = 45;
        spatial_recipe.basic_tone.texture = 40;
        spatial_recipe.basic_tone.clarity = 50;
        spatial_recipe.detail_optics.sharpening_amount = 65;
        spatial_recipe.detail_optics.grain_amount = 30;
        spatial_recipe.detail_optics.grain_size = 35;

        let (gpu_sp, _) = gpu
            .process_buffer_with_alpha_f32(&srgb_buf, &spatial_recipe, None)
            .expect("gpu spatial");
        let (cpu_sp, _) = cpu
            .process_buffer_with_alpha_cpu_f32(&srgb_buf, &spatial_recipe)
            .expect("cpu spatial");
        assert_images_match("spatial detail + grain", &gpu_sp, &cpu_sp, tolerance);

        // 4. Composable Local Mask (Radial + Luminance Range + local Exposure & Clarity) + Studio Backdrop
        let matte_w = 16_u16;
        let matte_h = 16_u16;
        let mut matte = vec![0_u8; (matte_w as usize) * (matte_h as usize)];
        for y in 0..matte_h {
            for x in 0..matte_w {
                let dx = x as f32 - 8.0;
                let dy = y as f32 - 8.0;
                let v = if dx * dx + dy * dy < 25.0 { 255 } else { 0 };
                matte[(y as usize) * (matte_w as usize) + (x as usize)] = v;
            }
        }
        let matte_b64 = encode_matte_rle_b64(&matte);

        let mut mask_recipe = DevelopRecipe::default();
        mask_recipe.background_removal.mode = BackgroundReplaceMode::Gradient;
        mask_recipe.background_removal.color_rgb = [28, 34, 48];
        mask_recipe.background_removal.color2_rgb = [180, 195, 210];
        mask_recipe.background_removal.matte_width = matte_w;
        mask_recipe.background_removal.matte_height = matte_h;
        mask_recipe.background_removal.matte_rle_b64 = matte_b64;
        mask_recipe.masks.push(MaskLayer {
            id: "m1".to_string(),
            name: "Subject Pop".to_string(),
            enabled: true,
            inverted: false,
            opacity: 85,
            components: vec![
                MaskComponent::RadialGradient {
                    id: "rad1".to_string(),
                    op: MaskCombineOp::Add,
                    inverted: false,
                    center_x: 0.5,
                    center_y: 0.5,
                    radius_x: 0.35,
                    radius_y: 0.35,
                    rotation_deg: 0.0,
                    feather: 50.0,
                },
                MaskComponent::LuminanceRange {
                    id: "lum1".to_string(),
                    op: MaskCombineOp::Intersect,
                    inverted: false,
                    min_luma: 0.15,
                    max_luma: 0.90,
                    feather: 0.12,
                },
            ],
            adjustments: LocalAdjustments {
                exposure_ev: 0.45,
                contrast: 18,
                clarity: 35,
                saturation: 12,
                ..Default::default()
            },
        });

        let (gpu_mk, _) = gpu
            .process_buffer_with_alpha_f32(&srgb_buf, &mask_recipe, None)
            .expect("gpu masks + backdrop");
        let (cpu_mk, _) = cpu
            .process_buffer_with_alpha_cpu_f32(&srgb_buf, &mask_recipe)
            .expect("cpu masks + backdrop");
        // Allow tiny 2/255 margin for 8-bit spatial mask quantization on radial gradient boundaries
        assert_images_match("local masks + backdrop", &gpu_mk, &cpu_mk, 2.0 / 255.0);
    }
}
