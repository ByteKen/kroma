use crate::develop::color_math::CompiledDevelopGrade;
use crate::develop::cube_lut::CubeLut3D;
use crate::develop::histogram::compute_rgb_histogram;
use crate::develop::image_decoder;
use crate::develop::mask_engine::{
    apply_local_masks_f32, apply_opaque_backdrop_f32, compute_transparent_alpha_f32,
};
use crate::develop::wgpu_engine::shared_wgpu_develop_engine;
use crate::vision::onnx_runtime::is_gpu_acceleration_enabled;
use image::codecs::jpeg::JpegEncoder;
use image::{imageops, DynamicImage, ImageFormat, Rgb, Rgb32FImage, RgbImage, Rgba, RgbaImage};
use kroma_app::error::AppError;
use kroma_app::ports::develop_engine::{
    BatchRenderItemOutcome, BatchRenderJob, ImageDevelopPort, OutputImageFormat, RenderMetrics,
};
use kroma_domain::develop::{CropTransform, DevelopRecipe, NormalizedRect, RgbHistogram};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, Cursor};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Multi-threaded CPU parametric image develop engine backed by `rawler` (RAW) + `image` + `rayon`.
/// Grades in 32-bit floating point so 16-bit camera RAW files retain full highlight/shadow latitude.
#[derive(Debug, Default, Clone)]
pub struct RayonImageDevelopEngine;

impl RayonImageDevelopEngine {
    pub fn new() -> Self {
        Self
    }

    /// Runs the full non-destructive DevelopRecipe in-memory on an 8-bit `RgbImage`.
    pub fn process_rgb_image(
        &self,
        source: &RgbImage,
        recipe: &DevelopRecipe,
    ) -> Result<RgbImage, AppError> {
        let dyn_img = DynamicImage::ImageRgb8(source.clone());
        let graded_f32 = self.process_dynamic_to_f32(&dyn_img, recipe)?;
        Ok(finalize_rgb8(graded_f32))
    }

    /// Core 32-bit float grading pipeline preserving full 16-bit camera RAW dynamic range.
    pub fn process_dynamic_to_f32(
        &self,
        source: &DynamicImage,
        recipe: &DevelopRecipe,
    ) -> Result<Rgb32FImage, AppError> {
        let buffer = image_decoder::DecodedDevelopBuffer {
            image: source.to_rgb32f(),
            is_linear: false,
            is_wide_gamut: false,
            bit_depth: 16,
            is_raw_demosaic: false,
        };
        self.process_buffer_to_f32(&buffer, recipe)
    }

    /// Grades a decoded high-bit-depth buffer (`is_linear = true` for scene-linear Camera RAWs,
    /// `is_linear = false` for sRGB PNG/JPEG/TIFF) using the shared 32-bit WGSL `wgpu` compute
    /// engine when GPU acceleration is enabled, with automatic fallback to 32-bit Rayon CPU.
    pub fn process_buffer_to_f32(
        &self,
        buffer: &image_decoder::DecodedDevelopBuffer,
        recipe: &DevelopRecipe,
    ) -> Result<Rgb32FImage, AppError> {
        let (mut working, alpha_opt) = self.process_buffer_with_alpha_f32(buffer, recipe)?;
        if let Some(alpha) = alpha_opt {
            composite_over_white_f32(&mut working, &alpha);
        }
        Ok(working)
    }

    fn process_buffer_with_alpha_f32(
        &self,
        buffer: &image_decoder::DecodedDevelopBuffer,
        recipe: &DevelopRecipe,
    ) -> Result<(Rgb32FImage, Option<Vec<f32>>), AppError> {
        let loaded_lut = match &recipe.lut_3d {
            Some(lut_ref) if lut_ref.intensity > f32::EPSILON => {
                Some(CubeLut3D::load_cached(&lut_ref.cube_path)?)
            }
            _ => None,
        };

        // Route full-resolution export through the unified 32-bit WGSL `wgpu` compute engine
        // (DX12 / Metal) whenever GPU acceleration is enabled and available, falling back
        // automatically to Rayon 32-bit CPU on CPU-only mode or unsupported edge cases.
        if is_gpu_acceleration_enabled()
            && let Some(gpu) = shared_wgpu_develop_engine()
            && let Ok(res) = gpu.process_buffer_with_alpha_f32(buffer, recipe, loaded_lut.as_deref())
        {
            return Ok(res);
        }

        self.process_buffer_with_alpha_cpu_inner(buffer, recipe, loaded_lut.as_deref())
    }

    /// Pure 32-bit Rayon CPU implementation (used as automatic fallback, for fast 384px live
    /// histograms, and as the Golden Reference in GPU-vs-CPU parity tests).
    pub fn process_buffer_with_alpha_cpu_f32(
        &self,
        buffer: &image_decoder::DecodedDevelopBuffer,
        recipe: &DevelopRecipe,
    ) -> Result<(Rgb32FImage, Option<Vec<f32>>), AppError> {
        let loaded_lut = match &recipe.lut_3d {
            Some(lut_ref) if lut_ref.intensity > f32::EPSILON => {
                Some(CubeLut3D::load_cached(&lut_ref.cube_path)?)
            }
            _ => None,
        };
        self.process_buffer_with_alpha_cpu_inner(buffer, recipe, loaded_lut.as_deref())
    }

    fn process_buffer_with_alpha_cpu_inner(
        &self,
        buffer: &image_decoder::DecodedDevelopBuffer,
        recipe: &DevelopRecipe,
        loaded_lut: Option<&CubeLut3D>,
    ) -> Result<(Rgb32FImage, Option<Vec<f32>>), AppError> {
        // 1-2. Optional Crop & Bilinear Rotation on full 32-bit float pixels
        let mut working = apply_crop_and_rotation_f32(&buffer.image, recipe.crop_transform);

        // 3. Parallel per-pixel parametric color/tone grading (32-bit Linear Rec.2020 for RAW, sRGB otherwise)
        let grade = CompiledDevelopGrade::compile(recipe, loaded_lut);
        let is_linear = buffer.is_linear;
        let is_wide_gamut = buffer.is_wide_gamut;
        let width = working.width();
        let height = working.height();
        let half_w = (width as f32 * 0.5).max(1.0);
        let half_h = (height as f32 * 0.5).max(1.0);

        let raw_buf = working.as_mut();
        let row_stride = (width as usize) * 3;

        raw_buf
            .par_chunks_exact_mut(row_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let ny = ((y as f32) - half_h) / half_h;
                for x in 0..(width as usize) {
                    let nx = ((x as f32) - half_w) / half_w;
                    let idx = x * 3;
                    let px = [row[idx], row[idx + 1], row[idx + 2]];
                    let graded =
                        grade.grade_pixel_in_color_space(px, is_linear, is_wide_gamut, nx, ny);
                    row[idx] = graded[0];
                    row[idx + 1] = graded[1];
                    row[idx + 2] = graded[2];
                }
            });

        // 3a. Optional Opaque Studio Backdrop (Solid / Gradient) composited BEFORE local masks so
        // AI Background / Radial / Linear / Brush masks naturally adjust the replaced backdrop.
        apply_opaque_backdrop_f32(&mut working, recipe);

        // 3b. Parallel Local & AI Masking stack (Linear, Radial, Brush, Range, AI Matte)
        apply_local_masks_f32(&mut working, recipe);

        // 4. Parallel spatial detail pass (Luminance NR, Texture, Clarity, Sharpening, Film Grain)
        // matching the shared WGSL `spatial_pass.wgsl` and WebGL2 Pass 2 formula-for-formula.
        apply_spatial_detail_f32(&mut working, recipe);

        // 5. Optional Transparent Cutout alpha plane (for 32-bit RGBA transparent PNG export)
        let transparent_alpha = compute_transparent_alpha_f32(&working, recipe);

        Ok((working, transparent_alpha))
    }

    /// Computes a fast RGB histogram directly from an already-decoded high-bit-depth thumbnail
    /// buffer (zero disk I/O, exact linear RAW match, runs directly on fast CPU threads without
    /// GPU readback overhead).
    pub fn compute_histogram_from_buffer(
        &self,
        buffer: &image_decoder::DecodedDevelopBuffer,
        recipe: &DevelopRecipe,
    ) -> Result<RgbHistogram, AppError> {
        let (mut graded_f32, alpha_opt) = self.process_buffer_with_alpha_cpu_f32(buffer, recipe)?;
        if let Some(alpha) = alpha_opt {
            composite_over_white_f32(&mut graded_f32, &alpha);
        }
        let graded_rgb8 = finalize_rgb8(graded_f32);
        Ok(compute_rgb_histogram(&graded_rgb8))
    }

    /// Estimates the output byte size in memory for a single photo without decoding full-res RAW
    /// payloads or waking up the `wgpu` GPU device:
    /// - Reads exact native upright dimensions from file headers (~0.5ms, ~0 MB RAM) and applies
    ///   crop/rotation geometry.
    /// - Uncompressed 16-bit TIFF size is computed directly from exact output pixel count (`0` pixel decode).
    /// - JPEG and PNG sample a compact `<=1024px` preview (reusing Kroma's cached fast preview when
    ///   available), grade on CPU, encode with the target encoder/quality, and scale by pixel-area ratio.
    pub fn measure_rendered_bytes(
        &self,
        source_path: &str,
        recipe: &DevelopRecipe,
        format: OutputImageFormat,
    ) -> Result<u64, AppError> {
        self.measure_rendered_bytes_with_preview(source_path, None, recipe, format)
    }

    /// Lightweight export size estimator that optionally reuses an already-extracted fast JPEG
    /// preview for RAW/TIFF sources so estimating output size never triggers full Bayer demosaicing.
    pub fn measure_rendered_bytes_with_preview(
        &self,
        source_path: &str,
        preview_path: Option<&str>,
        recipe: &DevelopRecipe,
        format: OutputImageFormat,
    ) -> Result<u64, AppError> {
        const EXPORT_ESTIMATE_SAMPLE_MAX_DIM: u32 = 1024;

        let native_dims = image_decoder::read_native_photo_dimensions(source_path)?;
        let (out_w, out_h) = compute_cropped_output_dimensions(native_dims, recipe.crop_transform);
        let full_output_pixels = (out_w as u64).saturating_mul(out_h as u64);

        // 16-bit uncompressed RGB TIFF size is deterministic from pixel dimensions alone.
        if matches!(format, OutputImageFormat::Tiff) {
            return Ok(full_output_pixels.saturating_mul(6).saturating_add(4096));
        }

        let sample_rgb = match preview_path {
            Some(path) if Path::new(path).is_file() => {
                image_decoder::decode_fast_preview_max_dim(path, EXPORT_ESTIMATE_SAMPLE_MAX_DIM)
                    .or_else(|_| {
                        image_decoder::decode_fast_preview_max_dim(
                            source_path,
                            EXPORT_ESTIMATE_SAMPLE_MAX_DIM,
                        )
                    })?
            }
            _ => image_decoder::decode_fast_preview_max_dim(
                source_path,
                EXPORT_ESTIMATE_SAMPLE_MAX_DIM,
            )?,
        };

        let sample_buffer = image_decoder::DecodedDevelopBuffer {
            image: DynamicImage::ImageRgb8(sample_rgb).to_rgb32f(),
            is_linear: false,
            is_wide_gamut: false,
            bit_depth: 8,
            is_raw_demosaic: false,
        };

        // Run strictly on CPU (~10ms at <=1024px) so opening Export never initializes DX12/Metal wgpu.
        let (mut graded_f32, alpha_opt) =
            self.process_buffer_with_alpha_cpu_f32(&sample_buffer, recipe)?;
        let sample_pixels = (graded_f32.width() as u64).saturating_mul(graded_f32.height() as u64);

        let sample_bytes = if matches!(format, OutputImageFormat::Png) && alpha_opt.is_some() {
            encode_graded_bytes_with_alpha(graded_f32, alpha_opt.as_deref(), format)?
        } else {
            if let Some(alpha) = alpha_opt {
                composite_over_white_f32(&mut graded_f32, &alpha);
            }
            encode_graded_bytes_with_alpha(graded_f32, None, format)?
        };

        Ok(scale_sample_encoded_bytes(
            sample_bytes,
            sample_pixels,
            full_output_pixels,
            format,
        ))
    }

    /// Renders a 32-bit graded JPEG thumbnail directly from a pre-decoded `DecodedDevelopBuffer`
    /// using the exact same 32-bit WGSL `wgpu` / Rayon pipeline (tone, curves, HSL, 3-way wheels,
    /// camera calibration, 3D LUT, AI backdrops, local masks, spatial detail, crop/rotation) as
    /// full-resolution export, writing atomically via a sibling temporary file so concurrent readers
    /// never observe partially encoded JPEGs.
    pub fn render_buffer_to_jpeg(
        &self,
        buffer: &image_decoder::DecodedDevelopBuffer,
        recipe: &DevelopRecipe,
        output_path: &Path,
        quality: u8,
    ) -> Result<(), AppError> {
        let graded_f32 = self.process_buffer_to_f32(buffer, recipe)?;
        let rgb8 = finalize_rgb8(graded_f32);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                AppError::ImageProcessingFailure(format!(
                    "failed to create thumbnail directory '{}': {err}",
                    parent.display()
                ))
            })?;
        }
        let unique_suffix = format!(
            "{:?}.{}.tmp",
            std::thread::current().id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        )
        .replace(|c: char| !c.is_ascii_alphanumeric() && c != '.', "_");
        let tmp_path = output_path.with_extension(format!("jpg.{unique_suffix}"));
        {
            let file = File::create(&tmp_path).map_err(|err| {
                AppError::ImageProcessingFailure(format!(
                    "failed to create thumbnail '{}': {err}",
                    tmp_path.display()
                ))
            })?;
            let writer = BufWriter::new(file);
            let mut encoder = JpegEncoder::new_with_quality(writer, quality.clamp(50, 100));
            encoder.encode_image(&rgb8).map_err(|err| {
                AppError::ImageProcessingFailure(format!(
                    "failed to encode 32-bit thumbnail '{}': {err}",
                    output_path.display()
                ))
            })?;
        }
        if let Err(_rename_err) = fs::rename(&tmp_path, output_path) {
            // On Windows, rename fails if another thread already materialized the same immutable hash file.
            if output_path.exists() {
                let _ = fs::remove_file(&tmp_path);
            } else {
                fs::copy(&tmp_path, output_path).map_err(|err| {
                    let _ = fs::remove_file(&tmp_path);
                    AppError::ImageProcessingFailure(format!(
                        "failed to finalize 32-bit thumbnail '{}': {err}",
                        output_path.display()
                    ))
                })?;
                let _ = fs::remove_file(&tmp_path);
            }
        }
        Ok(())
    }
}

impl ImageDevelopPort for RayonImageDevelopEngine {
    fn render_photo(
        &self,
        source_path: &str,
        output_path: &str,
        recipe: &DevelopRecipe,
        format: OutputImageFormat,
    ) -> Result<RenderMetrics, AppError> {
        let start = Instant::now();
        let decoded = image_decoder::decode_develop_buffer(source_path, 0)?;
        let (graded_f32, transparent_alpha) =
            self.process_buffer_with_alpha_f32(&decoded, recipe)?;
        let output_width = graded_f32.width();
        let output_height = graded_f32.height();

        save_graded_output(graded_f32, transparent_alpha.as_deref(), output_path, format)?;

        Ok(RenderMetrics {
            output_width,
            output_height,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn render_batch(&self, jobs: &[BatchRenderJob]) -> Vec<BatchRenderItemOutcome> {
        self.render_batch_with_progress(jobs, &|_, _, _| {})
    }

    fn render_batch_with_progress(
        &self,
        jobs: &[BatchRenderJob],
        on_item_complete: &(dyn Fn(usize, usize, &str) + Sync),
    ) -> Vec<BatchRenderItemOutcome> {
        let total = jobs.len();
        let completed = AtomicUsize::new(0);
        jobs.par_iter()
            .map(|job| {
                let outcome = match self.render_photo(
                    &job.source_path,
                    &job.output_path,
                    &job.recipe,
                    job.format,
                ) {
                    Ok(metrics) => BatchRenderItemOutcome::Succeeded {
                        photo_id: job.photo_id.clone(),
                        output_path: job.output_path.clone(),
                        metrics,
                    },
                    Err(err) => BatchRenderItemOutcome::Failed {
                        photo_id: job.photo_id.clone(),
                        source_path: job.source_path.clone(),
                        reason: err.to_string(),
                    },
                };
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                let file_label = Path::new(&job.output_path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(job.photo_id.as_str());
                on_item_complete(done, total, file_label);
                outcome
            })
            .collect()
    }

    fn sample_average_rgb(
        &self,
        source_path: &str,
        region: NormalizedRect,
    ) -> Result<(f32, f32, f32), AppError> {
        // Use fast embedded preview for RAW whiteboard WB sampling when available
        let img = image_decoder::decode_fast_preview_rgb(source_path)?;
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 {
            return Err(AppError::ImageProcessingFailure(
                "cannot sample RGB from empty image".to_string(),
            ));
        }

        let x0 = ((region.left * w as f32).floor() as u32).min(w - 1);
        let y0 = ((region.top * h as f32).floor() as u32).min(h - 1);
        let x1 = (((region.left + region.width) * w as f32).ceil() as u32).clamp(x0 + 1, w);
        let y1 = (((region.top + region.height) * h as f32).ceil() as u32).clamp(y0 + 1, h);

        let mut sum_r = 0.0_f64;
        let mut sum_g = 0.0_f64;
        let mut sum_b = 0.0_f64;
        let mut count = 0_u64;

        for y in y0..y1 {
            for x in x0..x1 {
                let [r, g, b] = img.get_pixel(x, y).0;
                sum_r += (r as f64) / 255.0;
                sum_g += (g as f64) / 255.0;
                sum_b += (b as f64) / 255.0;
                count += 1;
            }
        }

        let denom = count.max(1) as f64;
        Ok((
            (sum_r / denom) as f32,
            (sum_g / denom) as f32,
            (sum_b / denom) as f32,
        ))
    }

    fn compute_histogram(
        &self,
        source_path: &str,
        recipe: &DevelopRecipe,
    ) -> Result<RgbHistogram, AppError> {
        // Use a compact 384px preview for live RGB histograms (<1.5 MB RAM, ~2ms) instead of
        // allocating a full-resolution 20–100 MB float buffer on every slider/undo update.
        let preview_rgb = image_decoder::decode_fast_preview_max_dim(source_path, 384)?;
        let decoded = DynamicImage::ImageRgb8(preview_rgb);
        let graded_f32 = self.process_dynamic_to_f32(&decoded, recipe)?;
        let graded_rgb8 = finalize_rgb8(graded_f32);
        Ok(compute_rgb_histogram(&graded_rgb8))
    }
}

fn finalize_rgb8(graded: Rgb32FImage) -> RgbImage {
    DynamicImage::ImageRgb32F(graded).to_rgb8()
}

/// Resolution-proportional spatial detail & texture engine (Luminance NR, Color NR, Texture,
/// Clarity, Sharpening, Film Grain) operating in 32-bit float on the post-grade image so JPEG,
/// PNG, 16-bit TIFF, live histograms, and the 2-pass WebGL2 Develop viewport match 1:1.
fn apply_spatial_detail_f32(img: &mut Rgb32FImage, recipe: &DevelopRecipe) {
    let nr = (recipe.detail_optics.luminance_noise_reduction as f32 / 100.0).clamp(0.0, 1.0);
    let cnr = (recipe.detail_optics.color_noise_reduction as f32 / 100.0).clamp(0.0, 1.0);
    let tex = (recipe.basic_tone.texture as f32 / 100.0).clamp(-1.0, 1.0);
    let clar = (recipe.basic_tone.clarity as f32 / 100.0).clamp(-1.0, 1.0);
    let sharp = (recipe.detail_optics.sharpening_amount as f32 / 100.0).clamp(0.0, 1.5);
    let grain_amount = (recipe.detail_optics.grain_amount as f32 / 100.0).clamp(0.0, 1.0);
    let grain_size = (recipe.detail_optics.grain_size as f32 / 100.0).clamp(0.0, 1.0);

    if nr <= 1e-4
        && cnr <= 1e-4
        && tex.abs() <= 1e-4
        && clar.abs() <= 1e-4
        && sharp <= 1e-4
        && grain_amount <= 1e-4
    {
        return;
    }

    let (w, h) = img.dimensions();
    if w < 2 || h < 2 {
        return;
    }

    let w_usize = w as usize;
    let h_usize = h as usize;
    let raw = img.as_raw();
    let need_luma = nr > 1e-4 || cnr > 1e-4 || tex.abs() > 1e-4 || clar.abs() > 1e-4 || sharp > 1e-4;
    let mut luma_plane = if need_luma {
        vec![0.0_f32; w_usize * h_usize]
    } else {
        Vec::new()
    };
    if need_luma {
        luma_plane
            .par_chunks_exact_mut(w_usize)
            .enumerate()
            .for_each(|(y, luma_row)| {
                let row_base = y * w_usize * 3;
                for (x, slot) in luma_row.iter_mut().enumerate() {
                    let idx = row_base + x * 3;
                    *slot = 0.2126 * raw[idx] + 0.7152 * raw[idx + 1] + 0.0722 * raw[idx + 2];
                }
            });
    }

    // Optional chroma planes (Cr = R - Y, Cb = B - Y) only allocated when Color NR > 0
    let (cr_plane, cb_plane) = if cnr > 1e-4 {
        let mut cr = vec![0.0_f32; w_usize * h_usize];
        let mut cb = vec![0.0_f32; w_usize * h_usize];
        cr.par_chunks_exact_mut(w_usize)
            .zip(cb.par_chunks_exact_mut(w_usize))
            .enumerate()
            .for_each(|(y, (cr_row, cb_row))| {
                let row_base = y * w_usize * 3;
                let luma_base = y * w_usize;
                for x in 0..w_usize {
                    let idx = row_base + x * 3;
                    let y_val = luma_plane[luma_base + x];
                    cr_row[x] = raw[idx] - y_val;
                    cb_row[x] = raw[idx + 2] - y_val;
                }
            });
        (cr, cb)
    } else {
        (Vec::new(), Vec::new())
    };

    let scale = (w.max(h) as f32 / 1024.0).max(1.0);
    let r_nr = 1.25 * scale;
    let r_cnr = 2.2 * scale;
    let r_tex = 2.5 * scale;
    let r_clar = 11.0 * scale;
    let r_sharp = 1.0 * scale;
    let diag_tex = 0.70710677 * r_tex;
    let inner_clar = 0.45 * r_clar;
    let diag_clar = 0.70710677 * r_clar;
    let grain_grid = 1024.0 / (0.8 + grain_size * 2.4);

    let row_stride = w_usize * 3;
    img.as_mut()
        .par_chunks_exact_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = y as f32;
            let row_luma_base = y * w_usize;
            for x in 0..w_usize {
                let fx = x as f32;
                let idx = x * 3;
                let mut rgb = [row[idx], row[idx + 1], row[idx + 2]];
                let lc = if need_luma {
                    luma_plane[row_luma_base + x]
                } else {
                    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                };

                // 0. Edge-preserving YCbCr Color Noise Reduction (smooths chroma speckle, keeps luma sharp)
                if cnr > 1e-4 {
                    let sigma_c = 0.04 + cnr * 0.14;
                    let inv_2sig2 = -0.5 / (sigma_c * sigma_c);
                    let cr0 = cr_plane[row_luma_base + x];
                    let cb0 = cb_plane[row_luma_base + x];
                    let mut sum_cr = cr0;
                    let mut sum_cb = cb0;
                    let mut sum_w = 1.0_f32;
                    const CNR_OFFSETS: [(f32, f32, f32); 8] = [
                        (-1.0, 0.0, 0.7),
                        (1.0, 0.0, 0.7),
                        (0.0, -1.0, 0.7),
                        (0.0, 1.0, 0.7),
                        (-1.0, -1.0, 0.5),
                        (1.0, -1.0, 0.5),
                        (-1.0, 1.0, 0.5),
                        (1.0, 1.0, 0.5),
                    ];
                    for (ox, oy, sw) in CNR_OFFSETS {
                        let sx = fx + ox * r_cnr;
                        let sy = fy + oy * r_cnr;
                        let ln = sample_luma_bilinear(&luma_plane, w_usize, h_usize, sx, sy);
                        let diff = ln - lc;
                        let w_tot = sw * (diff * diff * inv_2sig2).exp();
                        sum_cr += sample_luma_bilinear(&cr_plane, w_usize, h_usize, sx, sy) * w_tot;
                        sum_cb += sample_luma_bilinear(&cb_plane, w_usize, h_usize, sx, sy) * w_tot;
                        sum_w += w_tot;
                    }
                    let inv_w_sum = 1.0 / sum_w.max(1e-5);
                    let blend = cnr * 0.85;
                    let next_cr = cr0 + (sum_cr * inv_w_sum - cr0) * blend;
                    let next_cb = cb0 + (sum_cb * inv_w_sum - cb0) * blend;
                    // Reconstruct RGB from exact original luma Y + smoothed (Cr, Cb)
                    let next_r = (lc + next_cr).clamp(0.0, 1.0);
                    let next_b = (lc + next_cb).clamp(0.0, 1.0);
                    let next_g = ((lc - 0.2126 * next_r - 0.0722 * next_b) / 0.7152).clamp(0.0, 1.0);
                    rgb = [next_r, next_g, next_b];
                }

                let mut total_delta = 0.0_f32;

                // 1. Edge-preserving 3x3 bilateral Luminance NR
                if nr > 1e-4 {
                    let sigma_r = 0.03 + nr * 0.11;
                    let inv_2sig2 = -0.5 / (sigma_r * sigma_r);
                    let mut sum_val = lc;
                    let mut sum_w = 1.0_f32;
                    const OFFSETS: [(f32, f32, f32); 8] = [
                        (-1.0, 0.0, 0.65),
                        (1.0, 0.0, 0.65),
                        (0.0, -1.0, 0.65),
                        (0.0, 1.0, 0.65),
                        (-1.0, -1.0, 0.45),
                        (1.0, -1.0, 0.45),
                        (-1.0, 1.0, 0.45),
                        (1.0, 1.0, 0.45),
                    ];
                    for (ox, oy, sw) in OFFSETS {
                        let ln = sample_luma_bilinear(
                            &luma_plane,
                            w_usize,
                            h_usize,
                            fx + ox * r_nr,
                            fy + oy * r_nr,
                        );
                        let diff = ln - lc;
                        let w_tot = sw * (diff * diff * inv_2sig2).exp();
                        sum_val += ln * w_tot;
                        sum_w += w_tot;
                    }
                    let l_nr = sum_val / sum_w.max(1e-5);
                    total_delta += (l_nr - lc) * (nr * 0.90);
                }

                // 2. Medium-fine edge-clamped micro-contrast (Texture)
                if tex.abs() > 1e-4 {
                    let blur_tex = 0.20 * lc
                        + 0.12
                            * (sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx - r_tex, fy)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx + r_tex, fy)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx, fy - r_tex)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx, fy + r_tex))
                        + 0.08
                            * (sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx - diag_tex, fy - diag_tex)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx + diag_tex, fy - diag_tex)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx - diag_tex, fy + diag_tex)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx + diag_tex, fy + diag_tex));
                    let detail_tex = (lc - blur_tex).clamp(-0.18, 0.18);
                    total_delta += detail_tex * tex * 0.85;
                }

                // 3. Broad midtone-masked local contrast (Clarity)
                if clar.abs() > 1e-4 {
                    let blur_clar = 0.16 * lc
                        + 0.10
                            * (sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx - inner_clar, fy)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx + inner_clar, fy)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx, fy - inner_clar)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx, fy + inner_clar))
                        + 0.06
                            * (sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx - r_clar, fy)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx + r_clar, fy)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx, fy - r_clar)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx, fy + r_clar))
                        + 0.05
                            * (sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx - diag_clar, fy - diag_clar)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx + diag_clar, fy - diag_clar)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx - diag_clar, fy + diag_clar)
                                + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx + diag_clar, fy + diag_clar));
                    let midtone_mask = (4.0 * lc * (1.0 - lc)).clamp(0.0, 1.0).powf(0.70);
                    let detail_clar = (lc - blur_clar).clamp(-0.30, 0.30);
                    total_delta += detail_clar * clar * 1.15 * midtone_mask;
                }

                // 4. Thresholded luminance unsharp mask (Sharpening)
                if sharp > 1e-4 {
                    let avg4 = 0.25
                        * (sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx - r_sharp, fy)
                            + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx + r_sharp, fy)
                            + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx, fy - r_sharp)
                            + sample_luma_bilinear(&luma_plane, w_usize, h_usize, fx, fy + r_sharp));
                    let hp = lc - avg4;
                    let mag = (hp.abs() - 0.008).max(0.0);
                    let thresh_hp = hp.signum() * mag;
                    total_delta += (thresh_hp * sharp * 1.10).clamp(-0.25, 0.25);
                }

                if total_delta.abs() > 1e-6 {
                    let next_luma = (lc + total_delta).clamp(0.0, 1.0);
                    let actual_delta = next_luma - lc;
                    let ratio = if lc > 1e-4 {
                        (next_luma / lc).clamp(0.0, 3.0)
                    } else {
                        1.0
                    };
                    let t = ((lc - 0.02) / 0.14).clamp(0.0, 1.0);
                    let blend = t * t * (3.0 - 2.0 * t);

                    for channel in &mut rgb {
                        let additive = (*channel + actual_delta).clamp(0.0, 1.0);
                        let mult = if lc > 1e-4 {
                            (*channel * ratio).clamp(0.0, 1.0)
                        } else {
                            next_luma
                        };
                        *channel = additive + (mult - additive) * blend;
                    }
                }

                // 5. Resolution-proportional organic Film Grain (cross-platform arithmetic hash matching WGSL & WebGL2)
                if grain_amount > 1e-4 {
                    let u_norm = (fx + 0.5) / (w as f32);
                    let v_norm_exact = (fy + 0.5) / (h as f32);
                    let gx = (u_norm * grain_grid + 1e-4).floor();
                    let gy = (v_norm_exact * grain_grid + 1e-4).floor();
                    let n1 = grain_hash_12(gx, gy);
                    let n2 = grain_hash_12(gx + 17.0, gy + 41.0);
                    let signed_noise = (n1 + n2) - 1.0; // Triangular [-1..+1] film-like distribution
                    let l_cur = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
                    let mid_env = 0.35 + 0.65 * (4.0 * l_cur * (1.0 - l_cur)).clamp(0.0, 1.0);
                    let g_delta = signed_noise * grain_amount * 0.11 * mid_env;
                    rgb[0] = (rgb[0] + g_delta).clamp(0.0, 1.0);
                    rgb[1] = (rgb[1] + g_delta).clamp(0.0, 1.0);
                    rgb[2] = (rgb[2] + g_delta).clamp(0.0, 1.0);
                }

                row[idx] = rgb[0];
                row[idx + 1] = rgb[1];
                row[idx + 2] = rgb[2];
            }
        });
}

#[inline]
fn pcg_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

#[inline]
fn grain_hash_12(px: f32, py: f32) -> f32 {
    let ix = px.max(0.0) as u32;
    let iy = py.max(0.0) as u32;
    let h = pcg_hash(ix ^ pcg_hash(iy.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)));
    ((h & 0x00FF_FFFF) as f32) / 16_777_216.0
}

#[inline]
fn sample_luma_bilinear(plane: &[f32], w: usize, h: usize, fx: f32, fy: f32) -> f32 {
    let max_x = (w - 1) as f32;
    let max_y = (h - 1) as f32;
    let cx = fx.clamp(0.0, max_x);
    let cy = fy.clamp(0.0, max_y);
    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = cx - x0 as f32;
    let ty = cy - y0 as f32;

    let row0 = y0 * w;
    let row1 = y1 * w;
    let top = plane[row0 + x0] * (1.0 - tx) + plane[row0 + x1] * tx;
    let bot = plane[row1 + x0] * (1.0 - tx) + plane[row1 + x1] * tx;
    top * (1.0 - ty) + bot * ty
}

fn compute_cropped_output_dimensions(native_dims: (u32, u32), crop: CropTransform) -> (u32, u32) {
    let (mut w, mut h) = native_dims;
    if w == 0 || h == 0 {
        return (1, 1);
    }
    if crop.quarter_turns % 2 == 1 {
        std::mem::swap(&mut w, &mut h);
    }
    if crop.rect.is_full_frame() {
        return (w, h);
    }
    let x = ((crop.rect.left * w as f32).round() as u32).min(w.saturating_sub(1));
    let y = ((crop.rect.top * h as f32).round() as u32).min(h.saturating_sub(1));
    let cw = ((crop.rect.width * w as f32).round() as u32)
        .max(1)
        .min(w.saturating_sub(x));
    let ch = ((crop.rect.height * h as f32).round() as u32)
        .max(1)
        .min(h.saturating_sub(y));
    (cw, ch)
}

fn scale_sample_encoded_bytes(
    sample_bytes: u64,
    sample_pixels: u64,
    full_output_pixels: u64,
    format: OutputImageFormat,
) -> u64 {
    if sample_pixels == 0 || full_output_pixels <= sample_pixels {
        return sample_bytes;
    }
    let header_bytes: u64 = match format {
        OutputImageFormat::Jpeg { .. } => 600,
        OutputImageFormat::Png => 128,
        OutputImageFormat::Tiff => 4096,
    };
    let payload = sample_bytes.saturating_sub(header_bytes) as f64;
    let area_ratio = (full_output_pixels as f64) / (sample_pixels as f64);
    ((payload * area_ratio).round() as u64).saturating_add(header_bytes)
}

#[cfg(test)]
fn encode_graded_bytes(graded: Rgb32FImage, format: OutputImageFormat) -> Result<u64, AppError> {
    encode_graded_bytes_with_alpha(graded, None, format)
}

fn encode_graded_bytes_with_alpha(
    graded: Rgb32FImage,
    transparent_alpha: Option<&[f32]>,
    format: OutputImageFormat,
) -> Result<u64, AppError> {
    let mut buf = Cursor::new(Vec::<u8>::new());
    match format {
        OutputImageFormat::Jpeg { quality } => {
            let rgb8 = finalize_rgb8(graded);
            let mut encoder = JpegEncoder::new_with_quality(&mut buf, quality.clamp(1, 100));
            encoder.encode_image(&rgb8).map_err(|err| {
                AppError::ImageProcessingFailure(format!("JPEG encode failed: {err}"))
            })?;
        }
        OutputImageFormat::Png => {
            if let Some(alpha) = transparent_alpha {
                let rgba8 = finalize_rgba8(&graded, alpha);
                DynamicImage::ImageRgba8(rgba8)
                    .write_to(&mut buf, ImageFormat::Png)
                    .map_err(|err| {
                        AppError::ImageProcessingFailure(format!("PNG RGBA encode failed: {err}"))
                    })?;
            } else {
                let rgb8 = finalize_rgb8(graded);
                DynamicImage::ImageRgb8(rgb8)
                    .write_to(&mut buf, ImageFormat::Png)
                    .map_err(|err| {
                        AppError::ImageProcessingFailure(format!("PNG encode failed: {err}"))
                    })?;
            }
        }
        OutputImageFormat::Tiff => {
            let rgb16 = DynamicImage::ImageRgb32F(graded).to_rgb16();
            DynamicImage::ImageRgb16(rgb16)
                .write_to(&mut buf, ImageFormat::Tiff)
                .map_err(|err| {
                    AppError::ImageProcessingFailure(format!("TIFF encode failed: {err}"))
                })?;
        }
    }
    Ok(buf.into_inner().len() as u64)
}

fn composite_over_white_f32(img: &mut Rgb32FImage, alpha: &[f32]) {
    let w = img.width() as usize;
    let row_stride = w * 3;
    img.as_mut()
        .par_chunks_exact_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let a_base = y * w;
            for x in 0..w {
                let a = alpha[a_base + x].clamp(0.0, 1.0);
                let inv = 1.0 - a;
                let idx = x * 3;
                row[idx] = (row[idx] * a + inv).clamp(0.0, 1.0);
                row[idx + 1] = (row[idx + 1] * a + inv).clamp(0.0, 1.0);
                row[idx + 2] = (row[idx + 2] * a + inv).clamp(0.0, 1.0);
            }
        });
}

fn finalize_rgba8(graded: &Rgb32FImage, alpha: &[f32]) -> RgbaImage {
    let (w, h) = graded.dimensions();
    let w_usize = w as usize;
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        let a_base = (y as usize) * w_usize;
        for x in 0..w {
            let [r, g, b] = graded.get_pixel(x, y).0;
            let a = alpha[a_base + x as usize].clamp(0.0, 1.0);
            out.put_pixel(
                x,
                y,
                Rgba([
                    (r.clamp(0.0, 1.0) * 255.0).round() as u8,
                    (g.clamp(0.0, 1.0) * 255.0).round() as u8,
                    (b.clamp(0.0, 1.0) * 255.0).round() as u8,
                    (a * 255.0).round() as u8,
                ]),
            );
        }
    }
    out
}

fn save_graded_output(
    mut graded: Rgb32FImage,
    transparent_alpha: Option<&[f32]>,
    output_path: &str,
    format: OutputImageFormat,
) -> Result<(), AppError> {
    let path = Path::new(output_path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to create output directory '{}': {err}",
                parent.display()
            ))
        })?;
    }

    match format {
        OutputImageFormat::Jpeg { quality } => {
            if let Some(alpha) = transparent_alpha {
                composite_over_white_f32(&mut graded, alpha);
            }
            let rgb8 = finalize_rgb8(graded);
            let file = File::create(path).map_err(|err| {
                AppError::StorageFailure(format!("failed to create '{output_path}': {err}"))
            })?;
            let writer = BufWriter::new(file);
            let mut encoder = JpegEncoder::new_with_quality(writer, quality.clamp(1, 100));
            encoder.encode_image(&rgb8).map_err(|err| {
                AppError::ImageProcessingFailure(format!("JPEG encode failed: {err}"))
            })
        }
        OutputImageFormat::Png => {
            if let Some(alpha) = transparent_alpha {
                let rgba8 = finalize_rgba8(&graded, alpha);
                DynamicImage::ImageRgba8(rgba8)
                    .save_with_format(path, ImageFormat::Png)
                    .map_err(|err| {
                        AppError::ImageProcessingFailure(format!("PNG RGBA encode failed: {err}"))
                    })
            } else {
                let rgb8 = finalize_rgb8(graded);
                DynamicImage::ImageRgb8(rgb8)
                    .save_with_format(path, ImageFormat::Png)
                    .map_err(|err| {
                        AppError::ImageProcessingFailure(format!("PNG encode failed: {err}"))
                    })
            }
        }
        OutputImageFormat::Tiff => {
            if let Some(alpha) = transparent_alpha {
                composite_over_white_f32(&mut graded, alpha);
            }
            // Preserve true 16-bit color depth when exporting TIFF deliverables from RAW sources
            let rgb16 = DynamicImage::ImageRgb32F(graded).to_rgb16();
            DynamicImage::ImageRgb16(rgb16)
                .save_with_format(path, ImageFormat::Tiff)
                .map_err(|err| AppError::ImageProcessingFailure(format!("TIFF encode failed: {err}")))
        }
    }
}

fn apply_crop_and_rotation_f32(source: &Rgb32FImage, crop: CropTransform) -> Rgb32FImage {
    if crop.is_identity() {
        return source.clone();
    }

    let mut working = match crop.quarter_turns % 4 {
        1 => imageops::rotate90(source),
        2 => imageops::rotate180(source),
        3 => imageops::rotate270(source),
        _ => source.clone(),
    };
    if crop.flip_horizontal {
        working = imageops::flip_horizontal(&working);
    }
    if crop.flip_vertical {
        working = imageops::flip_vertical(&working);
    }
    if crop.rotation_deg.abs() > 1e-3 {
        working = rotate_bilinear_f32(&working, crop.rotation_deg);
    }

    if crop.rect.is_full_frame() {
        return working;
    }

    let (w, h) = working.dimensions();
    let x = ((crop.rect.left * w as f32).round() as u32).min(w.saturating_sub(1));
    let y = ((crop.rect.top * h as f32).round() as u32).min(h.saturating_sub(1));
    let cw = ((crop.rect.width * w as f32).round() as u32)
        .max(1)
        .min(w.saturating_sub(x));
    let ch = ((crop.rect.height * h as f32).round() as u32)
        .max(1)
        .min(h.saturating_sub(y));

    imageops::crop_imm(&working, x, y, cw, ch).to_image()
}

fn rotate_bilinear_f32(source: &Rgb32FImage, angle_deg: f32) -> Rgb32FImage {
    let (w, h) = source.dimensions();
    let mut out = Rgb32FImage::new(w, h);
    let rad = -angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();
    let cx = (w as f32 - 1.0) * 0.5;
    let cy = (h as f32 - 1.0) * 0.5;

    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let sx = dx * cos_a - dy * sin_a + cx;
            let sy = dx * sin_a + dy * cos_a + cy;

            if sx >= 0.0 && sy >= 0.0 && sx < (w as f32 - 1.0) && sy < (h as f32 - 1.0) {
                let x0 = sx.floor() as u32;
                let y0 = sy.floor() as u32;
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let p00 = source.get_pixel(x0, y0).0;
                let p10 = source.get_pixel(x0 + 1, y0).0;
                let p01 = source.get_pixel(x0, y0 + 1).0;
                let p11 = source.get_pixel(x0 + 1, y0 + 1).0;

                let mut rgb = [0.0_f32; 3];
                for c in 0..3 {
                    let top = p00[c] * (1.0 - fx) + p10[c] * fx;
                    let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
                    rgb[c] = (top * (1.0 - fy) + bot * fy).max(0.0);
                }
                out.put_pixel(x, y, Rgb(rgb));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_rgb_image_brightens_and_crops_cleanly() {
        let mut img = RgbImage::new(16, 16);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([100, 100, 100]);
        }

        let mut recipe = DevelopRecipe::default();
        recipe.basic_tone.exposure_ev = 1.0;
        recipe.crop_transform.rect = NormalizedRect::new(0.25, 0.25, 0.5, 0.5).unwrap();

        let engine = RayonImageDevelopEngine::new();
        let out = engine.process_rgb_image(&img, &recipe).expect("processed");

        assert_eq!(out.dimensions(), (8, 8));
        assert!(out.get_pixel(4, 4).0[0] > 130);
    }

    #[test]
    fn encode_graded_bytes_measures_realistic_relative_format_sizes() {
        let mut img = Rgb32FImage::new(64, 64);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let r = ((x * 7 + y * 13) % 255) as f32 / 255.0;
            let g = ((x * 11 + y * 5) % 255) as f32 / 255.0;
            let b = ((x * 3 + y * 17) % 255) as f32 / 255.0;
            *pixel = Rgb([r, g, b]);
        }

        let jpeg70 = encode_graded_bytes(img.clone(), OutputImageFormat::Jpeg { quality: 70 })
            .expect("jpeg70");
        let jpeg100 =
            encode_graded_bytes(img.clone(), OutputImageFormat::Jpeg { quality: 100 })
                .expect("jpeg100");
        let png = encode_graded_bytes(img.clone(), OutputImageFormat::Png).expect("png");
        let tiff = encode_graded_bytes(img, OutputImageFormat::Tiff).expect("tiff");

        assert!(jpeg70 > 0);
        assert!(jpeg100 > jpeg70);
        assert!(png > 0);
        assert!(tiff > jpeg100);
    }

    #[test]
    fn spatial_clarity_enhances_midtone_edge_contrast_and_nr_smooths_noise() {
        let mut edge = RgbImage::new(48, 48);
        for (x, _y, px) in edge.enumerate_pixels_mut() {
            let v = if x < 24 { 95 } else { 155 };
            *px = Rgb([v, v, v]);
        }

        let engine = RayonImageDevelopEngine::new();
        let mut clarity_recipe = DevelopRecipe::default();
        clarity_recipe.basic_tone.clarity = 80;
        let crisp = engine.process_rgb_image(&edge, &clarity_recipe).expect("clarity");
        // Just across the midtone boundary, positive clarity deepens the dark side and lifts the bright side
        assert!(crisp.get_pixel(22, 24).0[0] < 95);
        assert!(crisp.get_pixel(25, 24).0[0] > 155);

        let mut noisy = RgbImage::new(24, 24);
        for (x, y, px) in noisy.enumerate_pixels_mut() {
            let v = if (x + y) % 2 == 0 { 118 } else { 136 };
            *px = Rgb([v, v, v]);
        }
        let mut nr_recipe = DevelopRecipe::default();
        nr_recipe.detail_optics.luminance_noise_reduction = 90;
        let denoised = engine.process_rgb_image(&noisy, &nr_recipe).expect("nr");
        let diff_before = (136_i16 - 118_i16).abs();
        let diff_after =
            (denoised.get_pixel(12, 12).0[0] as i16 - denoised.get_pixel(13, 12).0[0] as i16).abs();
        assert!(diff_after < diff_before);
    }

    #[test]
    fn export_size_estimator_scales_area_accurately_and_handles_tiff_crop_and_preview() {
        let mut crop = CropTransform::default();
        assert_eq!(compute_cropped_output_dimensions((6000, 4000), crop), (6000, 4000));

        crop.quarter_turns = 1;
        assert_eq!(compute_cropped_output_dimensions((6000, 4000), crop), (4000, 6000));

        crop.rect = NormalizedRect::new(0.25, 0.25, 0.5, 0.5).unwrap();
        assert_eq!(compute_cropped_output_dimensions((6000, 4000), crop), (2000, 3000));

        // No downscaling -> exact bytes returned
        assert_eq!(
            scale_sample_encoded_bytes(12_000, 1000, 1000, OutputImageFormat::Jpeg { quality: 92 }),
            12_000
        );
        // 4x pixel area -> ~4x payload + fixed header
        let scaled = scale_sample_encoded_bytes(
            10_600,
            1_000_000,
            4_000_000,
            OutputImageFormat::Jpeg { quality: 92 },
        );
        assert_eq!(scaled, 40_600);

        // End-to-end lightweight measurement on a real image (with and without preview path)
        let pid = std::process::id();
        let src_path = std::env::temp_dir().join(format!("kroma_est_src_{pid}.png"));
        let prev_path = std::env::temp_dir().join(format!("kroma_est_prev_{pid}.jpg"));

        let mut img = RgbImage::new(240, 160);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = Rgb([
                ((x * 7 + y * 3) % 255) as u8,
                ((x * 5 + y * 11) % 255) as u8,
                ((x * 13 + y * 2) % 255) as u8,
            ]);
        }
        img.save(&src_path).expect("save src");
        img.save(&prev_path).expect("save preview");

        let engine = RayonImageDevelopEngine::new();
        let default_recipe = DevelopRecipe::default();
        let src_str = src_path.to_str().unwrap();
        let prev_str = prev_path.to_str().unwrap();

        let jpeg80 = engine
            .measure_rendered_bytes_with_preview(
                src_str,
                Some(prev_str),
                &default_recipe,
                OutputImageFormat::Jpeg { quality: 80 },
            )
            .expect("jpeg80");
        let jpeg98 = engine
            .measure_rendered_bytes_with_preview(
                src_str,
                Some(prev_str),
                &default_recipe,
                OutputImageFormat::Jpeg { quality: 98 },
            )
            .expect("jpeg98");
        let tiff_full = engine
            .measure_rendered_bytes(src_str, &default_recipe, OutputImageFormat::Tiff)
            .expect("tiff full");

        let mut cropped_recipe = DevelopRecipe::default();
        cropped_recipe.crop_transform.rect = NormalizedRect::new(0.25, 0.25, 0.5, 0.5).unwrap();
        let tiff_cropped = engine
            .measure_rendered_bytes(src_str, &cropped_recipe, OutputImageFormat::Tiff)
            .expect("tiff cropped");

        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&prev_path);

        assert!(jpeg98 > jpeg80);
        assert_eq!(tiff_full, 240 * 160 * 6 + 4096);
        assert_eq!(tiff_cropped, 120 * 80 * 6 + 4096);
    }
}
