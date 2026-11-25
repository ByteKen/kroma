//! LaMa Neural Generative Object Removal Engine + Pyramidal Harmonic Fallback (Phase 3 Opt-In).
//!
//! Runs `lama_fp32.onnx` (or any compatible 2-input `image [1, 3, H, W]` + `mask [1, 1, H, W]`
//! LaMa ONNX model) via ONNX Runtime DirectML/CoreML GPU or CPU fallback on a generous square
//! context crop around the user's painted strokes, then returns a compact non-destructive
//! `InpaintPatch` (tight padded PNG RGB + PackBits RLE soft feather alpha mask) that composites
//! identically across WebGPU preview, WebGL2 fallback, Rust `wgpu` export, and Rayon CPU export.

use crate::vision::onnx_runtime::shared_ort_environment;
use image::codecs::png::PngEncoder;
use image::imageops::{self, FilterType};
use image::{ColorType, ImageEncoder, Rgb, RgbImage};
use kroma_app::error::AppError;
use kroma_domain::develop::NormalizedRect;
use kroma_domain::masking::{
    base64_decode, base64_encode, decode_matte_rle_b64, encode_matte_rle_b64,
};
use kroma_domain::retouch::InpaintPatch;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_LAMA_DIM: u32 = 512;
const MAX_PATCH_OUTPUT_DIM: u32 = 512;

/// User-painted brush stroke in oriented full-frame normalized UV `[0..1]`.
#[derive(Debug, Clone)]
pub struct InpaintStrokeInput {
    /// Brush radius as a fraction of the shorter oriented image side (`0.004..=0.25`).
    pub radius: f32,
    /// Polyline points `[u, v]` in oriented full-frame UV `[0..1]`.
    pub points: Vec<[f32; 2]>,
}

#[derive(Debug, Clone)]
pub struct GeneratedInpaintPatch {
    pub patch: InpaintPatch,
    pub elapsed_ms: u64,
}

struct LamaSessionState {
    session: Session,
    using_gpu: bool,
}

/// Real ONNX Runtime adapter for LaMa (`lama_fp32.onnx` / `inpainting_lama_2025jan.onnx`).
pub struct OnnxLamaInpainter {
    state: Mutex<LamaSessionState>,
    model_path: PathBuf,
    intra_threads: usize,
    input_w: u32,
    input_h: u32,
    image_input_name: String,
    mask_input_name: String,
    model_label: String,
}

/// Windows DirectML (`DML_GEMM_OPERATOR_DESC`) supports at most 4D tensors for `MatMul`, whereas
/// LaMa's Fast Fourier Convolution (`ffc/convg2g/fu/rttn/MatMul`) emits 5D+ complex-DFT `MatMul`
/// nodes that load cleanly but fail at runtime with `HRESULT 0x80070057 (E_INVALIDARG)`.
/// Skip DirectML up front for known LaMa FFC models on Windows to avoid a ~400 MB wasted VRAM load
/// and runtime crash; custom non-LaMa models still try GPU and fall back automatically on error.
fn should_skip_directml_for_model(path: &Path) -> bool {
    #[cfg(target_os = "windows")]
    {
        path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|name| name.to_ascii_lowercase().contains("lama"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = path;
        false
    }
}

fn run_lama_session(
    session: &mut Session,
    image_input_name: &str,
    mask_input_name: &str,
    ih: usize,
    iw: usize,
    img_tensor_data: Vec<f32>,
    mask_tensor_data: Vec<f32>,
) -> Result<(usize, usize, Vec<f32>), AppError> {
    let img_tensor =
        Tensor::from_array(([1usize, 3, ih, iw], img_tensor_data)).map_err(|err| {
            AppError::ImageProcessingFailure(format!("LaMa image tensor error: {err}"))
        })?;
    let mask_tensor =
        Tensor::from_array(([1usize, 1, ih, iw], mask_tensor_data)).map_err(|err| {
            AppError::ImageProcessingFailure(format!("LaMa mask tensor error: {err}"))
        })?;

    let outputs = session
        .run(ort::inputs![
            image_input_name => img_tensor,
            mask_input_name => mask_tensor
        ])
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!("LaMa ONNX inference failed: {err}"))
        })?;

    if outputs.len() == 0 {
        return Err(AppError::ImageProcessingFailure(
            "LaMa ONNX model returned zero output tensors".to_string(),
        ));
    }

    let (shape, raw) = outputs[0].try_extract_tensor::<f32>().map_err(|err| {
        AppError::ImageProcessingFailure(format!("LaMa output extraction failed: {err}"))
    })?;
    let dims: Vec<usize> = shape.iter().map(|&d| d.max(1) as usize).collect();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(AppError::ImageProcessingFailure(format!(
            "unexpected LaMa output tensor shape: {dims:?}"
        )));
    }
    let oh = dims[2];
    let ow = dims[3];
    let out_plane = oh * ow;
    if raw.len() < 3 * out_plane {
        return Err(AppError::ImageProcessingFailure(
            "LaMa output buffer shorter than tensor shape".to_string(),
        ));
    }

    Ok((oh, ow, raw[..3 * out_plane].to_vec()))
}

impl OnnxLamaInpainter {
    pub fn from_model_file(
        model_path: impl AsRef<Path>,
        intra_threads: usize,
    ) -> Result<Self, AppError> {
        let path_ref = model_path.as_ref();
        let env = shared_ort_environment()?;
        let effective_threads = intra_threads.max(1);

        let (session, using_gpu) = if should_skip_directml_for_model(path_ref) {
            (build_cpu_session(env, path_ref, effective_threads)?, false)
        } else {
            match crate::vision::onnx_runtime::try_build_gpu_session(env, path_ref) {
                Some(gpu_session) => (gpu_session, true),
                None => (build_cpu_session(env, path_ref, effective_threads)?, false),
            }
        };

        let (image_input_name, mask_input_name, input_h, input_w) = detect_lama_io(&session)?;
        let model_label = friendly_lama_label(path_ref);

        Ok(Self {
            state: Mutex::new(LamaSessionState { session, using_gpu }),
            model_path: path_ref.to_path_buf(),
            intra_threads: effective_threads,
            input_w,
            input_h,
            image_input_name,
            mask_input_name,
            model_label,
        })
    }

    pub fn engine_label(&self) -> String {
        let using_gpu = self.state.lock().map(|s| s.using_gpu).unwrap_or(false);
        let ep = if using_gpu {
            crate::vision::onnx_runtime::gpu_execution_provider_label()
        } else {
            "CPU"
        };
        format!("{} ({ep})", self.model_label)
    }

    /// Native square inference resolution expected by the loaded ONNX model (e.g. `512`).
    pub fn inference_dim(&self) -> u32 {
        self.input_w.min(self.input_h).max(128)
    }

    /// Runs LaMa neural inpainting on `crop_rgb` with binary `mask_u8` (`> 96` = erase).
    fn inpaint_crop(&self, crop_rgb: &RgbImage, mask_u8: &[u8]) -> Result<RgbImage, AppError> {
        let (cw, ch) = crop_rgb.dimensions();
        let iw = self.input_w as usize;
        let ih = self.input_h as usize;
        let plane = iw * ih;

        let owned_resized;
        let sample_rgb: &RgbImage = if cw == self.input_w && ch == self.input_h {
            crop_rgb
        } else {
            owned_resized = imageops::resize(crop_rgb, self.input_w, self.input_h, FilterType::CatmullRom);
            &owned_resized
        };

        let mut img_tensor_data = vec![0.0_f32; 3 * plane];
        let mut mask_tensor_data = vec![0.0_f32; plane];

        for y in 0..ih {
            let sy = if ih == ch as usize {
                y
            } else {
                ((y as f32 + 0.5) * ch as f32 / ih as f32)
                    .floor()
                    .clamp(0.0, (ch - 1) as f32) as usize
            };
            for x in 0..iw {
                let sx = if iw == cw as usize {
                    x
                } else {
                    ((x as f32 + 0.5) * cw as f32 / iw as f32)
                        .floor()
                        .clamp(0.0, (cw - 1) as f32) as usize
                };
                let [r8, g8, b8] = sample_rgb.get_pixel(x as u32, y as u32).0;
                let off = y * iw + x;
                let m = if mask_u8[sy * cw as usize + sx] > 96 {
                    1.0_f32
                } else {
                    0.0_f32
                };
                mask_tensor_data[off] = m;
                // Zero-mask erased region input pixels cleanly while normalizing to [0..1]
                let keep = 1.0 - m;
                img_tensor_data[off] = (r8 as f32 / 255.0) * keep;
                img_tensor_data[plane + off] = (g8 as f32 / 255.0) * keep;
                img_tensor_data[2 * plane + off] = (b8 as f32 / 255.0) * keep;
            }
        }

        let mut state = self.state.lock().map_err(|_| {
            AppError::ImageProcessingFailure("LaMa ONNX session mutex poisoned".to_string())
        })?;

        let (oh, ow, raw) = if state.using_gpu {
            match run_lama_session(
                &mut state.session,
                &self.image_input_name,
                &self.mask_input_name,
                ih,
                iw,
                img_tensor_data.clone(),
                mask_tensor_data.clone(),
            ) {
                Ok(out) => out,
                Err(gpu_err) => {
                    // Runtime GPU EP failure (e.g., DirectML/CoreML rejecting FFC high-rank ops):
                    // transparently switch this cached inpainter to CPU ORT and retry seamlessly.
                    eprintln!(
                        "[kroma] LaMa GPU inference failed ({gpu_err}); falling back to CPU ONNX Runtime"
                    );
                    let env = shared_ort_environment()?;
                    let cpu_session = build_cpu_session(env, &self.model_path, self.intra_threads)?;
                    state.session = cpu_session;
                    state.using_gpu = false;
                    run_lama_session(
                        &mut state.session,
                        &self.image_input_name,
                        &self.mask_input_name,
                        ih,
                        iw,
                        img_tensor_data,
                        mask_tensor_data,
                    )?
                }
            }
        } else {
            run_lama_session(
                &mut state.session,
                &self.image_input_name,
                &self.mask_input_name,
                ih,
                iw,
                img_tensor_data,
                mask_tensor_data,
            )?
        };
        drop(state);

        let out_plane = oh * ow;

        // Auto-detect [0..255] vs [0..1] output scaling across Carve/OpenCV LaMa ONNX variants
        let mut max_val = 0.0_f32;
        for &v in raw.iter().step_by(17) {
            if v.is_finite() && v > max_val {
                max_val = v;
            }
        }
        let scale = if max_val > 1.5 { 1.0 } else { 255.0 };

        let mut out_img = RgbImage::new(ow as u32, oh as u32);
        for y in 0..oh {
            for x in 0..ow {
                let off = y * ow + x;
                let r = (raw[off] * scale).clamp(0.0, 255.0).round() as u8;
                let g = (raw[out_plane + off] * scale).clamp(0.0, 255.0).round() as u8;
                let b = (raw[2 * out_plane + off] * scale)
                    .clamp(0.0, 255.0)
                    .round() as u8;
                out_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        if ow as u32 == cw && oh as u32 == ch {
            Ok(out_img)
        } else {
            Ok(imageops::resize(&out_img, cw, ch, FilterType::CatmullRom))
        }
    }
}

/// Generates a compact, non-destructive `InpaintPatch` for painted `strokes` on oriented `rgb`.
/// Uses `inpainter` (`lama_fp32.onnx`) when installed, or falls back cleanly to Kroma's fast
/// multi-scale pyramidal harmonic fill so the workflow stays functional even offline.
pub fn generate_inpaint_patch(
    rgb: &RgbImage,
    strokes: &[InpaintStrokeInput],
    inpainter: Option<&OnnxLamaInpainter>,
) -> Result<GeneratedInpaintPatch, AppError> {
    let started = Instant::now();
    let (img_w, img_h) = rgb.dimensions();
    if img_w < 8 || img_h < 8 {
        return Err(AppError::ImageProcessingFailure(
            "image too small for generative remove".to_string(),
        ));
    }

    let valid_strokes: Vec<&InpaintStrokeInput> = strokes
        .iter()
        .filter(|s| s.radius >= 0.002 && !s.points.is_empty())
        .collect();
    if valid_strokes.is_empty() {
        return Err(AppError::ImageProcessingFailure(
            "paint over an object before running AI Remove".to_string(),
        ));
    }

    let fw = img_w as f32;
    let fh = img_h as f32;
    let min_side = fw.min(fh);

    // 1. Tight bounding box of painted strokes + bounded soft edge feather + untouched perimeter
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut max_r_norm = 0.004_f32;
    for s in &valid_strokes {
        let r_norm = s.radius.clamp(0.003, 0.25);
        max_r_norm = max_r_norm.max(r_norm);
        let r_px = (r_norm * min_side).max(3.0);
        for pt in &s.points {
            let px = pt[0].clamp(0.0, 1.0) * fw;
            let py = pt[1].clamp(0.0, 1.0) * fh;
            min_x = min_x.min(px - r_px);
            min_y = min_y.min(py - r_px);
            max_x = max_x.max(px + r_px);
            max_y = max_y.max(py + r_px);
        }
    }

    let raw_w = (max_x - min_x).max(8.0);
    let raw_h = (max_y - min_y).max(8.0);
    let feather_px = compute_inpaint_feather_norm(max_r_norm, min_side) * min_side;
    let bin_margin_px = compute_inpaint_binary_margin_norm(min_side) * min_side;
    // Leave a clean untouched outer perimeter beyond the binary mask so all 8 border adaptation
    // taps (at 2% inset) in `inpaint_pass.wgsl` sample untouched pixels at every patch size.
    let outer_margin_px = (raw_w.max(raw_h) * 0.035).clamp(10.0, 28.0);
    let pad_px = feather_px + bin_margin_px + outer_margin_px;
    let px0 = (min_x - pad_px).floor().clamp(0.0, (fw - 8.0).max(0.0)) as u32;
    let py0 = (min_y - pad_px).floor().clamp(0.0, (fh - 8.0).max(0.0)) as u32;
    let px1 = (max_x + pad_px).ceil().clamp((px0 + 8) as f32, fw) as u32;
    let py1 = (max_y + pad_px).ceil().clamp((py0 + 8) as f32, fh) as u32;
    let tight_w = (px1 - px0).max(8);
    let tight_h = (py1 - py0).max(8);

    // 2. Strictly square aspect-preserving context window (shifted inward at borders, reflect-padded
    // only when larger than a shorter image side) resized in ONE CatmullRom step to model resolution.
    let (ctx_x0, ctx_y0, ctx_side) =
        compute_square_context_window(px0, py0, px1, py1, img_w, img_h);
    let work_dim = match inpainter {
        Some(onnx) => onnx.inference_dim(),
        None => ctx_side.clamp(16, DEFAULT_LAMA_DIM),
    };

    let ctx_rgb = extract_square_context_rgb(rgb, ctx_x0, ctx_y0, ctx_side);
    let work_rgb = if work_dim == ctx_side {
        ctx_rgb
    } else {
        imageops::resize(&ctx_rgb, work_dim, work_dim, FilterType::CatmullRom)
    };

    // Rasterize binary inpaint mask extending cleanly beyond the tight soft feather seam so LaMa
    // erases edge fringes before the soft mask blends into untouched background pixels.
    let (binary_mask, _soft_ctx_mask) = rasterize_strokes_in_rect(
        &valid_strokes,
        ctx_x0 as f32 / fw,
        ctx_y0 as f32 / fh,
        ctx_side as f32 / fw,
        ctx_side as f32 / fh,
        fw,
        fh,
        work_dim,
        work_dim,
    );

    // 3. Run LaMa Neural ONNX (1:1 at native 512×512 with zero extra resize) or Harmonic Fill
    let (inpainted_work, engine_label) = match inpainter {
        Some(onnx) => (onnx.inpaint_crop(&work_rgb, &binary_mask)?, "LaMa AI".to_string()),
        None => (
            inpaint_harmonic_pyramidal(&work_rgb, &binary_mask),
            "Harmonic Fill".to_string(),
        ),
    };

    // 4. Extract tight `patch_bounds` crop via exact sub-pixel CatmullRom sampling (zero integer
    // rounding shift) and restore exact untouched source pixels outside the binary mask so GPU
    // 8-tap perimeter adaptation starts at 0.0 color error on high-contrast edges.
    let out_scale = if tight_w.max(tight_h) > MAX_PATCH_OUTPUT_DIM {
        MAX_PATCH_OUTPUT_DIM as f32 / tight_w.max(tight_h) as f32
    } else {
        1.0
    };
    let patch_w = ((tight_w as f32 * out_scale).round() as u32).clamp(8, MAX_PATCH_OUTPUT_DIM);
    let patch_h = ((tight_h as f32 * out_scale).round() as u32).clamp(8, MAX_PATCH_OUTPUT_DIM);

    let bounds = NormalizedRect {
        left: px0 as f32 / fw,
        top: py0 as f32 / fh,
        width: tight_w as f32 / fw,
        height: tight_h as f32 / fh,
    };

    let (bin_patch, soft_patch_mask) = rasterize_strokes_in_rect(
        &valid_strokes,
        bounds.left,
        bounds.top,
        bounds.width,
        bounds.height,
        fw,
        fh,
        patch_w,
        patch_h,
    );

    let mut final_patch_rgb = compose_aligned_patch_rgb(
        rgb,
        &inpainted_work,
        &bin_patch,
        PatchRectPx {
            x0: px0,
            y0: py0,
            w: tight_w,
            h: tight_h,
        },
        SquareContextPx {
            x0: ctx_x0,
            y0: ctx_y0,
            side: ctx_side,
        },
        patch_w,
        patch_h,
    );

    if inpainter.is_some() {
        let grain_sigma = estimate_border_luma_grain_sigma(&final_patch_rgb, &bin_patch);
        apply_matched_micro_grain(&mut final_patch_rgb, &soft_patch_mask, grain_sigma);
    }

    let mut png_bytes = Vec::new();
    PngEncoder::new(&mut png_bytes)
        .write_image(
            final_patch_rgb.as_raw(),
            patch_w,
            patch_h,
            ColorType::Rgb8.into(),
        )
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!("failed encoding inpaint patch PNG: {err}"))
        })?;

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let mut patch = InpaintPatch {
        id: format!("inpaint-{stamp}"),
        bounds,
        patch_width: patch_w as u16,
        patch_height: patch_h as u16,
        opacity: 1.0,
        mask_rle_b64: encode_matte_rle_b64(&soft_patch_mask),
        patch_image_b64: base64_encode(&png_bytes),
        engine_label,
    };
    patch.clamp_invariants();

    Ok(GeneratedInpaintPatch {
        patch,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// Decodes a persisted `InpaintPatch` into tightly packed RGBA8 (`RGB = patch`, `A = soft mask`)
/// for Rust CPU / `wgpu` rendering.
pub fn decode_inpaint_patch_rgba(patch: &InpaintPatch) -> Option<(u32, u32, Vec<u8>)> {
    if !patch.is_effective() {
        return None;
    }
    let w = patch.patch_width as u32;
    let h = patch.patch_height as u32;
    let mask = decode_matte_rle_b64(&patch.mask_rle_b64, (w as usize) * (h as usize))?;
    let raw_img_bytes = base64_decode(patch.patch_image_b64.trim())?;
    let decoded = image::load_from_memory(&raw_img_bytes).ok()?.to_rgb8();
    let rgb = if decoded.width() == w && decoded.height() == h {
        decoded
    } else {
        imageops::resize(&decoded, w, h, FilterType::Triangle)
    };
    let pixel_count = (w as usize) * (h as usize);
    let raw_rgb = rgb.as_raw();
    let mut rgba = vec![0u8; pixel_count * 4];
    for (i, &alpha) in mask.iter().enumerate().take(pixel_count) {
        let p = i * 3;
        let o = i * 4;
        rgba[o] = raw_rgb[p];
        rgba[o + 1] = raw_rgb[p + 1];
        rgba[o + 2] = raw_rgb[p + 2];
        rgba[o + 3] = alpha;
    }
    Some((w, h, rgba))
}

#[derive(Debug, Clone, Copy)]
struct PatchRectPx {
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
}

#[derive(Debug, Clone, Copy)]
struct SquareContextPx {
    x0: i32,
    y0: i32,
    side: u32,
}

/// Computes a strictly square `(x0, y0, side)` context crop centered on `[px0..px1] × [py0..py1]`.
/// Shifts inward to stay inside `[0, img_w] × [0, img_h]` when the square fits, and only extends
/// past image borders (to be mirror-padded) when `side` exceeds a shorter image dimension.
fn compute_square_context_window(
    px0: u32,
    py0: u32,
    px1: u32,
    py1: u32,
    img_w: u32,
    img_h: u32,
) -> (i32, i32, u32) {
    let tight_w = (px1.saturating_sub(px0)).max(8);
    let tight_h = (py1.saturating_sub(py0)).max(8);
    let tight_max = tight_w.max(tight_h) as f32;
    let fw = img_w.max(1) as f32;
    let fh = img_h.max(1) as f32;
    let min_side = fw.min(fh);

    // Give small distractions ~2.1× surrounding context while tightening larger masks toward
    // ~1.5× so LaMa's fixed 512×512 grid preserves ~2× more spatial detail inside the fill.
    let fill_ratio = (tight_max / min_side).clamp(0.0, 1.0);
    let ctx_mult = (2.15 - fill_ratio * 0.95).clamp(1.50, 2.15);
    let side = (tight_max * ctx_mult)
        .max(256.0)
        .max(tight_max)
        .min(fw.max(fh))
        .ceil() as u32;
    let side = side.max(tight_w.max(tight_h)).max(16);

    let cx = px0 as f32 + tight_w as f32 * 0.5;
    let cy = py0 as f32 + tight_h as f32 * 0.5;
    let half = side as f32 * 0.5;

    let mut x0 = if side <= img_w {
        (cx - half).round().clamp(0.0, (img_w - side) as f32) as i32
    } else {
        (cx - half).round() as i32
    };
    let mut y0 = if side <= img_h {
        (cy - half).round().clamp(0.0, (img_h - side) as f32) as i32
    } else {
        (cy - half).round() as i32
    };

    let side_i = side as i32;
    let px0_i = px0 as i32;
    let py0_i = py0 as i32;
    let px1_i = px1 as i32;
    let py1_i = py1 as i32;

    if x0 > px0_i {
        x0 = px0_i;
    }
    if x0 + side_i < px1_i {
        x0 = px1_i - side_i;
    }
    if y0 > py0_i {
        y0 = py0_i;
    }
    if y0 + side_i < py1_i {
        y0 = py1_i - side_i;
    }

    (x0, y0, side)
}

#[inline]
fn mirror_coord(mut v: i32, limit: u32) -> u32 {
    let lim = limit as i32;
    if lim <= 1 {
        return 0;
    }
    while v < 0 || v >= lim {
        if v < 0 {
            v = -v;
        }
        if v >= lim {
            v = 2 * (lim - 1) - v;
        }
    }
    v as u32
}

/// Extracts a strictly square `side × side` RGB context image using mirror-reflection only for
/// coordinates that extend beyond the source image edges.
fn extract_square_context_rgb(rgb: &RgbImage, ctx_x0: i32, ctx_y0: i32, side: u32) -> RgbImage {
    let (img_w, img_h) = rgb.dimensions();
    if ctx_x0 >= 0
        && ctx_y0 >= 0
        && (ctx_x0 as u32).saturating_add(side) <= img_w
        && (ctx_y0 as u32).saturating_add(side) <= img_h
    {
        return imageops::crop_imm(rgb, ctx_x0 as u32, ctx_y0 as u32, side, side).to_image();
    }

    RgbImage::from_fn(side, side, |x, y| {
        let sx = mirror_coord(ctx_x0 + x as i32, img_w);
        let sy = mirror_coord(ctx_y0 + y as i32, img_h);
        *rgb.get_pixel(sx, sy)
    })
}

#[inline]
fn catmull_rom_weight(t: f32) -> f32 {
    let a = t.abs();
    if a < 1.0 {
        (1.5 * a - 2.5) * a * a + 1.0
    } else if a < 2.0 {
        ((-0.5 * a + 2.5) * a - 4.0) * a + 2.0
    } else {
        0.0
    }
}

/// Exact sub-pixel Catmull-Rom bicubic sampler on an `RgbImage` (exact identity at integer coords).
fn sample_rgb_catmull_rom(img: &RgbImage, fx: f32, fy: f32) -> [u8; 3] {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let max_x = (w - 1).max(0);
    let max_y = (h - 1).max(0);
    let cx = fx.clamp(0.0, max_x as f32);
    let cy = fy.clamp(0.0, max_y as f32);
    let ix = cx.floor() as i32;
    let iy = cy.floor() as i32;
    let tx = cx - ix as f32;
    let ty = cy - iy as f32;

    let wx = [
        catmull_rom_weight(tx + 1.0),
        catmull_rom_weight(tx),
        catmull_rom_weight(tx - 1.0),
        catmull_rom_weight(tx - 2.0),
    ];
    let wy = [
        catmull_rom_weight(ty + 1.0),
        catmull_rom_weight(ty),
        catmull_rom_weight(ty - 1.0),
        catmull_rom_weight(ty - 2.0),
    ];

    let mut acc = [0.0_f32; 3];
    for (ky, &wy_k) in wy.iter().enumerate() {
        let sy = (iy + ky as i32 - 1).clamp(0, max_y) as u32;
        for (kx, &wx_k) in wx.iter().enumerate() {
            let sx = (ix + kx as i32 - 1).clamp(0, max_x) as u32;
            let w_tot = wx_k * wy_k;
            let px = img.get_pixel(sx, sy).0;
            acc[0] += px[0] as f32 * w_tot;
            acc[1] += px[1] as f32 * w_tot;
            acc[2] += px[2] as f32 * w_tot;
        }
    }

    [
        acc[0].clamp(0.0, 255.0).round() as u8,
        acc[1].clamp(0.0, 255.0).round() as u8,
        acc[2].clamp(0.0, 255.0).round() as u8,
    ]
}

fn sample_rgb_bilinear(img: &RgbImage, fx: f32, fy: f32) -> [u8; 3] {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let max_x = (w - 1).max(0);
    let max_y = (h - 1).max(0);
    let cx = fx.clamp(0.0, max_x as f32);
    let cy = fy.clamp(0.0, max_y as f32);
    let x0 = cx.floor() as i32;
    let y0 = cy.floor() as i32;
    let x1 = (x0 + 1).min(max_x);
    let y1 = (y0 + 1).min(max_y);
    let tx = cx - x0 as f32;
    let ty = cy - y0 as f32;

    let p00 = img.get_pixel(x0 as u32, y0 as u32).0;
    let p10 = img.get_pixel(x1 as u32, y0 as u32).0;
    let p01 = img.get_pixel(x0 as u32, y1 as u32).0;
    let p11 = img.get_pixel(x1 as u32, y1 as u32).0;
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 + (p10[c] as f32 - p00[c] as f32) * tx;
        let bot = p01[c] as f32 + (p11[c] as f32 - p01[c] as f32) * tx;
        out[c] = (top + (bot - top) * ty).clamp(0.0, 255.0).round() as u8;
    }
    out
}

/// Samples the tight patch directly from `inpainted_work` at exact sub-pixel coordinates inside
/// `bin_patch > 0`, while restoring untouched `source_rgb` pixels outside `bin_patch` so GPU
/// perimeter color-adaptation starts at exact 0.0 drift.
fn compose_aligned_patch_rgb(
    source_rgb: &RgbImage,
    inpainted_work: &RgbImage,
    bin_patch: &[u8],
    patch_rect: PatchRectPx,
    ctx: SquareContextPx,
    patch_w: u32,
    patch_h: u32,
) -> RgbImage {
    let work_w = inpainted_work.width() as f32;
    let work_h = inpainted_work.height() as f32;
    let ctx_side_f = ctx.side.max(1) as f32;
    let exact_src_1to1 = patch_w == patch_rect.w && patch_h == patch_rect.h;
    let pw_usize = patch_w as usize;

    RgbImage::from_fn(patch_w, patch_h, |px, py| {
        let idx = py as usize * pw_usize + px as usize;
        let img_x = patch_rect.x0 as f32 + ((px as f32 + 0.5) / patch_w as f32) * patch_rect.w as f32;
        let img_y = patch_rect.y0 as f32 + ((py as f32 + 0.5) / patch_h as f32) * patch_rect.h as f32;

        if bin_patch[idx] == 0 {
            if exact_src_1to1 {
                return *source_rgb.get_pixel(patch_rect.x0 + px, patch_rect.y0 + py);
            }
            return Rgb(sample_rgb_bilinear(source_rgb, img_x - 0.5, img_y - 0.5));
        }

        let wx = ((img_x - ctx.x0 as f32) / ctx_side_f) * work_w - 0.5;
        let wy = ((img_y - ctx.y0 as f32) / ctx_side_f) * work_h - 0.5;
        Rgb(sample_rgb_catmull_rom(inpainted_work, wx, wy))
    })
}

#[inline]
fn rgb8_luma(px: [u8; 3]) -> f32 {
    0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32
}

/// Estimates natural camera luminance micro-grain (`0..=4.5` in 8-bit units) strictly from
/// untouched perimeter pixels (`bin_patch == 0`) using a robust 35th-percentile Laplacian
/// estimator so high-contrast shirt/backdrop edges never inflate the grain level.
fn estimate_border_luma_grain_sigma(patch_rgb: &RgbImage, bin_patch: &[u8]) -> f32 {
    let w = patch_rgb.width() as usize;
    let h = patch_rgb.height() as usize;
    if w < 5 || h < 5 {
        return 0.0;
    }

    let mut residuals = Vec::with_capacity((w * h) / 4);
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            if bin_patch[idx] != 0
                || bin_patch[idx - 1] != 0
                || bin_patch[idx + 1] != 0
                || bin_patch[idx - w] != 0
                || bin_patch[idx + w] != 0
            {
                continue;
            }
            let c = rgb8_luma(patch_rgb.get_pixel(x as u32, y as u32).0);
            let l = rgb8_luma(patch_rgb.get_pixel((x - 1) as u32, y as u32).0);
            let r = rgb8_luma(patch_rgb.get_pixel((x + 1) as u32, y as u32).0);
            let u = rgb8_luma(patch_rgb.get_pixel(x as u32, (y - 1) as u32).0);
            let d = rgb8_luma(patch_rgb.get_pixel(x as u32, (y + 1) as u32).0);
            let res = (c - 0.25 * (l + r + u + d)).abs();
            residuals.push(res);
        }
    }

    if residuals.len() < 32 {
        return 0.0;
    }
    let nth = (residuals.len() * 35) / 100;
    let (_, p35, _) = residuals.select_nth_unstable_by(nth, |a, b| a.total_cmp(b));
    let sigma = (*p35 * 1.45).clamp(0.0, 4.5);
    if sigma < 0.45 {
        0.0
    } else {
        sigma
    }
}

#[inline]
fn hash_unit_f32(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(seed.wrapping_mul(2_246_822_519));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32)
}

/// Applies subtle zero-mean luminance-only micro-grain inside `soft_mask > 0` so LaMa's smooth
/// CNN output matches surrounding RAW skin/fabric grain without introducing color speckle.
fn apply_matched_micro_grain(patch_rgb: &mut RgbImage, soft_mask: &[u8], sigma: f32) {
    if sigma <= 0.0 {
        return;
    }
    let w = patch_rgb.width() as usize;
    let h = patch_rgb.height() as usize;
    for y in 0..h {
        for x in 0..w {
            let alpha = soft_mask[y * w + x];
            if alpha == 0 {
                continue;
            }
            let u1 = hash_unit_f32(x as u32, y as u32, 17);
            let u2 = hash_unit_f32(x as u32, y as u32, 53);
            let tri = (u1 + u2) - 1.0; // zero-mean triangular noise in [-1, 1]
            let delta = tri * sigma * (alpha as f32 / 255.0);
            let px = patch_rgb.get_pixel_mut(x as u32, y as u32);
            px.0[0] = (px.0[0] as f32 + delta).clamp(0.0, 255.0).round() as u8;
            px.0[1] = (px.0[1] as f32 + delta).clamp(0.0, 255.0).round() as u8;
            px.0[2] = (px.0[2] as f32 + delta).clamp(0.0, 255.0).round() as u8;
        }
    }
}

/// Bounded edge-feather width in `min_side`-normalized UV units:
/// stays soft on small spot strokes (`<= 28%` of radius) while capping tightly around ~10 px
/// (`0.4%..1.2%` of shorter image side) so large brushes keep a 100% opaque core over the painted
/// stroke instead of producing wide ghostly outlines.
#[inline]
fn compute_inpaint_feather_norm(radius_norm: f32, min_side_px: f32) -> f32 {
    let max_feather = (10.0 / min_side_px.max(1.0)).clamp(0.004, 0.012);
    (radius_norm * 0.28).min(max_feather)
}

/// Outward binary-mask safety dilation beyond the soft feather transition so LaMa erases edge
/// fringes before the soft seam blends into untouched background pixels.
#[inline]
fn compute_inpaint_binary_margin_norm(min_side_px: f32) -> f32 {
    (6.0 / min_side_px.max(1.0)).clamp(0.004, 0.010)
}

#[allow(clippy::too_many_arguments)]
fn rasterize_strokes_in_rect(
    strokes: &[&InpaintStrokeInput],
    rect_left: f32,
    rect_top: f32,
    rect_width: f32,
    rect_height: f32,
    img_w: f32,
    img_h: f32,
    out_w: u32,
    out_h: u32,
) -> (Vec<u8>, Vec<u8>) {
    let min_side = img_w.min(img_h).max(1.0);
    let sx = img_w / min_side;
    let sy = img_h / min_side;
    let bin_margin = compute_inpaint_binary_margin_norm(min_side);
    let ow = out_w as usize;
    let oh = out_h as usize;
    let mut binary = vec![0u8; ow * oh];
    let mut soft = vec![0u8; ow * oh];

    for y in 0..oh {
        let v = rect_top + ((y as f32 + 0.5) / oh as f32) * rect_height;
        for x in 0..ow {
            let u = rect_left + ((x as f32 + 0.5) / ow as f32) * rect_width;
            let mut best_alpha = 0.0_f32;
            let mut inside_bin = false;

            for s in strokes {
                let inner_r = s.radius.clamp(0.003, 0.25);
                let feather_w = compute_inpaint_feather_norm(inner_r, min_side);
                let soft_outer_r = inner_r + feather_w;
                let bin_outer_r = soft_outer_r + bin_margin;
                let pts = &s.points;
                let mut min_d = f32::INFINITY;
                if pts.len() == 1 {
                    let dx = (u - pts[0][0]) * sx;
                    let dy = (v - pts[0][1]) * sy;
                    min_d = (dx * dx + dy * dy).sqrt();
                } else {
                    for k in 0..(pts.len() - 1) {
                        let d = point_to_segment_dist(u, v, pts[k], pts[k + 1], sx, sy);
                        if d < min_d {
                            min_d = d;
                        }
                    }
                }
                if min_d < bin_outer_r {
                    inside_bin = true;
                }
                if min_d <= inner_r {
                    best_alpha = 1.0;
                } else if min_d < soft_outer_r {
                    let t = ((min_d - inner_r) / feather_w.max(1e-5)).clamp(0.0, 1.0);
                    let a = 1.0 - t * t * (3.0 - 2.0 * t);
                    if a > best_alpha {
                        best_alpha = a;
                    }
                }
            }

            let idx = y * ow + x;
            if inside_bin {
                binary[idx] = 255;
            }
            soft[idx] = (best_alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }

    (binary, soft)
}

fn point_to_segment_dist(u: f32, v: f32, a: [f32; 2], b: [f32; 2], sx: f32, sy: f32) -> f32 {
    let px = u * sx;
    let py = v * sy;
    let ax = a[0] * sx;
    let ay = a[1] * sy;
    let bx = b[0] * sx;
    let by = b[1] * sy;
    let abx = bx - ax;
    let aby = by - ay;
    let len_sq = abx * abx + aby * aby;
    let t = if len_sq > 1e-10 {
        (((px - ax) * abx + (py - ay) * aby) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let qx = ax + abx * t;
    let qy = ay + aby * t;
    ((px - qx) * (px - qx) + (py - qy) * (py - qy)).sqrt()
}

/// Fast multi-scale coarse-to-fine harmonic diffusion inpainting fallback (runs in <15ms on CPU
/// when optional `lama_fp32.onnx` is not installed, and enables deterministic offline CI tests).
pub fn inpaint_harmonic_pyramidal(rgb: &RgbImage, mask_u8: &[u8]) -> RgbImage {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let mut cur = vec![[0.0_f32; 3]; w * h];
    let mut hole = vec![false; w * h];
    let mut sum_valid = [0.0_f32; 3];
    let mut valid_count = 0usize;

    for i in 0..(w * h) {
        let p = i * 3;
        let col = [
            rgb.as_raw()[p] as f32 / 255.0,
            rgb.as_raw()[p + 1] as f32 / 255.0,
            rgb.as_raw()[p + 2] as f32 / 255.0,
        ];
        cur[i] = col;
        if mask_u8[i] > 96 {
            hole[i] = true;
        } else {
            sum_valid[0] += col[0];
            sum_valid[1] += col[1];
            sum_valid[2] += col[2];
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return rgb.clone();
    }
    let mean_col = [
        sum_valid[0] / valid_count as f32,
        sum_valid[1] / valid_count as f32,
        sum_valid[2] / valid_count as f32,
    ];
    for i in 0..(w * h) {
        if hole[i] {
            cur[i] = mean_col;
        }
    }

    // Coarse-to-fine Red-Black SOR Laplace relaxation converges 10x faster than single-scale Jacobi
    let filled = solve_harmonic_recursive(&cur, &hole, w, h, 0);
    let mut out = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let c = filled[y * w + x];
            out.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    (c[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (c[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (c[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                ]),
            );
        }
    }
    out
}

fn solve_harmonic_recursive(
    init: &[[f32; 3]],
    hole: &[bool],
    w: usize,
    h: usize,
    depth: usize,
) -> Vec<[f32; 3]> {
    let mut grid = init.to_vec();

    if w >= 24 && h >= 24 && depth < 4 {
        let cw = (w / 2).max(1);
        let ch = (h / 2).max(1);
        let mut c_init = vec![[0.0_f32; 3]; cw * ch];
        let mut c_hole = vec![false; cw * ch];

        for cy in 0..ch {
            for cx in 0..cw {
                let mut acc = [0.0_f32; 3];
                let mut known = 0usize;
                let mut all_acc = [0.0_f32; 3];
                let mut all_n = 0usize;
                for dy in 0..2 {
                    let sy = (cy * 2 + dy).min(h - 1);
                    for dx in 0..2 {
                        let sx = (cx * 2 + dx).min(w - 1);
                        let idx = sy * w + sx;
                        all_acc[0] += init[idx][0];
                        all_acc[1] += init[idx][1];
                        all_acc[2] += init[idx][2];
                        all_n += 1;
                        if !hole[idx] {
                            acc[0] += init[idx][0];
                            acc[1] += init[idx][1];
                            acc[2] += init[idx][2];
                            known += 1;
                        }
                    }
                }
                let ci = cy * cw + cx;
                if known > 0 {
                    c_init[ci] = [
                        acc[0] / known as f32,
                        acc[1] / known as f32,
                        acc[2] / known as f32,
                    ];
                    c_hole[ci] = false;
                } else {
                    c_init[ci] = [
                        all_acc[0] / all_n.max(1) as f32,
                        all_acc[1] / all_n.max(1) as f32,
                        all_acc[2] / all_n.max(1) as f32,
                    ];
                    c_hole[ci] = true;
                }
            }
        }

        let coarse = solve_harmonic_recursive(&c_init, &c_hole, cw, ch, depth + 1);
        // Bilinearly upsample coarse solution into fine holes
        for y in 0..h {
            let fy = ((y as f32 + 0.5) * ch as f32 / h as f32 - 0.5).clamp(0.0, (ch - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(ch - 1);
            let ty = fy - y0 as f32;
            for x in 0..w {
                let idx = y * w + x;
                if !hole[idx] {
                    continue;
                }
                let fx =
                    ((x as f32 + 0.5) * cw as f32 / w as f32 - 0.5).clamp(0.0, (cw - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(cw - 1);
                let tx = fx - x0 as f32;
                let p00 = coarse[y0 * cw + x0];
                let p10 = coarse[y0 * cw + x1];
                let p01 = coarse[y1 * cw + x0];
                let p11 = coarse[y1 * cw + x1];
                for c in 0..3 {
                    let top = p00[c] + (p10[c] - p00[c]) * tx;
                    let bot = p01[c] + (p11[c] - p01[c]) * tx;
                    grid[idx][c] = top + (bot - top) * ty;
                }
            }
        }
    }

    // Gauss-Seidel SOR relaxation iterations at this pyramid level
    let iters = if depth >= 3 { 40 } else { 18 };
    let omega = 1.65_f32;
    for _ in 0..iters {
        for y in 0..h {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            for x in 0..w {
                let idx = y * w + x;
                if !hole[idx] {
                    continue;
                }
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                let l = grid[y * w + xm];
                let r = grid[y * w + xp];
                let u = grid[ym * w + x];
                let d = grid[yp * w + x];
                for c in 0..3 {
                    let avg = 0.25 * (l[c] + r[c] + u[c] + d[c]);
                    grid[idx][c] = (grid[idx][c] + omega * (avg - grid[idx][c])).clamp(0.0, 1.0);
                }
            }
        }
    }

    grid
}

fn build_cpu_session(
    env: &ort::environment::Environment,
    model_path: &Path,
    intra_threads: usize,
) -> Result<Session, AppError> {
    let builder = Session::builder(env).map_err(|err| {
        AppError::ImageProcessingFailure(format!("LaMa session builder error: {err}"))
    })?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| AppError::ImageProcessingFailure(format!("LaMa graph opt error: {err}")))?;
    let mut builder = builder
        .with_intra_threads(intra_threads.max(1))
        .map_err(|err| AppError::ImageProcessingFailure(format!("LaMa threads error: {err}")))?;
    builder.commit_from_file(model_path).map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "failed to load LaMa ONNX '{}': {err}",
            model_path.display()
        ))
    })
}

fn detect_lama_io(session: &Session) -> Result<(String, String, u32, u32), AppError> {
    let inputs = session.inputs();
    if inputs.len() < 2 {
        return Err(AppError::ImageProcessingFailure(
            "LaMa ONNX model must declare 2 inputs (image and mask)".to_string(),
        ));
    }

    let mut image_name: Option<String> = None;
    let mut mask_name: Option<String> = None;
    let mut h = DEFAULT_LAMA_DIM;
    let mut w = DEFAULT_LAMA_DIM;

    for inp in inputs {
        let name = inp.name().to_string();
        let lower = name.to_ascii_lowercase();
        let shape = inp.dtype().tensor_shape();
        let ch = shape.and_then(|s| s.get(1).copied()).unwrap_or(-1);
        if ch == 3 || (image_name.is_none() && (lower.contains("image") || lower.contains("img"))) {
            image_name = Some(name);
            if let Some(s) = shape
                && s.len() == 4
            {
                if s[2] >= 128 && s[2] <= 1024 {
                    h = s[2] as u32;
                }
                if s[3] >= 128 && s[3] <= 1024 {
                    w = s[3] as u32;
                }
            }
        } else if ch == 1 || (mask_name.is_none() && lower.contains("mask")) {
            mask_name = Some(name);
        }
    }

    let img_n = image_name.unwrap_or_else(|| inputs[0].name().to_string());
    let mask_n = mask_name.unwrap_or_else(|| inputs[1].name().to_string());
    Ok((img_n, mask_n, h, w))
}

fn friendly_lama_label(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("lama_fp32.onnx")
        .to_ascii_lowercase();
    if name.contains("lama") {
        "LaMa Neural Inpainter".to_string()
    } else {
        "Custom Inpainter ONNX".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonic_inpaint_patch_removes_dark_distraction_and_roundtrips_rgba() {
        // Warm studio backdrop with a dark distraction square in the center
        let mut rgb = RgbImage::from_fn(256, 256, |x, _| {
            let v = (180.0 + (x as f32 / 255.0) * 20.0).round() as u8;
            Rgb([v, 165, 150])
        });
        for y in 112..144 {
            for x in 112..144 {
                rgb.put_pixel(x, y, Rgb([18, 20, 22]));
            }
        }

        let generated = generate_inpaint_patch(
            &rgb,
            &[InpaintStrokeInput {
                radius: 0.09,
                points: vec![[0.5, 0.5]],
            }],
            None,
        )
        .expect("harmonic inpaint should succeed");

        assert!(generated.patch.is_effective());
        assert_eq!(generated.patch.engine_label, "Harmonic Fill");
        let (pw, ph, rgba) =
            decode_inpaint_patch_rgba(&generated.patch).expect("patch rgba should decode");
        assert_eq!(pw as u16, generated.patch.patch_width);
        assert_eq!(ph as u16, generated.patch.patch_height);

        // Center pixel of the patch should match the warm studio backdrop (~190, 165, 150), not dark 18!
        let ci = ((ph / 2) as usize * pw as usize + (pw / 2) as usize) * 4;
        assert!(rgba[ci] > 165, "expected red > 165, got {}", rgba[ci]);
        assert!(rgba[ci + 1] > 145, "expected green > 145, got {}", rgba[ci + 1]);
        assert!(rgba[ci + 3] > 230, "expected opaque center mask, got {}", rgba[ci + 3]);
    }

    #[test]
    fn windows_directml_skip_targets_lama_ffc_models_only() {
        let lama_fp32 = Path::new("models/retouch/lama_fp32.onnx");
        let opencv_lama = Path::new("C:/models/inpainting_lama_2025jan.onnx");
        let custom_non_lama = Path::new("C:/models/custom_inpaint_unet.onnx");

        if cfg!(target_os = "windows") {
            assert!(should_skip_directml_for_model(lama_fp32));
            assert!(should_skip_directml_for_model(opencv_lama));
            assert!(!should_skip_directml_for_model(custom_non_lama));
        } else {
            assert!(!should_skip_directml_for_model(lama_fp32));
            assert!(!should_skip_directml_for_model(custom_non_lama));
        }
    }

    #[test]
    fn large_brush_keeps_full_opaque_core_and_tight_pixel_bounded_feather() {
        // 1000x1500 portrait with a large 15% brush (`150 px` radius)
        let stroke = InpaintStrokeInput {
            radius: 0.15,
            points: vec![[0.5, 0.5]],
        };
        let (bin, soft) = rasterize_strokes_in_rect(
            &[&stroke],
            0.0,
            0.0,
            1.0,
            1.0,
            1000.0,
            1500.0,
            1000,
            1500,
        );

        let at = |x: usize, y: usize| (bin[y * 1000 + x], soft[y * 1000 + x]);
        // Center and near-edge INSIDE painted 150px radius (at dx=145px) must be 100% opaque (255)
        assert_eq!(at(500, 750), (255, 255));
        assert_eq!(at(645, 750), (255, 255));
        // Inside the tight 10px feather zone just outside 150px (dx=154px): semi-transparent AND binary=255
        let (b_mid, s_mid) = at(654, 750);
        assert_eq!(b_mid, 255, "LaMa binary mask must cover the full soft feather ramp");
        assert!(s_mid > 20 && s_mid < 240, "expected smooth feather mid-ramp, got {s_mid}");
        // Outside feather (dx=162px): soft alpha must already be 0 (no 50px ghost band!), while binary safety margin still covers fringe
        let (b_margin, s_margin) = at(662, 750);
        assert_eq!(s_margin, 0, "feather band must stay tightly bounded <= 10px");
        assert_eq!(b_margin, 255, "binary dilation margin should cover fringe outside soft mask");
        // Well outside (dx=175px): both binary and soft are 0
        assert_eq!(at(675, 750), (0, 0));
    }

    #[test]
    fn square_context_window_preserves_aspect_and_covers_edge_strokes() {
        // 1. Small wrist-watch stroke near the bottom edge of a 1067x1600 portrait:
        //    must stay strictly square by shifting upward instead of clipping height.
        let (x0, y0, side) = compute_square_context_window(600, 1410, 760, 1580, 1067, 1600);
        assert!(x0 >= 0 && (x0 as u32 + side) <= 1067);
        assert!(y0 >= 0 && (y0 as u32 + side) <= 1600);
        assert!(x0 <= 600 && (x0 + side as i32) >= 760);
        assert!(y0 <= 1410 && (y0 + side as i32) >= 1580);
        assert!(side >= 256);

        // 2. Large torso mask spanning 70% of portrait width: tightens context multiplier below
        //    old 2.25x so LaMa's 512x512 grid preserves far more resolution on the hole, while
        //    keeping the context strictly square and covering the entire mask.
        let (lx0, ly0, lside) = compute_square_context_window(150, 450, 900, 1100, 1067, 1600);
        assert!(lside < 1450, "expected adaptive context tighter than 2.25x, got {lside}");
        assert!(lx0 <= 150 && (lx0 + lside as i32) >= 900);
        assert!(ly0 <= 450 && (ly0 + lside as i32) >= 1100);
    }

    #[test]
    fn generated_patch_preserves_exact_original_border_pixels_and_zero_alignment_shift() {
        // High-contrast vertical edge (bright shirt left x<180, dark studio backdrop right x>=180)
        // with a small distraction dot on the dark side near the edge.
        let mut rgb = RgbImage::from_fn(360, 480, |x, _| {
            if x < 180 {
                Rgb([210, 222, 218])
            } else {
                Rgb([46, 52, 62])
            }
        });
        for y in 225..255 {
            for x in 205..235 {
                rgb.put_pixel(x, y, Rgb([240, 240, 240]));
            }
        }

        let generated = generate_inpaint_patch(
            &rgb,
            &[InpaintStrokeInput {
                radius: 0.055,
                points: vec![[220.0 / 360.0, 240.0 / 480.0]],
            }],
            None,
        )
        .expect("patch generation should succeed");

        let (pw, ph, rgba) =
            decode_inpaint_patch_rgba(&generated.patch).expect("patch rgba should decode");
        let px0 = (generated.patch.bounds.left * 360.0).round() as u32;
        let py0 = (generated.patch.bounds.top * 480.0).round() as u32;

        // Every untouched 8-tap border sample used by `inpaint_pass.wgsl` must have alpha == 0
        // and exact 0-error RGB parity with `rgb` (preventing false color drift across sharp edges).
        let border_taps = [
            (0.02_f32, 0.02_f32),
            (0.50, 0.02),
            (0.98, 0.02),
            (0.98, 0.50),
            (0.98, 0.98),
            (0.50, 0.98),
            (0.02, 0.98),
            (0.02, 0.50),
        ];
        for (tu, tv) in border_taps {
            let bx = ((tu * pw as f32).floor() as u32).min(pw - 1);
            let by = ((tv * ph as f32).floor() as u32).min(ph - 1);
            let o = (by as usize * pw as usize + bx as usize) * 4;
            assert_eq!(rgba[o + 3], 0, "border tap ({tu},{tv}) must sit in untouched alpha==0 margin");
            let orig = rgb.get_pixel((px0 + bx).min(359), (py0 + by).min(479)).0;
            for c in 0..3 {
                assert!(
                    (rgba[o + c] as i32 - orig[c] as i32).abs() <= 1,
                    "border tap ({tu},{tv}) channel {c} drifted: patch={} orig={}",
                    rgba[o + c],
                    orig[c]
                );
            }
        }
    }
}
