//! BiSeNet 19-Class CelebAMask-HQ Neural Face Parsing Engine (Phase 3 Opt-In).
//!
//! Runs `parsing_resnet18.onnx` (or any compatible `[1, 3, 512, 512] -> [1, 19, 512, 512]`
//! CelebAMask-HQ ONNX face parser) via ONNX Runtime DirectML/CoreML GPU or CPU fallback.
//!
//! Extracts expanded square crops around up to 4 YuNet-detected faces so even full-body
//! graduation portraits receive crisp 512px per-face semantic parsing, then projects a compact
//! 4-channel RGBA probability map (`R = skin`, `G = eyes`, `B = teeth/mouth`, `A = lips`)
//! directly into Kroma's 32-bit `portrait_retouch_pass.wgsl` shader and AI Face & Skin masks.

use crate::vision::onnx_runtime::shared_ort_environment;
use image::imageops::{self, FilterType};
use image::RgbImage;
use kroma_app::error::AppError;
use kroma_domain::retouch::{FaceParseMap, PortraitFaceGuide, MAX_PORTRAIT_FACE_GUIDES};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const PARSE_MAP_MAX_DIM: u32 = 512;
const DEFAULT_INPUT_DIM: u32 = 512;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

struct FaceParserSessionState {
    session: Session,
    using_gpu: bool,
}

/// Real ONNX Runtime adapter for 19-class CelebAMask-HQ BiSeNet face parsing models.
pub struct OnnxBiSeNetFaceParser {
    state: Mutex<FaceParserSessionState>,
    model_path: PathBuf,
    intra_threads: usize,
    input_w: u32,
    input_h: u32,
    model_label: String,
}

fn run_face_parser_session(
    session: &mut Session,
    ih: usize,
    iw: usize,
    tensor_data: Vec<f32>,
) -> Result<(usize, usize, usize, Vec<f32>), AppError> {
    let input_tensor = Tensor::from_array(([1usize, 3, ih, iw], tensor_data)).map_err(|err| {
        AppError::ImageProcessingFailure(format!("face parser input tensor error: {err}"))
    })?;

    let outputs = session.run(ort::inputs![input_tensor]).map_err(|err| {
        AppError::ImageProcessingFailure(format!("face parser ONNX inference failed: {err}"))
    })?;

    if outputs.len() == 0 {
        return Err(AppError::ImageProcessingFailure(
            "face parser ONNX model returned zero output tensors".to_string(),
        ));
    }

    let (shape, logits) = outputs[0].try_extract_tensor::<f32>().map_err(|err| {
        AppError::ImageProcessingFailure(format!("face parser output extraction failed: {err}"))
    })?;

    let dims: Vec<usize> = shape.iter().map(|&d| d.max(1) as usize).collect();
    if dims.len() != 4 || dims[1] < 14 {
        return Err(AppError::ImageProcessingFailure(format!(
            "unexpected face parser output shape: {dims:?}"
        )));
    }

    let num_classes = dims[1];
    let oh = dims[2];
    let ow = dims[3];
    let spatial = oh * ow;
    if logits.len() < num_classes * spatial {
        return Err(AppError::ImageProcessingFailure(
            "face parser output buffer shorter than tensor shape".to_string(),
        ));
    }

    Ok((num_classes, oh, ow, logits[..num_classes * spatial].to_vec()))
}

impl OnnxBiSeNetFaceParser {
    pub fn from_model_file(
        model_path: impl AsRef<Path>,
        intra_threads: usize,
    ) -> Result<Self, AppError> {
        let path_ref = model_path.as_ref();
        let env = shared_ort_environment()?;
        let effective_threads = intra_threads.max(1);

        let (session, using_gpu) =
            match crate::vision::onnx_runtime::try_build_gpu_session(env, path_ref) {
                Some(gpu_session) => (gpu_session, true),
                None => (build_cpu_session(env, path_ref, effective_threads)?, false),
            };

        let (input_h, input_w) = detect_input_hw(&session);
        let model_label = friendly_parser_label(path_ref);

        Ok(Self {
            state: Mutex::new(FaceParserSessionState { session, using_gpu }),
            model_path: path_ref.to_path_buf(),
            intra_threads: effective_threads,
            input_w,
            input_h,
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

    /// Runs 19-class BiSeNet inference on an already-cropped face RGB image and returns
    /// `(out_h, out_w, 4-channel RGBA soft probabilities [R=skin, G=eyes, B=teeth, A=lips])`.
    fn parse_crop_semantic_rgba(&self, crop_rgb: &RgbImage) -> Result<(usize, usize, Vec<f32>), AppError> {
        let resized = imageops::resize(crop_rgb, self.input_w, self.input_h, FilterType::Triangle);
        let iw = self.input_w as usize;
        let ih = self.input_h as usize;
        let plane = iw * ih;
        let mut tensor_data = vec![0.0_f32; 3 * plane];

        for y in 0..ih {
            for x in 0..iw {
                let [r8, g8, b8] = resized.get_pixel(x as u32, y as u32).0;
                let off = y * iw + x;
                tensor_data[off] = (r8 as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                tensor_data[plane + off] =
                    (g8 as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                tensor_data[2 * plane + off] =
                    (b8 as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
        }

        let mut state = self.state.lock().map_err(|_| {
            AppError::ImageProcessingFailure("face parser ONNX session mutex poisoned".to_string())
        })?;

        let (num_classes, oh, ow, logits) = if state.using_gpu {
            match run_face_parser_session(&mut state.session, ih, iw, tensor_data.clone()) {
                Ok(out) => out,
                Err(gpu_err) => {
                    eprintln!(
                        "[kroma] BiSeNet GPU inference failed ({gpu_err}); falling back to CPU ONNX Runtime"
                    );
                    let env = shared_ort_environment()?;
                    let cpu_session = build_cpu_session(env, &self.model_path, self.intra_threads)?;
                    state.session = cpu_session;
                    state.using_gpu = false;
                    run_face_parser_session(&mut state.session, ih, iw, tensor_data)?
                }
            }
        } else {
            run_face_parser_session(&mut state.session, ih, iw, tensor_data)?
        };
        drop(state);

        let spatial = oh * ow;

        let mut rgba = vec![0.0_f32; spatial * 4];
        let mut probs = vec![0.0_f32; num_classes];

        for idx in 0..spatial {
            // Numerically stable softmax across CelebAMask-HQ 19 classes
            let mut max_logit = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let v = logits[c * spatial + idx];
                if v > max_logit {
                    max_logit = v;
                }
            }
            let mut sum_exp = 0.0_f32;
            for (c, p_slot) in probs.iter_mut().enumerate().take(num_classes) {
                let e = (logits[c * spatial + idx] - max_logit).clamp(-20.0, 0.0).exp();
                *p_slot = e;
                sum_exp += e;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-6);
            for p_slot in &mut probs {
                *p_slot *= inv_sum;
            }

            // CelebAMask-HQ class mapping:
            // 0: bg, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 6: eyeglass,
            // 7: l_ear, 8: r_ear, 9: earring, 10: nose, 11: mouth/teeth, 12: u_lip,
            // 13: l_lip, 14: neck, 15: necklace, 16: cloth, 17: hair, 18: hat
            let p_skin_raw = probs[1]
                + probs[10]
                + probs.get(7).copied().unwrap_or(0.0)
                + probs.get(8).copied().unwrap_or(0.0)
                + probs.get(14).copied().unwrap_or(0.0);
            let p_eye = probs[4] + probs[5];
            let p_teeth = probs[11];
            let p_lip = probs[12] + probs[13];
            let p_protect = probs[2]
                + probs[3]
                + p_eye
                + probs[6] // eyeglasses — never blur frames or lenses!
                + p_teeth
                + p_lip
                + probs.get(9).copied().unwrap_or(0.0)
                + probs.get(15).copied().unwrap_or(0.0)
                + probs.get(17).copied().unwrap_or(0.0) // hair / bangs
                + probs.get(18).copied().unwrap_or(0.0);

            let clean_skin = (p_skin_raw * (1.0 - (p_protect * 1.45).min(1.0))).clamp(0.0, 1.0);
            let o = idx * 4;
            rgba[o] = clean_skin;
            rgba[o + 1] = p_eye.clamp(0.0, 1.0);
            rgba[o + 2] = p_teeth.clamp(0.0, 1.0);
            rgba[o + 3] = p_lip.clamp(0.0, 1.0);
        }

        Ok((oh, ow, rgba))
    }

    /// Parses all detected faces (up to 4) in an oriented portrait image and projects their
    /// semantic probabilities into a full-frame compact RGBA8 `FaceParseMap` (max 512px long edge).
    pub fn parse_portrait_rgba(
        &self,
        rgb: &RgbImage,
        faces: &[PortraitFaceGuide],
    ) -> Result<FaceParseMap, AppError> {
        let (img_w, img_h) = rgb.dimensions();
        if img_w == 0 || img_h == 0 {
            return Err(AppError::ImageProcessingFailure(
                "cannot run face parsing on an empty image".to_string(),
            ));
        }

        let max_side = img_w.max(img_h);
        let scale = if max_side > PARSE_MAP_MAX_DIM {
            PARSE_MAP_MAX_DIM as f32 / max_side as f32
        } else {
            1.0
        };
        let out_w = ((img_w as f32 * scale).round() as u32).clamp(16, PARSE_MAP_MAX_DIM) as usize;
        let out_h = ((img_h as f32 * scale).round() as u32).clamp(16, PARSE_MAP_MAX_DIM) as usize;
        let mut accum = vec![0.0_f32; out_w * out_h * 4];

        let crop_rects = build_face_crop_rects(img_w, img_h, faces);
        for (cx0, cy0, cw, ch, is_cropped) in crop_rects {
            let sub = imageops::crop_imm(rgb, cx0, cy0, cw, ch).to_image();
            let (ph, pw, crop_rgba) = self.parse_crop_semantic_rgba(&sub)?;

            // Project crop semantic probabilities back onto full-frame map
            let u0 = cx0 as f32 / img_w as f32;
            let v0 = cy0 as f32 / img_h as f32;
            let uw = cw as f32 / img_w as f32;
            let vh = ch as f32 / img_h as f32;

            let mx0 = ((u0 * out_w as f32).floor() as i32).clamp(0, out_w as i32 - 1) as usize;
            let my0 = ((v0 * out_h as f32).floor() as i32).clamp(0, out_h as i32 - 1) as usize;
            let mx1 = (((u0 + uw) * out_w as f32).ceil() as i32).clamp(0, out_w as i32) as usize;
            let my1 = (((v0 + vh) * out_h as f32).ceil() as i32).clamp(0, out_h as i32) as usize;

            for my in my0..my1 {
                let fy_norm = ((my as f32 + 0.5) / out_h as f32 - v0) / vh.max(1e-5);
                if !(0.0..=1.0).contains(&fy_norm) {
                    continue;
                }
                let py_edge = if is_cropped {
                    border_fade(fy_norm)
                } else {
                    1.0
                };
                for mx in mx0..mx1 {
                    let fx_norm = ((mx as f32 + 0.5) / out_w as f32 - u0) / uw.max(1e-5);
                    if !(0.0..=1.0).contains(&fx_norm) {
                        continue;
                    }
                    let edge_w = py_edge
                        * if is_cropped {
                            border_fade(fx_norm)
                        } else {
                            1.0
                        };
                    let sampled = sample_rgba_bilinear(&crop_rgba, pw, ph, fx_norm, fy_norm);
                    let dst = (my * out_w + mx) * 4;
                    accum[dst] = accum[dst].max(sampled[0] * edge_w);
                    accum[dst + 1] = accum[dst + 1].max(sampled[1] * edge_w);
                    accum[dst + 2] = accum[dst + 2].max(sampled[2] * edge_w);
                    accum[dst + 3] = accum[dst + 3].max(sampled[3] * edge_w);
                }
            }
        }

        // Gentle 1px 3x3 cross smooth removes discrete argmax stair-stepping at hair/lip edges
        let smoothed = smooth_rgba_3x3(&accum, out_w, out_h);
        let rgba: Vec<u8> = smoothed
            .into_iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();

        Ok(FaceParseMap {
            width: out_w as u16,
            height: out_h as u16,
            rgba,
        })
    }

    /// Extracts a soft `[0..1]` skin probability matte (sized to `out_w x out_h`) for Kroma's
    /// local **AI Face & Skin** mask generator.
    pub fn predict_face_skin_matte_f32(
        &self,
        rgb: &RgbImage,
        faces: &[PortraitFaceGuide],
        out_w: u32,
        out_h: u32,
    ) -> Result<Vec<f32>, AppError> {
        let map = self.parse_portrait_rgba(rgb, faces)?;
        let mw = map.width as usize;
        let mh = map.height as usize;
        let ow = out_w.max(1) as usize;
        let oh = out_h.max(1) as usize;
        let mut out = vec![0.0_f32; ow * oh];
        for y in 0..oh {
            let v = (y as f32 + 0.5) / oh as f32;
            for x in 0..ow {
                let u = (x as f32 + 0.5) / ow as f32;
                let fx = (u * mw as f32 - 0.5).clamp(0.0, (mw - 1) as f32);
                let fy = (v * mh as f32 - 0.5).clamp(0.0, (mh - 1) as f32);
                let x0 = fx.floor() as usize;
                let y0 = fy.floor() as usize;
                let x1 = (x0 + 1).min(mw - 1);
                let y1 = (y0 + 1).min(mh - 1);
                let tx = fx - x0 as f32;
                let ty = fy - y0 as f32;
                let r00 = map.rgba[(y0 * mw + x0) * 4] as f32 / 255.0;
                let r10 = map.rgba[(y0 * mw + x1) * 4] as f32 / 255.0;
                let r01 = map.rgba[(y1 * mw + x0) * 4] as f32 / 255.0;
                let r11 = map.rgba[(y1 * mw + x1) * 4] as f32 / 255.0;
                let top = r00 + (r10 - r00) * tx;
                let bot = r01 + (r11 - r01) * tx;
                out[y * ow + x] = (top + (bot - top) * ty).clamp(0.0, 1.0);
            }
        }
        Ok(out)
    }
}

fn build_face_crop_rects(
    img_w: u32,
    img_h: u32,
    faces: &[PortraitFaceGuide],
) -> Vec<(u32, u32, u32, u32, bool)> {
    if faces.is_empty() {
        return vec![(0, 0, img_w, img_h, false)];
    }
    let fw = img_w as f32;
    let fh = img_h as f32;
    let mut rects = Vec::with_capacity(faces.len().min(MAX_PORTRAIT_FACE_GUIDES));
    for face in faces.iter().take(MAX_PORTRAIT_FACE_GUIDES) {
        let bx = face.box_left * fw;
        let by = face.box_top * fh;
        let bw = face.box_width * fw;
        let bh = face.box_height * fh;
        let cx = bx + bw * 0.5;
        let cy = by + bh * 0.54; // slight downward bias includes chin & neck cleanly
        let side = (bw.max(bh) * 2.35).clamp(64.0, fw.max(fh));
        let half = side * 0.5;
        let x0 = (cx - half).floor().clamp(0.0, (fw - 16.0).max(0.0)) as u32;
        let y0 = (cy - half).floor().clamp(0.0, (fh - 16.0).max(0.0)) as u32;
        let x1 = (cx + half).ceil().clamp((x0 + 16) as f32, fw) as u32;
        let y1 = (cy + half).ceil().clamp((y0 + 16) as f32, fh) as u32;
        rects.push((x0, y0, (x1 - x0).max(16), (y1 - y0).max(16), true));
    }
    rects
}

fn border_fade(t: f32) -> f32 {
    let margin = 0.05_f32;
    let d = t.min(1.0 - t);
    if d >= margin {
        1.0
    } else {
        let x = (d / margin).clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }
}

fn sample_rgba_bilinear(buf: &[f32], w: usize, h: usize, u: f32, v: f32) -> [f32; 4] {
    let fx = (u * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (v * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let i00 = (y0 * w + x0) * 4;
    let i10 = (y0 * w + x1) * 4;
    let i01 = (y1 * w + x0) * 4;
    let i11 = (y1 * w + x1) * 4;

    let mut out = [0.0_f32; 4];
    for c in 0..4 {
        let top = buf[i00 + c] + (buf[i10 + c] - buf[i00 + c]) * tx;
        let bot = buf[i01 + c] + (buf[i11 + c] - buf[i01 + c]) * tx;
        out[c] = top + (bot - top) * ty;
    }
    out
}

fn smooth_rgba_3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut dst = vec![0.0_f32; src.len()];
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let ic = (y * w + x) * 4;
            let il = (y * w + xm) * 4;
            let ir = (y * w + xp) * 4;
            let iu = (ym * w + x) * 4;
            let id = (yp * w + x) * 4;
            for c in 0..4 {
                dst[ic + c] =
                    (src[ic + c] * 2.0 + src[il + c] + src[ir + c] + src[iu + c] + src[id + c])
                        / 6.0;
            }
        }
    }
    dst
}

fn build_cpu_session(
    env: &ort::environment::Environment,
    model_path: &Path,
    intra_threads: usize,
) -> Result<Session, AppError> {
    let builder = Session::builder(env).map_err(|err| {
        AppError::ImageProcessingFailure(format!("face parser session builder error: {err}"))
    })?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!("face parser graph opt error: {err}"))
        })?;
    let mut builder = builder
        .with_intra_threads(intra_threads.max(1))
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!("face parser threads error: {err}"))
        })?;
    builder.commit_from_file(model_path).map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "failed to load face parser ONNX '{}': {err}",
            model_path.display()
        ))
    })
}

fn detect_input_hw(session: &Session) -> (u32, u32) {
    if let Some(inp) = session.inputs().first()
        && let Some(shape) = inp.dtype().tensor_shape()
        && shape.len() == 4
    {
        let h = if shape[2] >= 128 && shape[2] <= 1024 {
            shape[2] as u32
        } else {
            DEFAULT_INPUT_DIM
        };
        let w = if shape[3] >= 128 && shape[3] <= 1024 {
            shape[3] as u32
        } else {
            DEFAULT_INPUT_DIM
        };
        return (h, w);
    }
    (DEFAULT_INPUT_DIM, DEFAULT_INPUT_DIM)
}

fn friendly_parser_label(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("parsing_resnet18.onnx")
        .to_ascii_lowercase();
    if name.contains("resnet34") {
        "BiSeNet ResNet-34 Face Parsing".to_string()
    } else if name.contains("resnet18") || name.contains("parsing") || name.contains("bisenet") {
        "BiSeNet Face Parsing".to_string()
    } else {
        "Custom Face Parsing ONNX".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_clamped_face_crop_rects_around_yunet_guides_and_falls_back_to_full_frame() {
        let full = build_face_crop_rects(1200, 1600, &[]);
        assert_eq!(full, vec![(0, 0, 1200, 1600, false)]);

        let guide = PortraitFaceGuide::from_normalized(
            0.35,
            0.20,
            0.25,
            0.20,
            [0.42, 0.28],
            [0.53, 0.28],
            [0.47, 0.33],
            [0.43, 0.37],
            [0.52, 0.37],
        );
        let crops = build_face_crop_rects(1200, 1600, &[guide]);
        assert_eq!(crops.len(), 1);
        let (x0, y0, cw, ch, is_cropped) = crops[0];
        assert!(is_cropped);
        assert!(x0 + cw <= 1200 && y0 + ch <= 1600);
        assert!(cw >= 400 && ch >= 400);
    }
}
