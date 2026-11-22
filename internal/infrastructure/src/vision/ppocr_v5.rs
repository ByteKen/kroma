use crate::develop::image_decoder;
use crate::vision::onnx_runtime::shared_ort_environment;
use image::imageops::{self, FilterType};
use image::{Rgb, RgbImage};
use kroma_app::error::AppError;
use kroma_app::ports::vision::{SlateOcrReading, SlateReaderPort};
use kroma_domain::develop::NormalizedRect;
use kroma_domain::slate::{parse_whiteboard_text_lines, OcrTextLine, SlateCandidateParse};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub struct PpOcrV5Config {
    /// Longest side for PP-OCRv5 DBNet detection input (default `960`, rounded to multiple of 32).
    pub det_max_side: u32,
    /// Probability threshold on the DBNet segmentation map (`0.30` standard PaddleOCR default).
    pub det_thresh: f32,
    /// Minimum average probability inside a connected text component (`0.55`).
    pub box_thresh: f32,
    /// DBNet unclip expansion ratio around shrunk text kernels (`1.6`).
    pub unclip_ratio: f32,
    /// Maximum text crops recognized per image (guards against busy patterned backgrounds).
    pub max_text_boxes: usize,
    /// Minimum recognition confidence kept in returned lines (`0.35`).
    pub min_rec_confidence: f32,
    /// CPU intra-op threads when DirectML GPU is unavailable.
    pub intra_threads: usize,
}

impl Default for PpOcrV5Config {
    fn default() -> Self {
        Self {
            det_max_side: 960,
            det_thresh: 0.30,
            box_thresh: 0.55,
            unclip_ratio: 1.55,
            max_text_boxes: 24,
            min_rec_confidence: 0.35,
            intra_threads: 4,
        }
    }
}

#[derive(Debug, Clone)]
struct DetectedTextBox {
    /// Normalized axis-aligned bounding box on the original image `[0..1]`.
    region: NormalizedRect,
    /// Center in original image pixels.
    center_x: f32,
    center_y: f32,
    /// Unclipped box dimensions in original image pixels.
    width_px: f32,
    height_px: f32,
    /// Principal-axis tilt in radians (`0.0` when nearly horizontal or short word).
    angle_rad: f32,
    det_score: f32,
}

/// Real ONNX Runtime adapter for **PaddleOCR PP-OCRv5** (`PP-OCRv5_server_det` / `PP-OCRv5_mobile_det`
/// + `en_PP-OCRv5_mobile_rec` / `latin_PP-OCRv5_mobile_rec` + matching dictionary).
///
/// Runs 100% locally offline with DirectML GPU acceleration on Windows and automatic CPU fallback.
pub struct PpOcrV5Engine {
    det_session: Mutex<Session>,
    rec_session: Mutex<Session>,
    dictionary: Vec<String>,
    rec_height: u32,
    using_gpu: bool,
    config: PpOcrV5Config,
}

impl PpOcrV5Engine {
    pub fn from_model_files(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        dict_path: impl AsRef<Path>,
        config: PpOcrV5Config,
    ) -> Result<Self, AppError> {
        let det_ref = det_model_path.as_ref();
        let rec_ref = rec_model_path.as_ref();
        let dict_ref = dict_path.as_ref();

        let dict_raw = fs::read_to_string(dict_ref).map_err(|err| {
            AppError::ImageProcessingFailure(format!(
                "failed to read PP-OCRv5 dictionary '{}': {err}",
                dict_ref.display()
            ))
        })?;
        let dictionary: Vec<String> = dict_raw
            .lines()
            .map(|line| line.trim_end_matches('\r').to_string())
            .filter(|line| !line.is_empty())
            .collect();
        if dictionary.is_empty() {
            return Err(AppError::ImageProcessingFailure(format!(
                "PP-OCRv5 dictionary '{}' is empty",
                dict_ref.display()
            )));
        }

        let env = shared_ort_environment()?;
        // Prefer GPU/NPU acceleration (DirectML on Windows, CoreML on macOS) for DBNet detector;
        // the lightweight 7MB dynamic-width recognizer runs fastest on CPU with zero shader recompiles.
        let (det_session, det_gpu) =
            match crate::vision::onnx_runtime::try_build_gpu_session(env, det_ref) {
                Some(s) => (s, true),
                None => (build_cpu_session(env, det_ref, config.intra_threads)?, false),
            };
        let rec_session = build_cpu_session(env, rec_ref, config.intra_threads)?;

        // Respect static model input height if fixed (e.g. 48 or 32), defaulting to 48 for PP-OCRv5.
        let rec_height = detect_rec_input_height(&rec_session).unwrap_or(48);

        Ok(Self {
            det_session: Mutex::new(det_session),
            rec_session: Mutex::new(rec_session),
            dictionary,
            rec_height,
            using_gpu: det_gpu,
            config,
        })
    }

    pub fn execution_provider_label(&self) -> &'static str {
        if self.using_gpu {
            crate::vision::onnx_runtime::gpu_execution_provider_label()
        } else {
            "CPU"
        }
    }

    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }

    pub fn rec_height(&self) -> u32 {
        self.rec_height
    }

    /// Runs full PP-OCRv5 text detection + recognition on a decoded `RgbImage` and returns
    /// sorted reading-order text lines along with the parsed whiteboard subject candidate.
    pub fn read_slate_from_rgb(
        &self,
        rgb: &RgbImage,
    ) -> Result<(Vec<OcrTextLine>, Option<SlateCandidateParse>), AppError> {
        let lines = self.read_lines_from_rgb(rgb)?;
        let parsed = parse_whiteboard_text_lines(&lines);
        Ok((lines, parsed))
    }

    /// Detects and recognizes all text lines in an `RgbImage` sorted top-to-bottom, left-to-right.
    pub fn read_lines_from_rgb(&self, rgb: &RgbImage) -> Result<Vec<OcrTextLine>, AppError> {
        let boxes = self.detect_text_boxes(rgb)?;
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        let mut lines = Vec::with_capacity(boxes.len());
        for det_box in &boxes {
            let crop = extract_deskewed_crop(rgb, det_box);
            if crop.width() < 4 || crop.height() < 4 {
                continue;
            }
            let (text, rec_conf) = self.recognize_text_crop(&crop)?;
            let cleaned = text.trim();
            if cleaned.is_empty()
                || rec_conf < self.config.min_rec_confidence
                || !cleaned.chars().any(|c| c.is_alphanumeric())
            {
                continue;
            }
            // Combine recognition confidence with detection score mildly weighted toward recognition
            let combined_conf = (rec_conf * 0.85 + det_box.det_score * 0.15).clamp(0.0, 1.0);
            lines.push(OcrTextLine {
                text: cleaned.to_string(),
                confidence: combined_conf,
                region: det_box.region,
            });
        }
        Ok(lines)
    }

    /// Reads all text lines + parsed slate directly from a RAW/JPEG/PNG photo path on disk.
    pub fn read_slate_from_path(
        &self,
        source_path: &str,
    ) -> Result<(Vec<OcrTextLine>, Option<SlateCandidateParse>), AppError> {
        let rgb = image_decoder::decode_fast_preview_max_dim(source_path, 1600)?;
        self.read_slate_from_rgb(&rgb)
    }

    fn detect_text_boxes(&self, rgb: &RgbImage) -> Result<Vec<DetectedTextBox>, AppError> {
        let (orig_w, orig_h) = rgb.dimensions();
        if orig_w < 8 || orig_h < 8 {
            return Ok(Vec::new());
        }

        // Fixed square letterbox canvas so DirectML compiles shaders ONCE and reuses them across
        // both portrait and landscape photos in 500+ subject shoots!
        let canvas = round_to_32(self.config.det_max_side);
        let max_side = orig_w.max(orig_h);
        let scale = canvas as f32 / max_side as f32;
        let res_w = ((orig_w as f32 * scale).round() as u32).clamp(1, canvas);
        let res_h = ((orig_h as f32 * scale).round() as u32).clamp(1, canvas);

        let resized = if res_w == orig_w && res_h == orig_h {
            rgb.clone()
        } else {
            imageops::resize(rgb, res_w, res_h, FilterType::Triangle)
        };

        // PaddleOCR DBNet preprocessing: BGR channel order + ImageNet mean/std normalization.
        // Unused right/bottom letterbox pixels stay 0.0 (mean gray in normalized space).
        let w = canvas as usize;
        let h = canvas as usize;
        let plane = w * h;
        let mut tensor_data = vec![0.0_f32; 3 * plane];
        const MEAN_BGR: [f32; 3] = [0.406, 0.456, 0.485];
        const STD_BGR: [f32; 3] = [0.225, 0.224, 0.229];

        for y in 0..res_h as usize {
            for x in 0..res_w as usize {
                let [r, g, b] = resized.get_pixel(x as u32, y as u32).0;
                let idx = y * w + x;
                tensor_data[idx] = (b as f32 / 255.0 - MEAN_BGR[0]) / STD_BGR[0];
                tensor_data[plane + idx] = (g as f32 / 255.0 - MEAN_BGR[1]) / STD_BGR[1];
                tensor_data[2 * plane + idx] = (r as f32 / 255.0 - MEAN_BGR[2]) / STD_BGR[2];
            }
        }

        let input_tensor = Tensor::from_array(([1usize, 3, h, w], tensor_data)).map_err(|err| {
            AppError::ImageProcessingFailure(format!("PP-OCRv5 det input tensor error: {err}"))
        })?;

        let (map_w, map_h, prob_map) = {
            let mut session = self.det_session.lock().map_err(|_| {
                AppError::ImageProcessingFailure("PP-OCRv5 det session mutex poisoned".to_string())
            })?;
            let outputs = session.run(ort::inputs![input_tensor]).map_err(|err| {
                AppError::ImageProcessingFailure(format!("PP-OCRv5 det inference failed: {err}"))
            })?;
            let (shape, raw) = outputs[0].try_extract_tensor::<f32>().map_err(|err| {
                AppError::ImageProcessingFailure(format!(
                    "failed to extract PP-OCRv5 det output: {err}"
                ))
            })?;
            if shape.len() < 4 {
                return Err(AppError::ImageProcessingFailure(format!(
                    "unexpected PP-OCRv5 det output rank {}",
                    shape.len()
                )));
            }
            let out_h = shape[2].max(0) as usize;
            let out_w = shape[3].max(0) as usize;
            if raw.len() < out_h * out_w {
                return Err(AppError::ImageProcessingFailure(
                    "PP-OCRv5 det output buffer shorter than H*W".to_string(),
                ));
            }
            // Slice out only the valid unpadded image portion of the probability map
            let valid_w = ((res_w as f32 / canvas as f32) * out_w as f32)
                .round()
                .clamp(1.0, out_w as f32) as usize;
            let valid_h = ((res_h as f32 / canvas as f32) * out_h as f32)
                .round()
                .clamp(1.0, out_h as f32) as usize;
            let mut cropped = vec![0.0_f32; valid_w * valid_h];
            for y in 0..valid_h {
                let src_row = &raw[y * out_w..y * out_w + valid_w];
                cropped[y * valid_w..(y + 1) * valid_w].copy_from_slice(src_row);
            }
            (valid_w, valid_h, cropped)
        };

        let mut boxes = extract_boxes_from_prob_map(
            &prob_map,
            map_w,
            map_h,
            orig_w,
            orig_h,
            self.config.det_thresh,
            self.config.box_thresh,
            self.config.unclip_ratio,
        );

        // Sort by detection score to prune extreme noise, then sort in natural reading order
        if boxes.len() > self.config.max_text_boxes {
            boxes.sort_by(|a, b| b.det_score.total_cmp(&a.det_score));
            boxes.truncate(self.config.max_text_boxes);
        }
        sort_boxes_reading_order(&mut boxes);
        Ok(boxes)
    }

    fn recognize_text_crop(&self, crop: &RgbImage) -> Result<(String, f32), AppError> {
        let (cw, ch) = crop.dimensions();
        if cw == 0 || ch == 0 {
            return Ok((String::new(), 0.0));
        }

        let target_h = self.rec_height.max(16);
        let aspect = cw as f32 / ch as f32;
        let target_w = ((target_h as f32 * aspect).ceil() as u32).clamp(16, 1600);
        let resized = imageops::resize(crop, target_w, target_h, FilterType::CatmullRom);

        let w = target_w as usize;
        let h = target_h as usize;
        let plane = w * h;
        let mut tensor_data = vec![0.0_f32; 3 * plane];

        // PP-OCRv5 recognition normalization: BGR scaled to [-1.0, 1.0]
        for y in 0..h {
            for x in 0..w {
                let [r, g, b] = resized.get_pixel(x as u32, y as u32).0;
                let idx = y * w + x;
                tensor_data[idx] = (b as f32 / 255.0 - 0.5) / 0.5;
                tensor_data[plane + idx] = (g as f32 / 255.0 - 0.5) / 0.5;
                tensor_data[2 * plane + idx] = (r as f32 / 255.0 - 0.5) / 0.5;
            }
        }

        let input_tensor = Tensor::from_array(([1usize, 3, h, w], tensor_data)).map_err(|err| {
            AppError::ImageProcessingFailure(format!("PP-OCRv5 rec input tensor error: {err}"))
        })?;

        let (time_steps, num_classes, logits) = {
            let mut session = self.rec_session.lock().map_err(|_| {
                AppError::ImageProcessingFailure("PP-OCRv5 rec session mutex poisoned".to_string())
            })?;
            let outputs = session.run(ort::inputs![input_tensor]).map_err(|err| {
                AppError::ImageProcessingFailure(format!("PP-OCRv5 rec inference failed: {err}"))
            })?;
            let (shape, raw) = outputs[0].try_extract_tensor::<f32>().map_err(|err| {
                AppError::ImageProcessingFailure(format!(
                    "failed to extract PP-OCRv5 rec output: {err}"
                ))
            })?;
            if shape.len() != 3 {
                return Err(AppError::ImageProcessingFailure(format!(
                    "unexpected PP-OCRv5 rec output shape rank {}",
                    shape.len()
                )));
            }
            let t = shape[1].max(0) as usize;
            let c = shape[2].max(0) as usize;
            if raw.len() < t * c {
                return Err(AppError::ImageProcessingFailure(
                    "PP-OCRv5 rec output buffer shorter than T*C".to_string(),
                ));
            }
            (t, c, raw[..t * c].to_vec())
        };

        Ok(ctc_greedy_decode(
            &logits,
            time_steps,
            num_classes,
            &self.dictionary,
        ))
    }
}

impl SlateReaderPort for PpOcrV5Engine {
    fn read_slate(&self, source_path: &str) -> Result<Option<SlateOcrReading>, AppError> {
        let (_lines, parsed) = self.read_slate_from_path(source_path)?;
        Ok(parsed.map(|p| SlateOcrReading {
            subject_name: p.subject_name,
            subject_code: p.subject_code,
            ocr_confidence: p.confidence,
            whiteboard_region: p.whiteboard_region,
        }))
    }
}

fn round_to_32(dim: u32) -> u32 {
    let rounded = ((dim as f32 / 32.0).round() as u32) * 32;
    rounded.max(32)
}

fn detect_rec_input_height(session: &Session) -> Option<u32> {
    let shape = session.inputs().first()?.dtype().tensor_shape()?;
    let h = *shape.get(2)?;
    if h > 0 {
        Some(h as u32)
    } else {
        None
    }
}

/// Extracts unclipped, orientation-aware text boxes from a DBNet `[H, W]` probability map using
/// fast iterative 8-connected components + PCA tilt estimation.
#[allow(clippy::too_many_arguments)]
fn extract_boxes_from_prob_map(
    prob_map: &[f32],
    map_w: usize,
    map_h: usize,
    orig_w: u32,
    orig_h: u32,
    det_thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
) -> Vec<DetectedTextBox> {
    if map_w == 0 || map_h == 0 {
        return Vec::new();
    }

    let mut visited = vec![false; map_w * map_h];
    let mut queue = VecDeque::new();
    let mut pts: Vec<(usize, usize)> = Vec::new();
    let mut boxes = Vec::new();
    let scale_x = orig_w as f32 / map_w as f32;
    let scale_y = orig_h as f32 / map_h as f32;

    for sy in 0..map_h {
        for sx in 0..map_w {
            let start_idx = sy * map_w + sx;
            if visited[start_idx] || prob_map[start_idx] < det_thresh {
                continue;
            }

            visited[start_idx] = true;
            queue.clear();
            pts.clear();
            queue.push_back((sx, sy));

            let mut min_x = sx;
            let mut max_x = sx;
            let mut min_y = sy;
            let mut max_y = sy;
            let mut prob_sum = 0.0_f32;
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;
            let mut sum_xx = 0.0_f64;
            let mut sum_yy = 0.0_f64;
            let mut sum_xy = 0.0_f64;

            while let Some((cx, cy)) = queue.pop_front() {
                let idx = cy * map_w + cx;
                let p = prob_map[idx];
                pts.push((cx, cy));
                prob_sum += p;
                min_x = min_x.min(cx);
                max_x = max_x.max(cx);
                min_y = min_y.min(cy);
                max_y = max_y.max(cy);

                let px = (cx as f64 + 0.5) * scale_x as f64;
                let py = (cy as f64 + 0.5) * scale_y as f64;
                sum_x += px;
                sum_y += py;
                sum_xx += px * px;
                sum_yy += py * py;
                sum_xy += px * py;

                let y_lo = cy.saturating_sub(1);
                let y_hi = (cy + 1).min(map_h - 1);
                let x_lo = cx.saturating_sub(1);
                let x_hi = (cx + 1).min(map_w - 1);
                for ny in y_lo..=y_hi {
                    for nx in x_lo..=x_hi {
                        let nidx = ny * map_w + nx;
                        if !visited[nidx] && prob_map[nidx] >= det_thresh {
                            visited[nidx] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }

            let count = pts.len();
            if count < 10 {
                continue;
            }
            let mean_prob = prob_sum / count as f32;
            if mean_prob < box_thresh {
                continue;
            }

            let bw_map = (max_x - min_x + 1) as f32;
            let bh_map = (max_y - min_y + 1) as f32;
            if bw_map < 3.0 || bh_map < 2.0 {
                continue;
            }

            // PCA principal axis in image pixel coordinates — captures even slight 1°-4° whiteboard tilts
            let n = count as f64;
            let mx_px = sum_x / n;
            let my_px = sum_y / n;
            let cov_xx = (sum_xx / n - mx_px * mx_px).max(0.0);
            let cov_yy = (sum_yy / n - my_px * my_px).max(0.0);
            let cov_xy = sum_xy / n - mx_px * my_px;
            let mut angle_rad = 0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy) as f32;

            // Keep short/square glyphs ("P.", "07") axis-aligned so they never rotate sideways
            let trace = cov_xx + cov_yy;
            let disc = ((cov_xx - cov_yy).powi(2) + 4.0 * cov_xy * cov_xy).sqrt();
            let eig_major = ((trace + disc) * 0.5).max(1e-6);
            let eig_minor = ((trace - disc) * 0.5).max(1e-6);
            let elongation = (eig_major / eig_minor).sqrt();
            let abs_deg = angle_rad.to_degrees().abs();
            if elongation < 1.8 || abs_deg > 40.0 {
                angle_rad = 0.0;
            }

            // Project component pixels onto its true oriented frame (u, v) so tilted long lines
            // keep their true thin height instead of inflating via an axis-aligned bounding box!
            let cos_a = angle_rad.cos();
            let sin_a = angle_rad.sin();
            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for &(cx, cy) in &pts {
                let dx = (cx as f32 + 0.5) * scale_x - mx_px as f32;
                let dy = (cy as f32 + 0.5) * scale_y - my_px as f32;
                let u = dx * cos_a + dy * sin_a;
                let v = -dx * sin_a + dy * cos_a;
                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }

            let kernel_w = (max_u - min_u) + scale_x;
            let kernel_h = (max_v - min_v) + scale_y;

            // Exact DBNet Vatti-style unclipping using real component pixel area / OBB perimeter
            let area_px = (count as f32) * scale_x * scale_y;
            let perimeter_px = 2.0 * (kernel_w + kernel_h);
            let expand = (area_px * unclip_ratio / perimeter_px).max(1.5 * scale_y);
            let pad_x = expand * 1.25;
            let pad_y = expand * 1.15;

            let crop_w_px = (kernel_w + 2.0 * pad_x).max(1.0);
            let crop_h_px = (kernel_h + 2.0 * pad_y).max(1.0);
            if crop_w_px < 8.0 || crop_h_px < 6.0 {
                continue;
            }

            // Reconstruct exact OBB center in original image coordinates
            let cu = (min_u + max_u) * 0.5;
            let cv = (min_v + max_v) * 0.5;
            let center_x = mx_px as f32 + cu * cos_a - cv * sin_a;
            let center_y = my_px as f32 + cu * sin_a + cv * cos_a;

            // Compute tight axis-aligned UI overlay box from the 4 rotated OBB corners
            let half_w = crop_w_px * 0.5;
            let half_h = crop_h_px * 0.5;
            let corners = [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ];
            let mut x1 = orig_w as f32;
            let mut y1 = orig_h as f32;
            let mut x2 = 0.0_f32;
            let mut y2 = 0.0_f32;
            for (lu, lv) in corners {
                let wx = (center_x + lu * cos_a - lv * sin_a).clamp(0.0, orig_w as f32);
                let wy = (center_y + lu * sin_a + lv * cos_a).clamp(0.0, orig_h as f32);
                x1 = x1.min(wx);
                y1 = y1.min(wy);
                x2 = x2.max(wx);
                y2 = y2.max(wy);
            }
            let aabb_w = (x2 - x1).max(1.0);
            let aabb_h = (y2 - y1).max(1.0);

            boxes.push(DetectedTextBox {
                region: NormalizedRect {
                    left: (x1 / orig_w as f32).clamp(0.0, 1.0),
                    top: (y1 / orig_h as f32).clamp(0.0, 1.0),
                    width: (aabb_w / orig_w as f32).clamp(0.0, 1.0),
                    height: (aabb_h / orig_h as f32).clamp(0.0, 1.0),
                },
                center_x: center_x.clamp(0.0, orig_w as f32),
                center_y: center_y.clamp(0.0, orig_h as f32),
                width_px: crop_w_px,
                height_px: crop_h_px,
                angle_rad,
                det_score: mean_prob.clamp(0.0, 1.0),
            });
        }
    }

    boxes
}

/// Crops (and bilinear-deskews when tilted) a detected text strip from the source image.
fn extract_deskewed_crop(rgb: &RgbImage, b: &DetectedTextBox) -> RgbImage {
    let (img_w, img_h) = rgb.dimensions();
    if b.angle_rad.abs() <= 0.005 {
        let x1 = (b.center_x - b.width_px * 0.5).floor().clamp(0.0, (img_w - 1) as f32) as u32;
        let y1 = (b.center_y - b.height_px * 0.5).floor().clamp(0.0, (img_h - 1) as f32) as u32;
        let x2 = (b.center_x + b.width_px * 0.5).ceil().clamp((x1 + 1) as f32, img_w as f32) as u32;
        let y2 = (b.center_y + b.height_px * 0.5).ceil().clamp((y1 + 1) as f32, img_h as f32) as u32;
        return imageops::crop_imm(rgb, x1, y1, x2 - x1, y2 - y1).to_image();
    }

    let out_w = (b.width_px.round() as u32).clamp(8, img_w.max(8));
    let out_h = (b.height_px.round() as u32).clamp(8, img_h.max(8));
    let cos_a = b.angle_rad.cos();
    let sin_a = b.angle_rad.sin();
    let half_w = out_w as f32 * 0.5;
    let half_h = out_h as f32 * 0.5;

    let mut out = RgbImage::new(out_w, out_h);
    for dy in 0..out_h {
        let ly = dy as f32 - half_h + 0.5;
        for dx in 0..out_w {
            let lx = dx as f32 - half_w + 0.5;
            let sx = b.center_x + lx * cos_a - ly * sin_a;
            let sy = b.center_y + lx * sin_a + ly * cos_a;
            out.put_pixel(dx, dy, sample_bilinear(rgb, sx, sy));
        }
    }
    out
}

fn sample_bilinear(rgb: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let max_x = (rgb.width() - 1) as f32;
    let max_y = (rgb.height() - 1) as f32;
    let cx = x.clamp(0.0, max_x);
    let cy = y.clamp(0.0, max_y);
    let x0 = cx.floor() as u32;
    let y0 = cy.floor() as u32;
    let x1 = (x0 + 1).min(rgb.width() - 1);
    let y1 = (y0 + 1).min(rgb.height() - 1);
    let tx = cx - x0 as f32;
    let ty = cy - y0 as f32;

    let p00 = rgb.get_pixel(x0, y0).0;
    let p10 = rgb.get_pixel(x1, y0).0;
    let p01 = rgb.get_pixel(x0, y1).0;
    let p11 = rgb.get_pixel(x1, y1).0;

    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - tx) + p10[c] as f32 * tx;
        let bot = p01[c] as f32 * (1.0 - tx) + p11[c] as f32 * tx;
        out[c] = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}

fn sort_boxes_reading_order(boxes: &mut [DetectedTextBox]) {
    boxes.sort_by(|a, b| {
        let avg_h = (a.region.height + b.region.height) * 0.5;
        let dy = (a.region.top - b.region.top).abs();
        if dy <= avg_h * 0.55 {
            a.region.left.total_cmp(&b.region.left)
        } else {
            a.region.top.total_cmp(&b.region.top)
        }
    });
}

/// CTC greedy decoder for PaddleOCR PP-OCRv5 recognition outputs (`[T, C]`).
/// Index `0` is CTC blank, `1..=dict.len()` map directly to `dict[idx - 1]`, and `dict.len() + 1`
/// maps to `' '` (space) when `use_space_char` appends the space class.
pub fn ctc_greedy_decode(
    scores: &[f32],
    time_steps: usize,
    num_classes: usize,
    dictionary: &[String],
) -> (String, f32) {
    if time_steps == 0 || num_classes == 0 {
        return (String::new(), 0.0);
    }

    let mut text = String::new();
    let mut conf_sum = 0.0_f32;
    let mut conf_count = 0usize;
    let mut prev_idx: Option<usize> = None;

    for t in 0..time_steps {
        let row = &scores[t * num_classes..(t + 1) * num_classes];
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        let mut min_val = f32::INFINITY;
        for (c, &v) in row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = c;
            }
            if v < min_val {
                min_val = v;
            }
        }

        // Convert raw logits to probability if output is not already post-softmax
        let prob = if min_val < -1e-3 || best_val > 1.001 {
            let denom: f32 = row.iter().map(|&v| (v - best_val).exp()).sum();
            if denom > 0.0 {
                1.0 / denom
            } else {
                0.0
            }
        } else {
            best_val.clamp(0.0, 1.0)
        };

        if best_idx != 0 && prev_idx != Some(best_idx) {
            if best_idx <= dictionary.len() {
                text.push_str(&dictionary[best_idx - 1]);
                conf_sum += prob;
                conf_count += 1;
            } else if best_idx == dictionary.len() + 1 {
                text.push(' ');
                conf_sum += prob;
                conf_count += 1;
            }
        }
        prev_idx = Some(best_idx);
    }

    let avg_conf = if conf_count > 0 {
        conf_sum / conf_count as f32
    } else {
        0.0
    };
    (text, avg_conf)
}

fn build_cpu_session(
    env: &ort::environment::Environment,
    model_path: &Path,
    intra_threads: usize,
) -> Result<Session, AppError> {
    Session::builder(env)
        .map_err(|err| AppError::ImageProcessingFailure(format!("PP-OCRv5 builder error: {err}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| AppError::ImageProcessingFailure(format!("PP-OCRv5 opt error: {err}")))?
        .with_intra_threads(intra_threads.max(1))
        .map_err(|err| AppError::ImageProcessingFailure(format!("PP-OCRv5 threads error: {err}")))?
        .commit_from_file(model_path)
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!(
                "failed to load PP-OCRv5 ONNX model '{}': {err}",
                model_path.display()
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_greedy_decode_collapses_repeats_skips_blank_and_maps_space() {
        let dict = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        // num_classes = 5: 0=blank, 1='A', 2='B', 3='C', 4=' '
        let mut probs = vec![0.0_f32; 7 * 5];
        let seq = [1usize, 1, 0, 1, 4, 2, 3]; // "AA" (blank) "A" " " "B" "C" -> "AA BC"
        for (t, &idx) in seq.iter().enumerate() {
            probs[t * 5 + idx] = 0.95;
        }
        let (decoded, conf) = ctc_greedy_decode(&probs, 7, 5, &dict);
        assert_eq!(decoded, "AA BC");
        assert!((conf - 0.95).abs() < 1e-4);
    }

    #[test]
    fn prob_map_box_extractor_finds_and_sorts_two_stacked_lines() {
        let w = 160;
        let h = 160;
        let mut map = vec![0.0_f32; w * h];
        // Line 1: y in 30..42, x in 20..120
        for y in 30..42 {
            for x in 20..120 {
                map[y * w + x] = 0.92;
            }
        }
        // Line 2: y in 70..82, x in 35..110
        for y in 70..82 {
            for x in 35..110 {
                map[y * w + x] = 0.88;
            }
        }
        let mut boxes = extract_boxes_from_prob_map(&map, w, h, 640, 640, 0.3, 0.55, 1.5);
        sort_boxes_reading_order(&mut boxes);
        assert_eq!(boxes.len(), 2);
        assert!(boxes[0].region.top < boxes[1].region.top);
        assert!(boxes[0].det_score > 0.9);
    }

    #[test]
    fn tilted_long_text_line_keeps_tight_oriented_height_instead_of_inflating_aabb() {
        let w = 240;
        let h = 160;
        let mut map = vec![0.0_f32; w * h];
        // Draw a 160px-wide, 6px-thick text kernel tilted downward by ~5.7 degrees (slope 0.1)
        for x in 30..190 {
            let cy = 60.0 + (x as f32 - 30.0) * 0.1;
            for dy in -3..=3 {
                let y = (cy.round() as isize + dy).clamp(0, (h - 1) as isize) as usize;
                map[y * w + x] = 0.92;
            }
        }
        let boxes = extract_boxes_from_prob_map(&map, w, h, 480, 320, 0.30, 0.55, 1.55);
        assert_eq!(boxes.len(), 1);
        let b = &boxes[0];
        // True angle should be detected (~5.7 deg) and OBB crop height should stay slim (< 45px in 320px image)
        assert!(
            b.angle_rad.to_degrees().abs() > 3.0,
            "expected detected tilt > 3 deg, got {:.2}",
            b.angle_rad.to_degrees()
        );
        assert!(
            b.height_px < 45.0,
            "expected tight oriented crop height < 45px, got {:.1}px",
            b.height_px
        );
        assert!(
            b.width_px / b.height_px > 7.5,
            "expected elongated aspect ratio > 7.5:1, got {:.2}:1",
            b.width_px / b.height_px
        );
    }
}
