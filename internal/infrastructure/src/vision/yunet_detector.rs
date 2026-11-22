use crate::develop::image_decoder;
use crate::vision::onnx_runtime::shared_ort_environment;
use crate::vision::yunet_decode::{decode_yunet_outputs, YuNetDecodeSpec, YuNetStrideOutputs};
use image::imageops::{self, FilterType};
use image::RgbImage;
use kroma_app::error::AppError;
use kroma_app::ports::vision::{FaceDetectorPort, RawFaceDetection};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub struct YuNetConfig {
    /// Longest edge of the resized input image before padding to multiples of 32 (default 640).
    pub max_input_dim: u32,
    pub score_threshold: f32,
    pub nms_iou_threshold: f32,
    pub top_k: usize,
    pub intra_threads: usize,
}

impl Default for YuNetConfig {
    fn default() -> Self {
        Self {
            max_input_dim: 640,
            score_threshold: 0.65,
            nms_iou_threshold: 0.30,
            top_k: 500,
            intra_threads: 4,
        }
    }
}

/// Real ONNX Runtime adapter for OpenCV Zoo `face_detection_yunet_2023mar.onnx`.
pub struct YuNetOnnxDetector {
    session: Mutex<Session>,
    config: YuNetConfig,
}

impl YuNetOnnxDetector {
    pub fn from_model_file(
        model_path: impl AsRef<Path>,
        config: YuNetConfig,
    ) -> Result<Self, AppError> {
        let env = shared_ort_environment()?;
        let session = Session::builder(env)
            .map_err(|err| AppError::ImageProcessingFailure(format!("YuNet builder error: {err}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|err| AppError::ImageProcessingFailure(format!("YuNet opt error: {err}")))?
            .with_intra_threads(config.intra_threads.max(1))
            .map_err(|err| AppError::ImageProcessingFailure(format!("YuNet threads error: {err}")))?
            .commit_from_file(model_path.as_ref())
            .map_err(|err| {
                AppError::ImageProcessingFailure(format!(
                    "failed to load YuNet ONNX model '{}': {err}",
                    model_path.as_ref().display()
                ))
            })?;

        Ok(Self {
            session: Mutex::new(session),
            config,
        })
    }

    pub fn detect_faces_in_rgb(&self, rgb: &RgbImage) -> Result<Vec<RawFaceDetection>, AppError> {
        let (orig_w, orig_h) = rgb.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Ok(Vec::new());
        }

        // Resize proportionally to fit `max_input_dim` while keeping aspect ratio intact
        let max_side = orig_w.max(orig_h);
        let (resized, res_w, res_h) = if max_side > self.config.max_input_dim {
            let scale = (self.config.max_input_dim as f32) / (max_side as f32);
            let nw = ((orig_w as f32) * scale).round().max(1.0) as u32;
            let nh = ((orig_h as f32) * scale).round().max(1.0) as u32;
            (imageops::resize(rgb, nw, nh, FilterType::Triangle), nw, nh)
        } else {
            (rgb.clone(), orig_w, orig_h)
        };

        // YuNet FPN requires width & height divisible by 32
        let pad_w = align_up_32(res_w) as usize;
        let pad_h = align_up_32(res_h) as usize;

        // Pack NCHW [1, 3, pad_h, pad_w] in BGR [0..255] matching OpenCV FaceDetectorYNImpl
        let plane = pad_w * pad_h;
        let mut tensor_data = vec![0.0_f32; 3 * plane];
        for y in 0..(res_h as usize) {
            for x in 0..(res_w as usize) {
                let [r, g, b] = resized.get_pixel(x as u32, y as u32).0;
                let offset = y * pad_w + x;
                tensor_data[offset] = b as f32;
                tensor_data[plane + offset] = g as f32;
                tensor_data[2 * plane + offset] = r as f32;
            }
        }

        let input_tensor = Tensor::from_array(([1usize, 3, pad_h, pad_w], tensor_data)).map_err(
            |err| AppError::ImageProcessingFailure(format!("YuNet input tensor error: {err}")),
        )?;

        let mut session = self.session.lock().map_err(|_| {
            AppError::ImageProcessingFailure("YuNet ONNX session mutex poisoned".to_string())
        })?;

        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|err| AppError::ImageProcessingFailure(format!("YuNet inference failed: {err}")))?;

        let cls_8 = extract_f32_slice(&outputs, "cls_8")?;
        let cls_16 = extract_f32_slice(&outputs, "cls_16")?;
        let cls_32 = extract_f32_slice(&outputs, "cls_32")?;
        let obj_8 = extract_f32_slice(&outputs, "obj_8")?;
        let obj_16 = extract_f32_slice(&outputs, "obj_16")?;
        let obj_32 = extract_f32_slice(&outputs, "obj_32")?;
        let bbox_8 = extract_f32_slice(&outputs, "bbox_8")?;
        let bbox_16 = extract_f32_slice(&outputs, "bbox_16")?;
        let bbox_32 = extract_f32_slice(&outputs, "bbox_32")?;
        let kps_8 = extract_f32_slice(&outputs, "kps_8")?;
        let kps_16 = extract_f32_slice(&outputs, "kps_16")?;
        let kps_32 = extract_f32_slice(&outputs, "kps_32")?;

        let pyramid = [
            YuNetStrideOutputs {
                stride: 8,
                cls: cls_8,
                obj: obj_8,
                bbox: bbox_8,
                kps: kps_8,
            },
            YuNetStrideOutputs {
                stride: 16,
                cls: cls_16,
                obj: obj_16,
                bbox: bbox_16,
                kps: kps_16,
            },
            YuNetStrideOutputs {
                stride: 32,
                cls: cls_32,
                obj: obj_32,
                bbox: bbox_32,
                kps: kps_32,
            },
        ];

        Ok(decode_yunet_outputs(
            &pyramid,
            YuNetDecodeSpec {
                pad_w,
                pad_h,
                resized_w: res_w as usize,
                resized_h: res_h as usize,
                score_threshold: self.config.score_threshold,
                nms_iou_threshold: self.config.nms_iou_threshold,
                top_k: self.config.top_k,
            },
        ))
    }
}

impl FaceDetectorPort for YuNetOnnxDetector {
    fn detect_faces(&self, source_path: &str) -> Result<Vec<RawFaceDetection>, AppError> {
        let preview_rgb = image_decoder::decode_fast_preview_rgb(source_path)?;
        self.detect_faces_in_rgb(&preview_rgb)
    }
}

fn align_up_32(v: u32) -> u32 {
    v.div_ceil(32).max(1) * 32
}

fn extract_f32_slice<'a>(
    outputs: &'a ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<&'a [f32], AppError> {
    let value = outputs.get(name).ok_or_else(|| {
        AppError::ImageProcessingFailure(format!("YuNet ONNX output tensor '{name}' not found"))
    })?;
    let (_shape, slice) = value.try_extract_tensor::<f32>().map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "failed to extract f32 tensor '{name}' from YuNet: {err}"
        ))
    })?;
    Ok(slice)
}
