use crate::develop::image_decoder;
use crate::vision::face_align::{align_face_112, SFACE_INPUT_SIZE};
use crate::vision::onnx_runtime::shared_ort_environment;
use image::RgbImage;
use kroma_app::error::AppError;
use kroma_app::ports::vision::{FaceRecognizerPort, RawFaceDetection};
use kroma_domain::identity::{
    FaceEmbedding, YuNetLandmarks, ARCFACE_EMBEDDING_DIM, SFACE_EMBEDDING_DIM,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

/// Preprocessing and output contract for supported ONNX face recognition models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceEmbeddingModelProfile {
    /// InsightFace `w600k_r50.onnx` (ArcFace ResNet-50, 512-D, RGB normalized to `[-1, 1]`).
    ArcFaceRgbNorm512,
    /// OpenCV Zoo `face_recognition_sface_2021dec.onnx` (SFace, 128-D, RGB `[0..255]` with internal Sub/Mul).
    SFaceRgb255_128,
}

impl FaceEmbeddingModelProfile {
    pub fn embedding_dim(self) -> usize {
        match self {
            Self::ArcFaceRgbNorm512 => ARCFACE_EMBEDDING_DIM,
            Self::SFaceRgb255_128 => SFACE_EMBEDDING_DIM,
        }
    }

    pub fn display_label(self) -> &'static str {
        match self {
            Self::ArcFaceRgbNorm512 => "ArcFace R50 512-D",
            Self::SFaceRgb255_128 => "SFace 128-D",
        }
    }
}

/// Real ONNX Runtime adapter supporting both InsightFace ArcFace ResNet-50 (`w600k_r50.onnx`, 512-D)
/// and OpenCV Zoo SFace (`face_recognition_sface_2021dec.onnx`, 128-D) behind `FaceRecognizerPort`.
///
/// Automatically inspects the ONNX model's output dimension so higher-accuracy 512-D ArcFace models
/// work out of the box alongside 128-D SFace fallbacks using the same 5-point `112x112` Umeyama alignment.
pub struct SFaceOnnxRecognizer {
    session: Mutex<Session>,
    profile: FaceEmbeddingModelProfile,
    using_gpu: bool,
    supports_dynamic_batch: bool,
}

impl SFaceOnnxRecognizer {
    pub fn from_model_file(
        model_path: impl AsRef<Path>,
        intra_threads: usize,
    ) -> Result<Self, AppError> {
        let path_ref = model_path.as_ref();
        let env = shared_ort_environment()?;

        // Prefer hardware GPU/NPU acceleration (DirectML on Windows, CoreML on macOS) with CPU fallback
        let (session, using_gpu) =
            match crate::vision::onnx_runtime::try_build_gpu_session(env, path_ref) {
                Some(gpu_session) => (gpu_session, true),
                None => (build_cpu_session(env, path_ref, intra_threads)?, false),
            };

        let profile = detect_model_profile(&session, path_ref);
        let supports_dynamic_batch = detect_dynamic_batch_support(&session);

        Ok(Self {
            session: Mutex::new(session),
            profile,
            using_gpu,
            supports_dynamic_batch,
        })
    }

    pub fn profile(&self) -> FaceEmbeddingModelProfile {
        self.profile
    }

    pub fn model_label(&self) -> &'static str {
        self.profile.display_label()
    }

    pub fn execution_provider_label(&self) -> &'static str {
        if self.using_gpu {
            crate::vision::onnx_runtime::gpu_execution_provider_label()
        } else {
            "CPU"
        }
    }

    pub fn embedding_dim(&self) -> usize {
        self.profile.embedding_dim()
    }

    pub fn extract_embedding_from_rgb(
        &self,
        rgb: &RgbImage,
        landmarks: &YuNetLandmarks,
    ) -> Result<FaceEmbedding, AppError> {
        let aligned = align_face_112(rgb, landmarks).ok_or_else(|| {
            AppError::ImageProcessingFailure(
                "failed to estimate 5-point similarity alignment for face crop".to_string(),
            )
        })?;
        let mut results = self.extract_embeddings_batch_from_aligned(&[aligned])?;
        results.pop().ok_or_else(|| {
            AppError::ImageProcessingFailure("face recognizer produced no embedding".to_string())
        })
    }

    /// Runs high-throughput batched ONNX inference over pre-aligned `112x112` RGB face crops.
    /// Uses dynamic `[B, 3, 112, 112]` batching for ArcFace (`w600k_r50.onnx`) for major SIMD/GPU speedups,
    /// and cleanly steps `B=1` for fixed-batch SFace models.
    pub fn extract_embeddings_batch_from_aligned(
        &self,
        aligned_crops: &[RgbImage],
    ) -> Result<Vec<FaceEmbedding>, AppError> {
        if aligned_crops.is_empty() {
            return Ok(Vec::new());
        }

        let dim = SFACE_INPUT_SIZE as usize;
        let plane = dim * dim;
        let per_image = 3 * plane;
        let expected_dim = self.profile.embedding_dim();

        let mut session = self.session.lock().map_err(|_| {
            AppError::ImageProcessingFailure(
                "face recognizer ONNX session mutex poisoned".to_string(),
            )
        })?;

        match self.profile {
            FaceEmbeddingModelProfile::ArcFaceRgbNorm512 if self.supports_dynamic_batch => {
                let batch = aligned_crops.len();
                let mut tensor_data = vec![0.0_f32; batch * per_image];
                for (b_idx, aligned) in aligned_crops.iter().enumerate() {
                    pack_arcface_rgb_norm(aligned, &mut tensor_data[b_idx * per_image..]);
                }

                let input_tensor = Tensor::from_array(([batch, 3, dim, dim], tensor_data))
                    .map_err(|err| {
                        AppError::ImageProcessingFailure(format!(
                            "ArcFace batched input tensor error: {err}"
                        ))
                    })?;
                let outputs = session.run(ort::inputs![input_tensor]).map_err(|err| {
                    AppError::ImageProcessingFailure(format!(
                        "ArcFace batched inference failed: {err}"
                    ))
                })?;
                let (_shape, raw_slice) =
                    outputs[0].try_extract_tensor::<f32>().map_err(|err| {
                        AppError::ImageProcessingFailure(format!(
                            "failed to extract batched ArcFace embedding tensor: {err}"
                        ))
                    })?;

                if raw_slice.len() < batch * expected_dim {
                    return Err(AppError::ImageProcessingFailure(format!(
                        "expected {} floats for batch of {batch}, got {}",
                        batch * expected_dim,
                        raw_slice.len()
                    )));
                }

                let mut embeddings = Vec::with_capacity(batch);
                for b_idx in 0..batch {
                    let start = b_idx * expected_dim;
                    embeddings.push(FaceEmbedding::from_vec(
                        raw_slice[start..start + expected_dim].to_vec(),
                    )?);
                }
                Ok(embeddings)
            }
            FaceEmbeddingModelProfile::ArcFaceRgbNorm512 => {
                // Model exports static [1, 512] output (e.g. w600k_r50.onnx under DirectML):
                // stream B=1 directly on the GPU/CPU session while holding the lock once per chunk.
                let mut embeddings = Vec::with_capacity(aligned_crops.len());
                let mut tensor_data = vec![0.0_f32; per_image];
                for aligned in aligned_crops {
                    pack_arcface_rgb_norm(aligned, &mut tensor_data);
                    let input_tensor =
                        Tensor::from_array(([1usize, 3, dim, dim], tensor_data.clone()))
                            .map_err(|err| {
                                AppError::ImageProcessingFailure(format!(
                                    "ArcFace input tensor error: {err}"
                                ))
                            })?;
                    let outputs = session.run(ort::inputs![input_tensor]).map_err(|err| {
                        AppError::ImageProcessingFailure(format!("ArcFace inference failed: {err}"))
                    })?;
                    let (_shape, raw_slice) =
                        outputs[0].try_extract_tensor::<f32>().map_err(|err| {
                            AppError::ImageProcessingFailure(format!(
                                "failed to extract ArcFace embedding tensor: {err}"
                            ))
                        })?;
                    if raw_slice.len() < expected_dim {
                        return Err(AppError::ImageProcessingFailure(format!(
                            "expected {expected_dim}-D ArcFace embedding, got {} elements",
                            raw_slice.len()
                        )));
                    }
                    embeddings.push(FaceEmbedding::from_vec(
                        raw_slice[..expected_dim].to_vec(),
                    )?);
                }
                Ok(embeddings)
            }
            FaceEmbeddingModelProfile::SFaceRgb255_128 => {
                let mut embeddings = Vec::with_capacity(aligned_crops.len());
                let mut tensor_data = vec![0.0_f32; per_image];
                for aligned in aligned_crops {
                    for y in 0..dim {
                        for x in 0..dim {
                            let [r, g, b] = aligned.get_pixel(x as u32, y as u32).0;
                            let offset = y * dim + x;
                            tensor_data[offset] = r as f32;
                            tensor_data[plane + offset] = g as f32;
                            tensor_data[2 * plane + offset] = b as f32;
                        }
                    }
                    let input_tensor =
                        Tensor::from_array(([1usize, 3, dim, dim], tensor_data.clone()))
                            .map_err(|err| {
                                AppError::ImageProcessingFailure(format!(
                                    "SFace input tensor error: {err}"
                                ))
                            })?;
                    let outputs = session.run(ort::inputs![input_tensor]).map_err(|err| {
                        AppError::ImageProcessingFailure(format!("SFace inference failed: {err}"))
                    })?;
                    let (_shape, raw_slice) =
                        outputs[0].try_extract_tensor::<f32>().map_err(|err| {
                            AppError::ImageProcessingFailure(format!(
                                "failed to extract SFace embedding tensor: {err}"
                            ))
                        })?;
                    if raw_slice.len() < expected_dim {
                        return Err(AppError::ImageProcessingFailure(format!(
                            "expected {expected_dim}-D SFace embedding, got {} elements",
                            raw_slice.len()
                        )));
                    }
                    embeddings.push(FaceEmbedding::from_vec(
                        raw_slice[..expected_dim].to_vec(),
                    )?);
                }
                Ok(embeddings)
            }
        }
    }
}

impl FaceRecognizerPort for SFaceOnnxRecognizer {
    fn extract_embedding(
        &self,
        source_path: &str,
        detection: &RawFaceDetection,
    ) -> Result<FaceEmbedding, AppError> {
        let preview_rgb = image_decoder::decode_fast_preview_rgb(source_path)?;
        self.extract_embedding_from_rgb(&preview_rgb, &detection.landmarks)
    }
}

fn pack_arcface_rgb_norm(aligned: &RgbImage, dst: &mut [f32]) {
    let dim = SFACE_INPUT_SIZE as usize;
    let plane = dim * dim;
    for y in 0..dim {
        for x in 0..dim {
            let [r, g, b] = aligned.get_pixel(x as u32, y as u32).0;
            let offset = y * dim + x;
            dst[offset] = (r as f32 - 127.5) / 127.5;
            dst[plane + offset] = (g as f32 - 127.5) / 127.5;
            dst[2 * plane + offset] = (b as f32 - 127.5) / 127.5;
        }
    }
}

/// Only enables `B > 1` tensor batching when BOTH input and output ONNX tensors declare
/// dynamic batch dimension `-1` (prevents DirectML `BatchNormalization` shape mismatch on
/// `w600k_r50.onnx` whose output tensor is exported as static `[1, 512]`).
fn detect_dynamic_batch_support(session: &Session) -> bool {
    let in_dyn = session
        .inputs()
        .first()
        .and_then(|i| i.dtype().tensor_shape())
        .and_then(|s| s.first().copied())
        .is_some_and(|d| d < 0);
    let out_dyn = session
        .outputs()
        .first()
        .and_then(|o| o.dtype().tensor_shape())
        .and_then(|s| s.first().copied())
        .is_some_and(|d| d < 0);
    in_dyn && out_dyn
}

fn build_cpu_session(
    env: &ort::environment::Environment,
    model_path: &Path,
    intra_threads: usize,
) -> Result<Session, AppError> {
    Session::builder(env)
        .map_err(|err| AppError::ImageProcessingFailure(format!("Face recognizer builder error: {err}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| AppError::ImageProcessingFailure(format!("Face recognizer opt error: {err}")))?
        .with_intra_threads(intra_threads.max(1))
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!("Face recognizer threads error: {err}"))
        })?
        .commit_from_file(model_path)
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!(
                "failed to load face recognition ONNX model '{}': {err}",
                model_path.display()
            ))
        })
}

fn detect_model_profile(session: &Session, model_path: &Path) -> FaceEmbeddingModelProfile {
    if let Some(first_output) = session.outputs().first()
        && let Some(shape) = first_output.dtype().tensor_shape()
    {
        if shape.contains(&(ARCFACE_EMBEDDING_DIM as i64)) {
            return FaceEmbeddingModelProfile::ArcFaceRgbNorm512;
        }
        if shape.contains(&(SFACE_EMBEDDING_DIM as i64)) {
            return FaceEmbeddingModelProfile::SFaceRgb255_128;
        }
    }

    let lower_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if lower_name.contains("w600k")
        || lower_name.contains("arcface")
        || lower_name.contains("glintr")
    {
        FaceEmbeddingModelProfile::ArcFaceRgbNorm512
    } else {
        FaceEmbeddingModelProfile::SFaceRgb255_128
    }
}
