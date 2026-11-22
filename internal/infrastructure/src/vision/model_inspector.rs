use crate::vision::onnx_runtime::shared_ort_environment;
use kroma_domain::identity::{ARCFACE_EMBEDDING_DIM, SFACE_EMBEDDING_DIM};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::fs;
use std::path::Path;

const REQUIRED_YUNET_OUTPUTS: [&str; 12] = [
    "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
];

const KNOWN_RECOGNIZER_HINTS: [&str; 13] = [
    "w600k",
    "arcface",
    "glintr",
    "adaface",
    "magface",
    "cosface",
    "mobilefacenet",
    "mbf",
    "r50",
    "r100",
    "buffalo",
    "sface",
    "insightface",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    Detector,
    Recognizer,
    Segmenter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCompatibility {
    Compatible,
    Warning,
    Incompatible,
}

impl ModelCompatibility {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Compatible => "compatible",
            Self::Warning => "warning",
            Self::Incompatible => "incompatible",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInspectionReport {
    pub compatibility: ModelCompatibility,
    pub summary: String,
    pub warnings: Vec<String>,
    pub detected_label: String,
    pub input_shape: String,
    pub output_shape: String,
    pub embedding_dim: Option<usize>,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct TensorSignature {
    pub name: String,
    pub shape: Vec<i64>,
}

/// Inspects an `.onnx` file by loading its graph metadata with ONNX Runtime (optimization disabled
/// for fast ~30ms verification) and checking Kroma's strict detector/recognizer tensor contracts.
pub fn inspect_onnx_model(path: &Path, role: ModelRole) -> ModelInspectionReport {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.onnx");

    let ext_ok = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
    if !ext_ok {
        return ModelInspectionReport {
            compatibility: ModelCompatibility::Incompatible,
            summary: "Selected file is not an .onnx model.".to_string(),
            warnings: vec!["Kroma only supports ONNX (.onnx) neural network files.".to_string()],
            detected_label: file_name.to_string(),
            input_shape: "—".to_string(),
            output_shape: "—".to_string(),
            embedding_dim: None,
            file_size_bytes: 0,
        };
    }

    let Ok(meta) = fs::metadata(path) else {
        return ModelInspectionReport {
            compatibility: ModelCompatibility::Incompatible,
            summary: "Model file could not be found on disk.".to_string(),
            warnings: vec![format!("Path not readable: {}", path.display())],
            detected_label: file_name.to_string(),
            input_shape: "—".to_string(),
            output_shape: "—".to_string(),
            embedding_dim: None,
            file_size_bytes: 0,
        };
    };
    let file_size_bytes = meta.len();

    let env = match shared_ort_environment() {
        Ok(env) => env,
        Err(err) => {
            return ModelInspectionReport {
                compatibility: ModelCompatibility::Incompatible,
                summary: "ONNX Runtime failed to initialize.".to_string(),
                warnings: vec![err.to_string()],
                detected_label: file_name.to_string(),
                input_shape: "—".to_string(),
                output_shape: "—".to_string(),
                embedding_dim: None,
                file_size_bytes,
            };
        }
    };

    let build_session = || -> Result<Session, String> {
        let builder = Session::builder(env).map_err(|e| e.to_string())?;
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|e| e.to_string())?;
        let mut builder = builder.with_intra_threads(1).map_err(|e| e.to_string())?;
        builder.commit_from_file(path).map_err(|e| e.to_string())
    };
    let session = match build_session() {
        Ok(s) => s,
        Err(err) => {
            return ModelInspectionReport {
                compatibility: ModelCompatibility::Incompatible,
                summary: "File is not a valid or supported ONNX model.".to_string(),
                warnings: vec![format!("ONNX Runtime error: {err}")],
                detected_label: file_name.to_string(),
                input_shape: "—".to_string(),
                output_shape: "—".to_string(),
                embedding_dim: None,
                file_size_bytes,
            };
        }
    };

    let inputs: Vec<TensorSignature> = session
        .inputs()
        .iter()
        .map(|i| TensorSignature {
            name: i.name().to_string(),
            shape: i
                .dtype()
                .tensor_shape()
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default(),
        })
        .collect();

    let outputs: Vec<TensorSignature> = session
        .outputs()
        .iter()
        .map(|o| TensorSignature {
            name: o.name().to_string(),
            shape: o
                .dtype()
                .tensor_shape()
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default(),
        })
        .collect();

    match role {
        ModelRole::Detector => {
            evaluate_detector_contract(file_name, file_size_bytes, &inputs, &outputs)
        }
        ModelRole::Recognizer => {
            evaluate_recognizer_contract(file_name, file_size_bytes, &inputs, &outputs)
        }
        ModelRole::Segmenter => {
            evaluate_segmenter_contract(file_name, file_size_bytes, &inputs, &outputs)
        }
    }
}

const KNOWN_SEGMENTER_HINTS: [&str; 9] = [
    "rmbg",
    "modnet",
    "birefnet",
    "u2net",
    "isnet",
    "silueta",
    "matting",
    "seg",
    "portrait",
];

pub fn evaluate_segmenter_contract(
    file_name: &str,
    file_size_bytes: u64,
    inputs: &[TensorSignature],
    outputs: &[TensorSignature],
) -> ModelInspectionReport {
    let input_shape = inputs
        .first()
        .map(|i| format_shape(&i.shape))
        .unwrap_or_else(|| "none".to_string());
    let output_shape = outputs
        .first()
        .map(|o| format_shape(&o.shape))
        .unwrap_or_else(|| "none".to_string());

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if inputs.is_empty() {
        errors.push("Model declares no input tensors.".to_string());
    } else {
        let shape = &inputs[0].shape;
        if shape.len() != 4 {
            errors.push(format!(
                "Expected 4D NCHW image input [N, 3, H, W], found {}.",
                format_shape(shape)
            ));
        } else if shape[1] != 3 && shape[1] > 0 {
            errors.push(format!(
                "Expected 3-channel RGB image input [N, 3, H, W], found {} channels.",
                shape[1]
            ));
        }
    }

    if outputs.is_empty() {
        errors.push("Model declares no output tensors.".to_string());
    } else {
        let shape = &outputs[0].shape;
        let valid_spatial = match shape.len() {
            4 => (shape[2] < 0 || shape[2] >= 32) && (shape[3] < 0 || shape[3] >= 32),
            3 => (shape[1] < 0 || shape[1] >= 32) && (shape[2] < 0 || shape[2] >= 32),
            2 => (shape[0] < 0 || shape[0] >= 32) && (shape[1] < 0 || shape[1] >= 32),
            _ => false,
        };
        if !valid_spatial {
            errors.push(format!(
                "Expected spatial mask output [N, 1, H, W] or [N, H, W], found {}.",
                format_shape(shape)
            ));
        }
    }

    let lower = file_name.to_ascii_lowercase();
    let label = if lower.contains("rmbg-2") || lower.contains("rmbg_2") || lower.contains("rmbg2") {
        "RMBG-2.0 Subject Matte".to_string()
    } else if lower.contains("rmbg") {
        "RMBG-1.4 Subject Matte".to_string()
    } else if lower.contains("birefnet") {
        "BiRefNet High-Res Matte".to_string()
    } else if lower.contains("modnet") {
        "MODNet Portrait Matte".to_string()
    } else if lower.contains("isnet") {
        "IS-Net Salient Matte".to_string()
    } else if lower.contains("u2net") {
        "U2-Net Subject Matte".to_string()
    } else if lower.contains("silueta") {
        "Silueta Subject Matte".to_string()
    } else {
        friendly_stem(file_name)
    };

    if !errors.is_empty() {
        return ModelInspectionReport {
            compatibility: ModelCompatibility::Incompatible,
            summary: "Not a compatible image segmentation / matting ONNX model.".to_string(),
            warnings: errors,
            detected_label: label,
            input_shape,
            output_shape,
            embedding_dim: None,
            file_size_bytes,
        };
    }

    if !KNOWN_SEGMENTER_HINTS.iter().any(|hint| lower.contains(hint)) {
        warnings.push(
            "Filename does not match a known matting family (RMBG / MODNet / BiRefNet / IS-Net / U2-Net), but tensor shapes look compatible."
                .to_string(),
        );
        return ModelInspectionReport {
            compatibility: ModelCompatibility::Warning,
            summary: "Tensor shapes look compatible, but verify mask output quality after activating.".to_string(),
            warnings,
            detected_label: label,
            input_shape,
            output_shape,
            embedding_dim: None,
            file_size_bytes,
        };
    }

    ModelInspectionReport {
        compatibility: ModelCompatibility::Compatible,
        summary: format!("Verified compatible ONNX subject matting model ({label})."),
        warnings: Vec::new(),
        detected_label: label,
        input_shape,
        output_shape,
        embedding_dim: None,
        file_size_bytes,
    }
}

pub fn evaluate_detector_contract(
    file_name: &str,
    file_size_bytes: u64,
    inputs: &[TensorSignature],
    outputs: &[TensorSignature],
) -> ModelInspectionReport {
    let input_shape = inputs
        .first()
        .map(|i| format_shape(&i.shape))
        .unwrap_or_else(|| "none".to_string());
    let output_shape = if outputs.is_empty() {
        "none".to_string()
    } else if outputs.len() == 1 {
        format_shape(&outputs[0].shape)
    } else {
        format!("{} tensors", outputs.len())
    };

    let mut errors = Vec::new();
    if inputs.len() != 1 {
        errors.push(format!(
            "Expected 1 image input tensor, found {}.",
            inputs.len()
        ));
    } else {
        let shape = &inputs[0].shape;
        if shape.len() != 4 {
            errors.push(format!(
                "Expected 4D NCHW input [N, 3, H, W], found {}.",
                format_shape(shape)
            ));
        } else {
            if shape[1] != 3 && shape[1] > 0 {
                errors.push(format!(
                    "Expected 3 color channels [N, 3, H, W], found {} channels.",
                    shape[1]
                ));
            }
            if (shape[2] > 0 && shape[2] % 32 != 0) || (shape[3] > 0 && shape[3] % 32 != 0) {
                errors.push(format!(
                    "YuNet requires spatial dimensions divisible by 32 (or dynamic), found {}.",
                    format_shape(shape)
                ));
            }
        }
    }

    let missing_heads: Vec<&str> = REQUIRED_YUNET_OUTPUTS
        .iter()
        .copied()
        .filter(|expected| !outputs.iter().any(|o| o.name == *expected))
        .collect();

    if !missing_heads.is_empty() {
        errors.push(format!(
            "Missing YuNet detection heads ({} of 12 missing, e.g. {}). Only YuNet-architecture face detectors with 5-point landmarks are supported.",
            missing_heads.len(),
            missing_heads[..missing_heads.len().min(3)].join(", ")
        ));
    }

    if !errors.is_empty() {
        return ModelInspectionReport {
            compatibility: ModelCompatibility::Incompatible,
            summary: "Incompatible face detector architecture.".to_string(),
            warnings: errors,
            detected_label: friendly_stem(file_name),
            input_shape,
            output_shape,
            embedding_dim: None,
            file_size_bytes,
        };
    }

    ModelInspectionReport {
        compatibility: ModelCompatibility::Compatible,
        summary: "Verified YuNet 12-head face + 5-landmark detector.".to_string(),
        warnings: Vec::new(),
        detected_label: friendly_detector_label(file_name),
        input_shape,
        output_shape: "12 YuNet heads (cls/obj/bbox/kps)".to_string(),
        embedding_dim: None,
        file_size_bytes,
    }
}

pub fn evaluate_recognizer_contract(
    file_name: &str,
    file_size_bytes: u64,
    inputs: &[TensorSignature],
    outputs: &[TensorSignature],
) -> ModelInspectionReport {
    let input_shape = inputs
        .first()
        .map(|i| format_shape(&i.shape))
        .unwrap_or_else(|| "none".to_string());
    let output_shape = outputs
        .first()
        .map(|o| format_shape(&o.shape))
        .unwrap_or_else(|| "none".to_string());

    let mut errors = Vec::new();
    if inputs.len() != 1 {
        errors.push(format!(
            "Expected 1 aligned face input tensor, found {}.",
            inputs.len()
        ));
    } else {
        let shape = &inputs[0].shape;
        if shape.len() != 4 {
            errors.push(format!(
                "Expected 4D input [N, 3, 112, 112], found {}.",
                format_shape(shape)
            ));
        } else {
            let channels_ok = shape[1] == 3 || shape[1] < 0;
            let height_ok = shape[2] == 112 || shape[2] < 0;
            let width_ok = shape[3] == 112 || shape[3] < 0;
            if !channels_ok || !height_ok || !width_ok {
                errors.push(format!(
                    "Expected 112×112 RGB aligned face input [N, 3, 112, 112], found {}.",
                    format_shape(shape)
                ));
            }
        }
    }

    let mut detected_dim: Option<usize> = None;
    if outputs.is_empty() {
        errors.push("Model has no output embedding tensor.".to_string());
    } else {
        let shape = &outputs[0].shape;
        if shape.len() != 2 {
            errors.push(format!(
                "Expected 2D face embedding output [N, 512] or [N, 128], found {}.",
                format_shape(shape)
            ));
        } else {
            let dim = shape[1];
            if dim == ARCFACE_EMBEDDING_DIM as i64 {
                detected_dim = Some(ARCFACE_EMBEDDING_DIM);
            } else if dim == SFACE_EMBEDDING_DIM as i64 {
                detected_dim = Some(SFACE_EMBEDDING_DIM);
            } else {
                errors.push(format!(
                    "Unsupported embedding size {dim}-D (Kroma supports 512-D ArcFace/InsightFace or 128-D SFace embeddings)."
                ));
            }
        }
    }

    if !errors.is_empty() {
        return ModelInspectionReport {
            compatibility: ModelCompatibility::Incompatible,
            summary: "Incompatible face recognition model.".to_string(),
            warnings: errors,
            detected_label: friendly_stem(file_name),
            input_shape,
            output_shape,
            embedding_dim: detected_dim,
            file_size_bytes,
        };
    }

    let dim = detected_dim.unwrap_or(ARCFACE_EMBEDDING_DIM);
    let lower = file_name.to_ascii_lowercase();
    let recognized_family = KNOWN_RECOGNIZER_HINTS.iter().any(|h| lower.contains(h));

    if recognized_family {
        return ModelInspectionReport {
            compatibility: ModelCompatibility::Compatible,
            summary: format!("Verified {dim}-D face recognition model (112×112 aligned input)."),
            warnings: Vec::new(),
            detected_label: friendly_recognizer_label(file_name, dim),
            input_shape,
            output_shape,
            embedding_dim: Some(dim),
            file_size_bytes,
        };
    }

    let expected_norm = if dim == ARCFACE_EMBEDDING_DIM {
        "RGB normalized to [-1, 1]"
    } else {
        "RGB in [0..255]"
    };
    ModelInspectionReport {
        compatibility: ModelCompatibility::Warning,
        summary: format!(
            "Tensor shape matches {dim}-D face recognition, but model family is unrecognized."
        ),
        warnings: vec![format!(
            "Kroma cannot verify from the ONNX graph alone that this model was trained for 112×112 aligned faces with {expected_norm}. If grouping quality drops, switch back to Default."
        )],
        detected_label: format!("{} ({dim}-D)", friendly_stem(file_name)),
        input_shape,
        output_shape,
        embedding_dim: Some(dim),
        file_size_bytes,
    }
}

fn format_shape(shape: &[i64]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }
    let dims: Vec<String> = shape
        .iter()
        .map(|&d| if d < 0 { "N".to_string() } else { d.to_string() })
        .collect();
    format!("[{}]", dims.join(", "))
}

fn friendly_stem(file_name: &str) -> String {
    Path::new(file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file_name)
        .to_string()
}

fn friendly_detector_label(file_name: &str) -> String {
    let lower = file_name.to_ascii_lowercase();
    if lower.contains("2026") {
        "YuNet 2026".to_string()
    } else if lower.contains("2023") {
        "YuNet 2023".to_string()
    } else if lower.contains("yunet") {
        "YuNet".to_string()
    } else {
        friendly_stem(file_name)
    }
}

fn friendly_recognizer_label(file_name: &str, dim: usize) -> String {
    let lower = file_name.to_ascii_lowercase();
    if lower.contains("w600k_r50") {
        format!("ArcFace R50 {dim}-D")
    } else if lower.contains("w600k_mbf") || lower.contains("mobilefacenet") {
        format!("ArcFace MobileFaceNet {dim}-D")
    } else if lower.contains("glintr100") {
        format!("GlintR100 {dim}-D")
    } else if lower.contains("glintr") {
        format!("GlintR {dim}-D")
    } else if lower.contains("adaface") {
        format!("AdaFace {dim}-D")
    } else if lower.contains("arcface") {
        format!("ArcFace {dim}-D")
    } else if lower.contains("sface") {
        format!("SFace {dim}-D")
    } else {
        format!("{} ({dim}-D)", friendly_stem(file_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizer_contract_accepts_arcface_warns_unknown_and_rejects_classifiers() {
        let valid_input = vec![TensorSignature {
            name: "input.1".to_string(),
            shape: vec![-1, 3, 112, 112],
        }];
        let valid_512 = vec![TensorSignature {
            name: "embedding".to_string(),
            shape: vec![-1, 512],
        }];

        let arcface = evaluate_recognizer_contract("w600k_r50.onnx", 174_000_000, &valid_input, &valid_512);
        assert_eq!(arcface.compatibility, ModelCompatibility::Compatible);
        assert_eq!(arcface.embedding_dim, Some(512));

        let unknown = evaluate_recognizer_contract("custom_encoder.onnx", 50_000_000, &valid_input, &valid_512);
        assert_eq!(unknown.compatibility, ModelCompatibility::Warning);
        assert!(!unknown.warnings.is_empty());

        let imagenet = evaluate_recognizer_contract(
            "resnet50_imagenet.onnx",
            98_000_000,
            &[TensorSignature {
                name: "data".to_string(),
                shape: vec![1, 3, 224, 224],
            }],
            &[TensorSignature {
                name: "logits".to_string(),
                shape: vec![1, 1000],
            }],
        );
        assert_eq!(imagenet.compatibility, ModelCompatibility::Incompatible);
        assert_eq!(imagenet.warnings.len(), 2);
    }

    #[test]
    fn detector_contract_requires_all_12_yunet_heads() {
        let yunet_in = vec![TensorSignature {
            name: "input".to_string(),
            shape: vec![1, 3, 640, 640],
        }];
        let yunet_outs: Vec<TensorSignature> = REQUIRED_YUNET_OUTPUTS
            .iter()
            .map(|&name| TensorSignature {
                name: name.to_string(),
                shape: vec![1, 1600, 1],
            })
            .collect();

        let ok = evaluate_detector_contract("face_detection_yunet_2023mar.onnx", 230_000, &yunet_in, &yunet_outs);
        assert_eq!(ok.compatibility, ModelCompatibility::Compatible);

        let yolo = evaluate_detector_contract(
            "yolov8n-face.onnx",
            12_000_000,
            &yunet_in,
            &[TensorSignature {
                name: "output0".to_string(),
                shape: vec![1, 20, 8400],
            }],
        );
        assert_eq!(yolo.compatibility, ModelCompatibility::Incompatible);
    }

    #[test]
    fn segmenter_contract_accepts_rmbg_modnet_and_rejects_1d_embeddings() {
        let seg_in = vec![TensorSignature {
            name: "input".to_string(),
            shape: vec![1, 3, 512, 512],
        }];
        let seg_out = vec![TensorSignature {
            name: "output".to_string(),
            shape: vec![1, 1, 512, 512],
        }];
        let ok = evaluate_segmenter_contract("modnet_photographic_portrait.onnx", 25_000_000, &seg_in, &seg_out);
        assert_eq!(ok.compatibility, ModelCompatibility::Compatible);

        let bad = evaluate_segmenter_contract(
            "w600k_r50.onnx",
            174_000_000,
            &seg_in,
            &[TensorSignature {
                name: "emb".to_string(),
                shape: vec![1, 512],
            }],
        );
        assert_eq!(bad.compatibility, ModelCompatibility::Incompatible);
    }
}
