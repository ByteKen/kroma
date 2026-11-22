use crate::error::AppError;
use kroma_domain::develop::NormalizedRect;
use kroma_domain::identity::{BoundingBox, FaceEmbedding, YuNetLandmarks};

/// Raw face detection returned by a YuNet-compatible detector before embedding extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct RawFaceDetection {
    pub bounding_box: BoundingBox,
    pub landmarks: YuNetLandmarks,
    pub detection_confidence: f32,
}

/// Optional whiteboard / placard OCR reading produced when Slate scanning is enabled.
#[derive(Debug, Clone, PartialEq)]
pub struct SlateOcrReading {
    pub subject_name: String,
    pub subject_code: Option<String>,
    pub ocr_confidence: f32,
    pub whiteboard_region: Option<NormalizedRect>,
}

/// Inward-owned port for face & 5-point landmark detection (implemented by `YuNetOnnxDetector`).
pub trait FaceDetectorPort: Send + Sync {
    fn detect_faces(&self, source_path: &str) -> Result<Vec<RawFaceDetection>, AppError>;
}

/// Inward-owned port for 128-D face identity embeddings (implemented by `SFaceOnnxRecognizer`).
pub trait FaceRecognizerPort: Send + Sync {
    fn extract_embedding(
        &self,
        source_path: &str,
        detection: &RawFaceDetection,
    ) -> Result<FaceEmbedding, AppError>;
}

/// Optional inward-owned port for reading handwritten/printed whiteboard slates.
/// Photographers can disable or omit this completely for pure face-clustering workflows!
pub trait SlateReaderPort: Send + Sync {
    fn read_slate(&self, source_path: &str) -> Result<Option<SlateOcrReading>, AppError>;
}
